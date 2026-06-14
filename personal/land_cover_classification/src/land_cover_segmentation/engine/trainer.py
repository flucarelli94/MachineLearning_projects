"""Plain-PyTorch training loop for semantic segmentation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from land_cover_segmentation.config import Config, dump
from land_cover_segmentation.dataset.loveda import LoveDADataModule
from land_cover_segmentation.engine.callbacks import EarlyStopping, JSONLLogger
from land_cover_segmentation.engine.checkpoint import CheckpointIO
from land_cover_segmentation.engine.evaluator import (
    evaluate_loader,
    metrics_from_confusion,
    resolve_device,
)
from land_cover_segmentation.engine.losses import DiceCELoss
from land_cover_segmentation.engine.metrics import StreamingConfusionMatrix
from land_cover_segmentation.utils import seed_everything


class Trainer:
    """Train a segmentation model with AdamW, AMP, and streaming val metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation network producing `(B, C, H, W)` logits.
    cfg : Config
        Resolved project configuration.
    datamodule : LoveDADataModule
        Prepared data module (`prepare_data` / `setup` are called by `fit`).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        datamodule: LoveDADataModule,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.datamodule = datamodule
        self.device = resolve_device(cfg.run.device)
        self.run_dir = Path(cfg.train.artifacts_root) / cfg.run.output_name

    def fit(self) -> dict[str, Any]:
        """Run the full training schedule and write checkpoints.

        Returns
        -------
        dict
            Keys: `run_dir`, `best_val_miou`, `epochs_run`, `stopped_early`.
        """
        seed_everything(self.cfg.run.seed, deterministic=self.cfg.run.deterministic)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump(self.cfg, self.run_dir / "config.yaml")

        self.datamodule.prepare_data()
        self.datamodule.setup()

        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()

        class_weights = (
            _load_or_compute_class_weights(self.cfg, self.datamodule, self.run_dir)
            if self.cfg.loss.use_class_weights
            else None
        )
        loss_fn = DiceCELoss(
            ignore_index=self.cfg.data.ignore_index,
            class_weights=class_weights,
        ).to(self.device)

        self.model.to(self.device)
        param_groups = _build_param_groups(self.model, self.cfg)
        optimizer = AdamW(
            param_groups,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, self.cfg, len(param_groups))
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        autocast_device = self.device.type

        early_stopping = EarlyStopping(patience=self.cfg.train.patience)
        jsonl_logger = JSONLLogger(self.run_dir / "run.jsonl")
        checkpoint_io = CheckpointIO(self.cfg, self.datamodule, self.run_dir)

        best_val_miou = float("-inf")
        epochs_run = 0
        stopped_early = False
        fit_start = time.perf_counter()

        for epoch in range(self.cfg.train.epochs):
            epoch_start = time.perf_counter()
            epochs_run = epoch + 1
            train_metrics = self._train_epoch(
                train_loader,
                loss_fn,
                optimizer,
                scaler,
                autocast_device,
                epoch,
            )
            val_metrics = self._validate_epoch(val_loader, loss_fn)
            scheduler.step()

            val_miou = val_metrics["miou"]
            is_best = val_miou > best_val_miou
            if is_best:
                best_val_miou = val_miou
                checkpoint_io.save(
                    self.run_dir / "best.pth",
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    best_val_miou=best_val_miou,
                    is_best=True,
                )
            checkpoint_io.save(
                self.run_dir / "last.pth",
                model=self.model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                best_val_miou=best_val_miou,
            )

            lrs = [group["lr"] for group in optimizer.param_groups]
            jsonl_logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_miou": train_metrics["miou"],
                    "val_miou": val_metrics["miou"],
                    "lr": lrs,
                    "elapsed_s": time.perf_counter() - epoch_start,
                    "per_class_iou": val_metrics["per_class_iou"],
                }
            )
            _log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=optimizer,
            )

            if early_stopping.step(val_miou):
                stopped_early = True
                break

        return {
            "run_dir": self.run_dir,
            "best_val_miou": best_val_miou,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "elapsed_s": time.perf_counter() - fit_start,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: DiceCELoss,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        autocast_device: str,
        epoch: int,
    ) -> dict[str, Any]:
        self.model.train()
        cm = StreamingConfusionMatrix(
            self.cfg.data.num_classes, self.cfg.data.ignore_index
        )
        loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"train {epoch + 1}/{self.cfg.train.epochs}")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=autocast_device,
                enabled=autocast_device == "cuda",
            ):
                logits = self.model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            num_batches += 1
            preds = logits.argmax(dim=1)
            cm.update(preds, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return metrics_from_confusion(cm, loss_sum / max(num_batches, 1))

    @torch.no_grad()
    def _validate_epoch(
        self,
        loader: DataLoader,
        loss_fn: DiceCELoss,
    ) -> dict[str, Any]:
        return evaluate_loader(
            self.model,
            loader,
            loss_fn,
            self.cfg,
            self.device,
            desc="val",
        )


def _build_param_groups(model: nn.Module, cfg: Config) -> list[dict[str, Any]]:
    if cfg.model.source == "smp" and hasattr(model, "encoder"):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        encoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for param in model.parameters():
            if id(param) in encoder_ids:
                encoder_params.append(param)
            else:
                other_params.append(param)
        return [
            {"params": encoder_params},
            {"params": other_params},
        ]
    return [{"params": list(model.parameters())}]


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    num_groups: int,
) -> LambdaLR:
    warmup = cfg.optim.encoder_lr_warmup_epochs
    total = cfg.train.epochs
    enc_scale = cfg.optim.encoder_lr_scale
    cosine_span = max(total - warmup, 1)

    def _cosine_factor(epoch: int) -> float:
        if epoch < warmup:
            return 1.0
        progress = (epoch - warmup) / cosine_span
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if num_groups == 2:

        def encoder_lambda(epoch: int) -> float:
            if epoch < warmup:
                return max((epoch + 1) / max(warmup, 1), 1e-8) * enc_scale
            return enc_scale * _cosine_factor(epoch)

        def decoder_lambda(epoch: int) -> float:
            if epoch < warmup:
                return 1.0
            return _cosine_factor(epoch)

        return LambdaLR(optimizer, lr_lambda=[encoder_lambda, decoder_lambda])

    def single_lambda(epoch: int) -> float:
        if epoch < warmup:
            return 1.0
        return _cosine_factor(epoch)

    return LambdaLR(optimizer, lr_lambda=single_lambda)


def _load_or_compute_class_weights(
    cfg: Config,
    datamodule: LoveDADataModule,
    run_dir: Path,
) -> torch.Tensor:
    cache_path = run_dir / "class_weights.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return torch.tensor(raw["weights"], dtype=torch.float32)

    counts = _class_pixel_counts(cfg, datamodule)
    num_classes = cfg.data.num_classes
    total = counts.sum()
    safe_counts = torch.clamp(counts, min=1.0)
    weights = (total / (num_classes * safe_counts)) ** 0.5
    weights = weights / weights.mean()

    cache_path.write_text(
        json.dumps(
            {
                "counts": counts.tolist(),
                "weights": weights.tolist(),
            },
            indent=2,
        )
    )
    return weights


def _class_pixel_counts(cfg: Config, datamodule: LoveDADataModule) -> torch.Tensor:
    datamodule._require_setup()
    raw = datamodule._train_ds_raw
    if raw is None:
        raise RuntimeError("Train split is not initialized; call setup() first.")

    counts = torch.zeros(cfg.data.num_classes, dtype=torch.float64)
    for idx in range(len(raw)):
        mask = raw[idx]["mask"].reshape(-1).to(torch.int64)
        valid = mask != cfg.data.nodata_label
        if not valid.any():
            continue
        labels = mask[valid]
        counts += torch.bincount(labels, minlength=cfg.data.num_classes).double()
    return counts


def _log_epoch(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> None:
    lrs = [group["lr"] for group in optimizer.param_groups]
    lr_str = ", ".join(f"{lr:.2e}" for lr in lrs)
    print(
        f"epoch {epoch + 1}: "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_miou={train_metrics['miou']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_miou={val_metrics['miou']:.4f} "
        f"lr=[{lr_str}]"
    )


__all__ = ["Trainer"]
