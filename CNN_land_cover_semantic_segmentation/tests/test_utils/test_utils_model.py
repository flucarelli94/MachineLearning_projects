import torch

from land_cover_segmentation.utils.model import resolve_device, seed_everything


def test_seed_everything_runs():
    seed_everything(0)
    seed_everything(0, deterministic=True)


def test_resolve_device_auto():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)


def test_resolve_device_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
