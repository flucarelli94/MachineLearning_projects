from land_cover_segmentation.utils.general import dir_size, hex_to_rgb, human_bytes


def test_human_bytes():
    assert human_bytes(1024) == "1.0 KiB"
    assert human_bytes(1024 * 1024) == "1.0 MiB"
    assert human_bytes(1024 * 1024 * 1024) == "1.0 GiB"
    assert human_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TiB"


def test_dir_size():
    size = dir_size(".")
    assert isinstance(size, int)
    assert size > 0


def test_hex_to_rgb():
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000FF") == (0, 0, 255)
