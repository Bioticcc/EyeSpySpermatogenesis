from pathlib import Path
import numpy as np
import tifffile

from src.utils.tiff_io import read_tiff


def test_read_tiff_uint16(tmp_path: Path) -> None:
    arr = (np.random.rand(16, 16) * 65535).astype(np.uint16)
    path = tmp_path / "sample.tiff"
    tifffile.imwrite(path, arr)
    out = read_tiff(path)
    assert out.dtype == np.float32
    assert out.shape == (16, 16, 3)
    assert out.max() <= 1.0


def test_read_tiff_multichannel(tmp_path: Path) -> None:
    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(path, arr)
    out = read_tiff(path)
    assert out.shape == (8, 8, 3)
