from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile


def _scale_to_float(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        data = image.astype(np.float32)
        if data.max() > 1.0:
            data = data / float(data.max())
        return data
    if np.issubdtype(image.dtype, np.uint16):
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.uint8):
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.integer):
        max_val = float(np.iinfo(image.dtype).max)
        return image.astype(np.float32) / max_val
    return image.astype(np.float32)


def read_tiff(path: Path | str, page: int = 0, force_three_channels: bool = True) -> np.ndarray:
    """Read TIFF to float32 array in [0,1]. Defaults to first page, handles grayscale."""
    array = tifffile.imread(str(path), key=page)
    array = np.array(array)
    array = _scale_to_float(array)

    # handle grayscale or single-channel
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3:
        if array.shape[0] in (1, 3) and array.shape[0] != array.shape[1]:
            # channel-first -> move channel last
            array = np.moveaxis(array, 0, -1)
        if array.shape[2] == 1 and force_three_channels:
            array = np.repeat(array, 3, axis=2)
        elif array.shape[2] > 3:
            array = array[:, :, :3]
    else:
        raise ValueError(f"Unsupported TIFF shape: {array.shape}")
    return array.astype(np.float32)


def save_tiff(array: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)
