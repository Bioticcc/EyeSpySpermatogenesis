from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim != 3:
        raise ValueError(f"Expected HWC array, got {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def build_transforms(train: bool = True, image_size: int = 224, standardize: bool = True) -> Callable[[np.ndarray], torch.Tensor]:
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]) if standardize else torch.nn.Identity()
    augmentations = []
    if train:
        augmentations.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=20),
            ]
        )
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            *augmentations,
            T.ToTensor(),  # converts to CHW float [0,1]
            normalize,
        ]
    )
    return transform
