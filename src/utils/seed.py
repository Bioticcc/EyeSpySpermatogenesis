from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, disable_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    if disable_cudnn and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]


def worker_init_fn(worker_id: int) -> None:
    # Ensures dataloader workers have different seeds derived from the global seed.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
