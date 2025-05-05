import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the seed for reproducibility in random, NumPy, and PyTorch.

    This function sets seeds for Python's built-in `random` module,
    NumPy, and PyTorch (both CPU and CUDA if available).

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
