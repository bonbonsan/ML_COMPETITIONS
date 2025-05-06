import os

import torch
from dotenv import load_dotenv

# Load .env file once at import
load_dotenv()


def get_env_bool(key: str, default: str = "False") -> bool:
    """Convert environment variable to boolean."""
    return os.getenv(key, default).lower() == "true"


def use_gpu_enabled() -> bool:
    """Return True if USE_GPU=True is set in .env."""
    return get_env_bool("USE_GPU")

def get_device(use_gpu: bool) -> str:
    """Return the device to be used for PyTorch operations."""
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
