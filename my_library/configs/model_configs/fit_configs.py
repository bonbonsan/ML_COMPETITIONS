"""
fit_config.py

Defines FitConfig dataclass for unifying fit() parameters across different model implementations.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class FitConfig:
    """
    Configuration object for controlling the `fit()` behavior of custom models.

    This class encapsulates all training-time parameters required by various
    model types (e.g., gradient boosting, deep learning). All attributes are
    optional and default to None; users should specify only those relevant
    to the target algorithm.

    Attributes:
        feats (Optional[List[str]]):
            Names of feature columns to use for training. If None, all columns
            in the provided DataFrame will be used.
        eval_set (Optional[List[Tuple[pd.DataFrame, pd.Series]]]):
            List of validation datasets for early stopping. Each element is a
            tuple of (X_validation, y_validation). Only used by models that
            support evaluation sets (e.g., GBDT).
        early_stopping_rounds (Optional[int]):
            Number of rounds with no improvement before stopping training.
            Applicable to algorithms with built-in early stopping.
        epochs (Optional[int]):
            Number of training epochs for deep learning models. Ignored by
            algorithms that do not use multiple epochs.
        batch_size (Optional[int]):
            Batch size for training deep learning models. Ignored by algorithms
            that train in a single pass (e.g., gradient boosting).
    """
    feats: Optional[List[str]] = None
    eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None
    early_stopping_rounds: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
