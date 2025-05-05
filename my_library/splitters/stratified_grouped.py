from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from my_library.splitters.splitter_base import BaseFoldSplitter


class StratifiedGroupKFoldSplitter(BaseFoldSplitter):
    """
    Stratified Group K-Fold splitter.

    Splits the dataset based on group labels while preserving the target’s
    class distribution across folds. Ensures that all samples from the same
    group are kept together in either the training or validation set.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize the stratified group K-Fold splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds.
        random_state : int, default=42
            Seed for random shuffling of group-level stratification.
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series
    ) -> List[Tuple[Tuple[pd.DataFrame, pd.Series],
                    Tuple[pd.DataFrame, pd.Series]]]:
        """
        Generate stratified train/validation splits by group.

        Each group is treated as a single unit for stratification, and the
        class distribution of the target y is preserved across folds.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : pd.Series or array-like
            Target vector used for stratification.
        groups : pd.Series or array-like
            Group labels indicating which samples belong to the same group.

        Returns
        -------
        List of tuples
            A list of length `n_splits`. Each element is:
            ((X_train, y_train), (X_valid, y_valid))
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: got {len(X)} vs {len(y)}")
        if len(X) != len(groups):
            raise ValueError(
                f"X and groups must have the same length: got {len(X)} vs {len(groups)}"
                )
        
        # Convert to one sample per group for stratification
        df = pd.DataFrame({"y": y, "group": groups}).drop_duplicates("group")
        stratify_labels = df["y"].values
        group_labels = df["group"].values

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        result = []

        for train_groups_idx, val_groups_idx in skf.split(
            np.zeros(len(group_labels)),
            stratify_labels
        ):
            train_groups = set(group_labels[train_groups_idx])
            val_groups = set(group_labels[val_groups_idx])

            train_idx = X.index[groups.isin(train_groups)]
            val_idx = X.index[groups.isin(val_groups)]

            result.append(
                ((X.loc[train_idx], y.loc[train_idx]),
                 (X.loc[val_idx],   y.loc[val_idx]))
            )
        return result
