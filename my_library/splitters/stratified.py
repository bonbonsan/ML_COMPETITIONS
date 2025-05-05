from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from my_library.splitters.splitter_base import BaseFoldSplitter


class StratifiedKFoldSplitter(BaseFoldSplitter):
    """
    Stratified K-Fold splitter using scikit-learn's StratifiedKFold.

    Splits the dataset into training and validation sets while preserving
    the class distribution of the target in each fold.

    Note
    ----
    This splitter only uses the target `y` for stratification. The `groups`
    parameter is present solely for interface compatibility and has no effect.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the stratified K-Fold splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds to generate.
        shuffle : bool, default=True
            Whether to shuffle data before splitting.
        random_state : int, default=42
            Seed for the random number generator used when shuffling.
        """
        self.splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
    ) -> List[Tuple[Tuple[pd.DataFrame, pd.Series],
                    Tuple[pd.DataFrame, pd.Series]]]:
        """
        Generate stratified train/validation splits.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : pd.Series or array-like
            Target vector used for stratification.
        groups : ignored
            Included for interface compatibility but not used internally.

        Returns
        -------
        List of tuples
            A list of length `n_splits`, each element is:
            ((X_train, y_train), (X_valid, y_valid))
        """
        return [
            ((X.iloc[train_idx], y.iloc[train_idx]),
             (X.iloc[val_idx],   y.iloc[val_idx]))
            for train_idx, val_idx in self.splitter.split(X, y)
        ]
