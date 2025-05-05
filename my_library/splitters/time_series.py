from typing import Optional

import pandas as pd

from my_library.splitters.splitter_base import BaseFoldSplitter


class TimeSeriesSplitter(BaseFoldSplitter):
    """
    Time series fold splitter supporting expanding and rolling windows.

    Splits temporal data into sequential train/validation folds, respecting
    the order of observations and preventing lookahead bias.

    - expanding: training window starts at the beginning and grows with each fold.
    - rolling  : training window moves forward with a fixed maximum size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        method: str = "expanding",  # "rolling" or "expanding"
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ):
        """
        Initialize the time series splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of sequential folds to generate.
        method : str, default='expanding'
            Windowing method: 'expanding' grows the training window each fold;
            'rolling' uses a fixed-size training window that moves forward.
        max_train_size : Optional[int], default=None
            Maximum number of samples in the training window when using 'rolling'.
            Ignored if method is 'expanding'.
        test_size : Optional[int], default=None
            Fixed size of the validation window. If None, computed as
            total_samples // (n_splits + 1).
        """
        assert method in ["expanding", "rolling"], "method must be 'expanding' or 'rolling'"
        self.n_splits = n_splits
        self.method = method
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X: pd.DataFrame, y: pd.Series, groups=None):
        """
        Generate time series train/validation splits.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix ordered by time.
        y : pd.Series or array-like
            Target vector aligned with X.
        groups : ignored
            Present for interface compatibility but not used here.

        Returns
        -------
        List of tuple
            List of folds, each as ((X_train, y_train), (X_valid, y_valid)).
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: got {len(X)} vs {len(y)}")
        
        n_samples = len(X)
        fold_sizes = self._compute_fold_boundaries(n_samples)

        result = []
        for train_end, test_start, test_end in fold_sizes:
            if self.method == "rolling" and self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:  # expanding
                train_start = 0

            train_idx = list(range(train_start, train_end))
            val_idx = list(range(test_start, test_end))

            result.append(
                ((X.iloc[train_idx], y.iloc[train_idx]),
                 (X.iloc[val_idx],   y.iloc[val_idx]))
            )
        return result

    def _compute_fold_boundaries(self, n_samples: int):
        """
        Compute indices for training and validation windows.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        List of tuple
            Each tuple is (train_end, test_start, test_end) for a fold.
        """
        test_size = self.test_size or n_samples // (self.n_splits + 1)
        fold_boundaries = []

        for i in range(self.n_splits):
            train_end = test_size * (i + 1)
            test_start = train_end
            test_end = test_start + test_size
            if test_end > n_samples:
                break
            fold_boundaries.append((train_end, test_start, test_end))

        return fold_boundaries
