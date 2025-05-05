from sklearn.model_selection import KFold

from my_library.splitters.splitter_base import BaseFoldSplitter


class CrossValidationSplitter(BaseFoldSplitter):
    """
    Cross-validation splitter using scikit-learn's KFold.

    Splits the dataset into multiple training/validation folds
    based on the specified n_splits, shuffle, and random_state.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Initialize the cross-validation splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        random_state : int, default=42
            Controls the shuffling applied to the data before splitting.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        # Only use random_state when shuffle=True to avoid KFold errors
        self.random_state = random_state if shuffle else None

        # Configure KFold accordingly
        self.kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

    def split(self, X, y, groups=None):
        """
        Generate cross-validation splits.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : pd.Series or array-like
            Target vector.
        groups : array-like, default=None
            Group labels for the samples used while splitting the dataset.

        Returns
        -------
        List of tuple
            A list containing `n_splits` elements, each is:
            ((X_train, y_train), (X_valid, y_valid))
        """
        splits = []
        for train_idx, valid_idx in self.kf.split(X, y, groups):
            # slice X
            if hasattr(X, "iloc"):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            else:
                X_train, X_valid = X[train_idx], X[valid_idx]
            # slice y
            if hasattr(y, "iloc"):
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            else:
                y_train, y_valid = y[train_idx], y[valid_idx]

            splits.append(((X_train, y_train), (X_valid, y_valid)))
        return splits
