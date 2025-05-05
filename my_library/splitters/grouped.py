from sklearn.model_selection import GroupKFold

from my_library.splitters.splitter_base import BaseFoldSplitter


class GroupKFoldSplitter(BaseFoldSplitter):
    """
    Group-aware K-Fold splitter using scikit-learn's GroupKFold.

    Splits the dataset into training and validation sets based on group labels,
    ensuring that the same group is not represented in both train and validation.
    """

    def __init__(self, n_splits: int = 5):
        """
        Initialize the group K-Fold splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds to generate.
        """
        self.splitter = GroupKFold(n_splits=n_splits)

    def split(self, X, y, groups):
        """
        Generate train/validation splits based on group labels.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : pd.Series or array-like
            Target vector.
        groups : pd.Series or array-like
            Group labels for the samples used to ensure group-wise splits.

        Returns
        -------
        List of tuple
            A list containing `n_splits` elements, each of the form:
            ((X_train, y_train), (X_valid, y_valid))
        """
        return [
            ((X.iloc[train_idx], y.iloc[train_idx]),
             (X.iloc[val_idx],   y.iloc[val_idx]))
            for train_idx, val_idx in self.splitter.split(X, y, groups)
        ]
