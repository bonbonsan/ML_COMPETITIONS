from sklearn.model_selection import train_test_split

from my_library.splitters.splitter_base import BaseFoldSplitter


class HoldoutSplitter(BaseFoldSplitter):
    """
    Simple hold-out splitter using scikit-learn's train_test_split.

    Splits the dataset into a single training set and validation set
    based on the specified test_size and random_state.
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the hold-out splitter.

        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of the dataset to include in the validation split.
        random_state : int, default=42
            Controls the shuffling applied to the data before splitting.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Generate a single train/validation split.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.
        y : pd.Series or array-like
            Target vector.
        groups : ignored
            This parameter is present to maintain interface compatibility
            with BaseFoldSplitter but not used in hold-out splitting.

        Returns
        -------
        List of tuple
            A list containing one element:
            ((X_train, y_train), (X_valid, y_valid))
        """
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return [((X_train, y_train), (X_valid, y_valid))]
