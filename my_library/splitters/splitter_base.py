from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd


class BaseFoldSplitter(ABC):
    """
    Abstract base class for splitting tabular data into training and testing folds.

    Subclasses must implement the `split` method to generate train/test splits
    given features `X`, target `y`, and optionally `groups`.

    Example usage:
        splitter = SomeFoldSplitter(n_splits=5, random_state=42)
        folds = splitter.split(X, y)
        for (X_train, y_train), (X_test, y_test) in folds:
            # train/test your model
    """

    @abstractmethod
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
    ) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
        """
        Generate a list of train/test splits.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        groups : Optional[pd.Series], default=None
            Group labels for the samples used while splitting the dataset into folds
            (e.g., for group k-fold).

        Returns
        -------
        List of tuples
            Each element is a tuple of the form:
                ((X_train, y_train), (X_test, y_test))
        """
        pass
