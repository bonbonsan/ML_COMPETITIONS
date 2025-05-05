import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]
MetricFunction = Callable[[np.ndarray, np.ndarray], float]


class BaseEnsembler(ABC):
    """
    Abstract base class for ensemble models.

    Handles storage of the final ensemble predictions, evaluation
    against ground truth with a provided metric, and saving results
    to disk.
    """
    def __init__(self):
        """
        Initialize the BaseEnsembler.

        Sets `final_prediction` to None until `ensemble` is called.
        """
        self.final_prediction: Optional[np.ndarray] = None

    @abstractmethod
    def ensemble(self, predictions: List[ArrayLike]) -> np.ndarray:
        """
        Perform the ensembling operation on a list of predictions.

        Subclasses must implement this method to combine the provided
        `predictions` into a single numpy array. The result should
        be stored in `self.final_prediction` and also returned.

        Args:
            predictions (List[ArrayLike]): A list of predictions from
                base models, each as numpy array, pandas or polars types.

        Returns:
            np.ndarray: The combined ensemble prediction.
        """
        pass

    def evaluate(self, y_true: ArrayLike, metric_fn: MetricFunction) -> float:
        """
        Evaluate the stored ensemble prediction against the ground truth.

        Args:
            y_true (ArrayLike): True labels or values for comparison.
            metric_fn (MetricFunction): A function that takes two numpy
                arrays (y_true, y_pred) and returns a float metric.

        Returns:
            float: The computed metric value.

        Raises:
            ValueError: If `ensemble` has not been called and no predictions
                are stored.
        """
        if self.final_prediction is None:
            raise ValueError("No ensemble prediction found. Please run `ensemble()` first.")

        y_true_np = self._to_numpy(y_true)
        return metric_fn(y_true_np, self.final_prediction)

    def save_prediction(self, path: str):
        """
        Save the stored ensemble predictions to a CSV file.

        Args:
            path (str): File path where the CSV will be saved. If the
                directory does not exist, it will be created.

        Raises:
            ValueError: If `ensemble` has not been called and no predictions
                are stored.
        """
        if self.final_prediction is None:
            raise ValueError("No ensemble prediction found. Please run `ensemble()` first.")

        df = pd.DataFrame(self.final_prediction)
        # only create directory if path contains one
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df.to_csv(path, index=False)

    def _to_numpy(self, arr: ArrayLike) -> np.ndarray:
        """
        Convert various ArrayLike inputs to a numpy array.

        Supports numpy arrays, pandas DataFrame/Series, polars
        DataFrame/Series, and Python lists or tuples.

        Args:
            arr (ArrayLike): The input array-like data.

        Returns:
            np.ndarray: The converted numpy array.

        Raises:
            TypeError: If the input type is not supported.
        """
        if isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, (pd.DataFrame, pd.Series)):
            return arr.values
        elif isinstance(arr, (pl.DataFrame, pl.Series)):
            return arr.to_numpy()
        elif isinstance(arr, (list, tuple)):
            return np.array(arr)
        else:
            raise TypeError(f"Unsupported array type: {type(arr)}")
