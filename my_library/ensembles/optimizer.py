from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize

from my_library.ensembles.ensemble_base import ArrayLike, MetricFunction


class WeightOptimizer:
    """
    Optimizes ensemble weights by minimizing (or maximizing) a given metric
    over holdout predictions.

    Uses scipy.optimize.minimize with optional constraints and bounds to find
    the best linear combination weights for multiple model predictions.

    Attributes:
        metric_fn (MetricFunction): Function to compute the metric given
            true and predicted values.
        loss_greater_is_better (bool): If True, the optimizer will maximize
            the metric function instead of minimizing it.
    """
    def __init__(self,
                 metric_fn: MetricFunction,
                 loss_greater_is_better: bool = False):
        """
        Initialize the WeightOptimizer.

        Args:
            metric_fn (MetricFunction): A function that accepts two numpy arrays
                (y_true, y_pred) and returns a float metric.
            loss_greater_is_better (bool): If True, treats the metric as a score
                to maximize; if False, treats it as a loss to minimize.
        """
        self.metric_fn = metric_fn
        self.loss_greater_is_better = loss_greater_is_better

    def optimize(self,
                 predictions: List[ArrayLike],
                 y_true: ArrayLike,
                 constraint_sum_to_one: bool = True,
                 bounds: Optional[List[Tuple[float, float]]] = None) -> List[float]:
        """
        Optimize weights for combining multiple model predictions.

        Args:
            predictions (List[ArrayLike]): List of base model predictions, each
                in array-like format (numpy, pandas, polars, list/tuple).
            y_true (ArrayLike): Ground truth values corresponding to predictions.
            constraint_sum_to_one (bool): If True, enforces the sum of weights
                equals 1 by normalizing during optimization and after.
            bounds (Optional[List[Tuple[float, float]]]): List of (min, max) bounds
                for each weight. Defaults to [(0.0, 1.0)] for each model if None.

        Returns:
            List[float]: Optimized weights for each model, normalized to sum to one
                if `constraint_sum_to_one` is True.
        """
        n_models = len(predictions)
        y_true_np = self._to_numpy(y_true)
        preds_np = [self._to_numpy(p) for p in predictions]

        def loss_fn(weights):
            weights = np.array(weights)
            if constraint_sum_to_one:
                weights = weights / weights.sum()

            ensemble_pred = np.zeros_like(preds_np[0], dtype=np.float64)
            for p, w in zip(preds_np, weights, strict=False):
                ensemble_pred += p * w

            score = self.metric_fn(y_true_np, ensemble_pred)
            return -score if self.loss_greater_is_better else score

        if bounds is None:
            bounds = [(0.0, 1.0)] * n_models

        init_weights = np.full(n_models, 1.0 / n_models)

        result = minimize(
            loss_fn,
            x0=init_weights,
            bounds=bounds,
            method="SLSQP"
        )

        final_weights = result.x
        if constraint_sum_to_one:
            final_weights /= final_weights.sum()

        return final_weights.tolist()

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
