from typing import List

import numpy as np

from my_library.ensembles.ensemble_base import ArrayLike, BaseEnsembler


class CVWeightAveragingEnsembler(BaseEnsembler):
    """
    Ensemble class that performs CV-score-based weighted averaging.

    Each model's prediction is weighted inversely proportional to its validation loss.
    This encourages better-performing models to contribute more to the ensemble.

    Attributes:
        weights (np.ndarray): Calculated weights based on validation scores.
        final_prediction (np.ndarray): Weighted average of predictions.
    """

    def __init__(self):
        super().__init__()

    def ensemble(self,
                 predictions: List[ArrayLike],
                 y_true: ArrayLike,
                 metric_fn) -> np.ndarray:
        """
        Compute weighted average of predictions using inverse CV scores.

        Args:
            predictions (List[ArrayLike]): List of base model predictions.
            y_true (ArrayLike): Ground truth for scoring.
            metric_fn (Callable): Scoring function (e.g., mean_squared_error, log_loss).

        Returns:
            np.ndarray: Weighted average prediction based on CV performance.
        """
        np_preds = [self._to_numpy(p) for p in predictions]
        y_true_np = self._to_numpy(y_true)

        scores = np.array([
            metric_fn(y_true_np, p) for p in np_preds
        ])

        inv_scores = 1.0 / (scores + 1e-8)
        weights = inv_scores / inv_scores.sum()

        stacked_preds = np.stack(np_preds, axis=0)
        weighted_pred = np.average(stacked_preds, axis=0, weights=weights)
        self.weights = weights
        self.final_prediction = weighted_pred
        return weighted_pred
