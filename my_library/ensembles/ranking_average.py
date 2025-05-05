from typing import List

import numpy as np

from my_library.ensembles.ensemble_base import ArrayLike, BaseEnsembler


class RankAveragingEnsembler(BaseEnsembler):
    """
    Ensemble class that performs rank averaging.

    This method transforms each model's predictions into ranks and computes
    the average rank across models. Particularly useful in classification tasks
    where relative ordering (e.g., ROC-AUC) is more important than exact values.

    Attributes:
        final_prediction (np.ndarray): Averaged rank-based prediction.
    """

    def __init__(self):
        super().__init__()

    def ensemble(self, predictions: List[ArrayLike]) -> np.ndarray:
        """
        Compute ensemble prediction by averaging model-wise ranks.

        Args:
            predictions (List[ArrayLike]): List of predictions from each model.

        Returns:
            np.ndarray: Averaged rank-based ensemble prediction.
        """
        np_preds = [self._to_numpy(p) for p in predictions]
        ranks = [self._rankdata(p) for p in np_preds]
        stacked_ranks = np.stack(ranks, axis=0)
        averaged_rank = np.mean(stacked_ranks, axis=0)
        self.final_prediction = averaged_rank
        return averaged_rank

    def _rankdata(self, arr: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata
        return rankdata(arr, method='average')
