from typing import List, Optional

import numpy as np

from my_library.ensembles.ensemble_base import ArrayLike, BaseEnsembler


class WeightedAverageEnsembler(BaseEnsembler):
    def __init__(self,
                 weights: Optional[List[float]] = None,
                 n_models: Optional[int] = None):
        """
        Args:
            weights: List of weights or None.
            n_models: Number of models. Required if weights is None.
        """
        super().__init__()

        if weights is not None:
            self.weights = np.array(weights)
        elif n_models is not None:
            self.weights = np.full(n_models, 1.0 / n_models)
        else:
            # postpone initialization until ensemble() is called
            self.weights = None

    def ensemble(self, predictions: List[ArrayLike]) -> np.ndarray:
        np_preds = [self._to_numpy(p) for p in predictions]
        stacked_preds = np.stack(np_preds, axis=0)  # Shape: (n_models, n_samples)

        n_models = stacked_preds.shape[0]

        if self.weights is None:
            self.weights = np.full(n_models, 1.0 / n_models)
        elif len(self.weights) != n_models:
            raise ValueError("Length of weights does not match number of predictions.")

        weighted_pred = np.average(stacked_preds, axis=0, weights=self.weights)
        self.final_prediction = weighted_pred
        return weighted_pred
