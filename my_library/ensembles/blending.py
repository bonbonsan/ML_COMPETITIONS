from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression

from my_library.ensembles.ensemble_base import ArrayLike, BaseEnsembler


class BlendingEnsembler(BaseEnsembler):
    """
    Ensemble class that performs blending using a holdout set.

    This method splits the data into training and holdout sets.
    Base models are trained on the training set, and their predictions
    on the holdout set are used to train a meta-model.
    """

    def __init__(self, meta_model=None):
        super().__init__()
        self.meta_model = meta_model if meta_model is not None else LinearRegression()

    def ensemble(self,
                 base_predictions: List[ArrayLike],
                 y_true: ArrayLike) -> np.ndarray:
        """
        Train the meta-model using base model predictions on a holdout set.

        Args:
            base_predictions (List[ArrayLike]): List of base model predictions on the holdout set.
            y_true (ArrayLike): Ground truth values for the holdout set.

        Returns:
            np.ndarray: Final prediction from the trained meta-model.
        """
        X_meta = np.stack([self._to_numpy(p) for p in base_predictions], axis=1)
        y_true_np = self._to_numpy(y_true)
        self.meta_model.fit(X_meta, y_true_np)
        y_pred = self.meta_model.predict(X_meta)
        self.final_prediction = y_pred
        return y_pred
