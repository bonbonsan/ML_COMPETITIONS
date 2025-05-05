from typing import List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from my_library.ensembles.ensemble_base import ArrayLike, BaseEnsembler


class StackingEnsembler(BaseEnsembler):
    """
    Ensemble class that performs stacking with a meta-model.

    This version allows combining base model predictions with additional features
    for training the meta-model.
    """

    def __init__(self, meta_model=None):
        super().__init__()
        self.meta_model = meta_model if meta_model is not None else LinearRegression()

    def ensemble(self,
                 predictions: List[ArrayLike],
                 X_meta_features: Optional[ArrayLike] = None,
                 y_true: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Train the meta-model using base model predictions and optional additional features.

        Args:
            predictions (List[ArrayLike]): List of base model predictions.
            X_meta_features (ArrayLike, optional): Additional features for the meta-model.
            y_true (ArrayLike, optional): Ground truth values for training the meta-model.

        Returns:
            np.ndarray: Final prediction from the trained meta-model.

        Raises:
            ValueError: If `y_true` is not provided.
        """
        base_preds = np.stack([self._to_numpy(p) for p in predictions], axis=1)

        if X_meta_features is not None:
            X_meta = np.hstack([base_preds, self._to_numpy(X_meta_features)])
        else:
            X_meta = base_preds

        if y_true is None:
            raise ValueError("y_true must be provided for training the meta model.")

        y_true_np = self._to_numpy(y_true)
        self.meta_model.fit(X_meta, y_true_np)
        y_pred = self.meta_model.predict(X_meta)
        self.final_prediction = y_pred
        return y_pred
