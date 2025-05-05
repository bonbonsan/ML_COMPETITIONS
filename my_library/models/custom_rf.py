import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.rf_configs import RandomForestConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomRandomForest(CustomModelInterface, BaseEstimator):
    """
    Sklearn-based Random Forest wrapper with a unified interface.

    Wraps RandomForestClassifier and RandomForestRegressor and provides:
      - fit/predict/predict_proba following CustomModelInterface
      - feature importance via permutation importance
      - model persistence via joblib
      - standardized logging
    """

    def __init__(self, config: RandomForestConfig):
        """
        Initialize the CustomRandomForest with the given configuration.

        Args:
            config (RandomForestConfig): Configuration object containing:
                - model_name (str)
                - task_type (Literal['classification','regression'])
                - use_gpu (bool)        # not natively supported by sklearn
                - save_log (bool)
                - params (dict)         # hyperparameters for RF
        """
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.use_gpu = config.use_gpu
        self.params = config.params or {}

        self.logger = Logger(self.__class__.__name__, save_to_file=config.save_log).get_logger()
        self.logger.info(f"Initialized {self.__class__.__name__} with model_name={self.model_name}")

        self.model = None
        self.used_features = None
        self.target_name = "target"
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None

    def build_model(self):
        """
        Instantiate the underlying sklearn RandomForest model based on task type.
        """
        self.logger.info("Building RandomForest model.")
        if self.task_type == "classification":
            self.model = RandomForestClassifier(**self.params)
        elif self.task_type == "regression":
            self.model = RandomForestRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, fit_config: FitConfig) -> None:
        """
        Train the random forest model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            fit_config (FitConfig): May specify subset of features.
        """
        self.target_name = y.name or self.target_name
        self.logger.info("Starting training of RandomForest model.")

        # select features
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[self.used_features]
        else:
            self.used_features = X.columns.tolist()
            X_train = X

        # store for permutation importance
        self._X_train = X_train.copy()
        self._y_train = y.copy()

        # build and fit
        self.build_model()
        self.model.fit(X_train, y)
        self.logger.info("RandomForest training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict labels or values.

        Returns:
            pd.Series: Predictions with same index as X.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        X_eval = X[self.used_features] if self.used_features else X
        preds = self.model.predict(X_eval)
        return pd.Series(preds, index=X.index, name=self.target_name)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for classification tasks.

        Returns:
            pd.DataFrame: Columns named '{target_name}_{class}'.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{type(self.model)} does not support predict_proba().")

        X_eval = X[self.used_features] if self.used_features else X
        proba = self.model.predict_proba(X_eval)
        columns = [f"{self.target_name}_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=columns)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """
        Get top-N features by permutation importance.

        Args:
            top_n (int): Number of top features.

        Returns:
            List of (feature_name, importance) sorted descending.
        """
        self.logger.info(f"Computing permutation importance for top {top_n} features.")
        if self.model is None or self._X_train is None or self._y_train is None:
            raise ValueError("Model must be trained and training data available.")

        result = permutation_importance(
            self.model,
            self._X_train,
            self._y_train,
            n_repeats=5,
            random_state=42,
        )
        importances = result.importances_mean
        idxs = importances.argsort()[::-1][:top_n]
        return [(self.used_features[i], importances[i]) for i in idxs]

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk via joblib.

        Args:
            path (str): File path to save.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path (str): File path to load.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
