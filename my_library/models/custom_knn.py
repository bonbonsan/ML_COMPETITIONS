import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.knn_configs import KNNConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomKNN(CustomModelInterface, BaseEstimator):
    """
    Sklearn-based kNN model wrapper with a unified interface.

    This class wraps sklearn's KNeighborsClassifier and KNeighborsRegressor to
    provide a consistent API for classification and regression tasks. It supports:
      - Configurable hyperparameters via KNNConfig.
      - Model persistence with joblib.
      - Feature importance via permutation importance.
      - Logging of training and inference steps.
    """

    def __init__(self, config: KNNConfig):
        """
        Initialize the CustomKNN model with the given configuration.

        Args:
            config (KNNConfig): Configuration object containing model settings.
        """
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params or {}
        self.logger = Logger(self.__class__.__name__, save_to_file=config.save_log).get_logger()
        self.logger.info(
            f"Initialized {self.__class__.__name__} with model_name={self.model_name}."
            )

        self.model = None
        self.used_features: list[str] | None = None
        self.target_name: str = "target"
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None

    def build_model(self) -> None:
        """
        Build the sklearn kNN model based on the specified task type.

        Raises:
            ValueError: If task_type is unsupported.
        """
        self.logger.info("Building the kNN model based on task type.")
        if self.task_type == "classification":
            self.model = KNeighborsClassifier(**self.params)
        elif self.task_type == "regression":
            self.model = KNeighborsRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, fit_config: FitConfig) -> None:
        """
        Train the kNN model on the provided dataset.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.
            fit_config (FitConfig): Configuration for fitting, may include feature subset.

        Raises:
            ValueError: If training fails or invalid data provided.
        """
        self.target_name = y.name or self.target_name
        self.logger.info("Starting training of the kNN model.")

        # Feature selection
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_train = X

        # Store training data for permutation importance
        self._X_train = X_train.copy()
        self._y_train = y.copy()

        # Build and fit model
        self.build_model()
        self.model.fit(X_train, y)
        self.logger.info("Model training completed successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions or class labels from the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.Series: Predicted values or labels with same index as X.

        Raises:
            ValueError: If model is not trained or loaded.
        """
        if self.model is None:
            error_msg = "Model has not been trained or loaded."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        X_eval = X[self.used_features] if self.used_features else X
        preds = self.model.predict(X_eval)
        return pd.Series(preds, index=X.index, name=self.target_name)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate class probability estimates for classification tasks.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.DataFrame: Probability estimates for each class.

        Raises:
            ValueError: If model not trained.
            NotImplementedError: If called for regression or unsupported model.
        """
        if self.model is None:
            error_msg = "Model has not been trained or loaded."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.task_type != "classification":
            error_msg = "predict_proba() only available for classification tasks."
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        if not hasattr(self.model, "predict_proba"):
            error_msg = f"{type(self.model)} does not support predict_proba()."
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        X_eval = X[self.used_features] if self.used_features else X
        proba = self.model.predict_proba(X_eval)
        class_names = [f"{self.target_name}_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=class_names)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """
        Retrieve the top-N most important features using permutation importance.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance_score) sorted descending.

        Raises:
            ValueError: If model not trained or no training data.
        """
        self.logger.info(f"Computing permutation importance for top {top_n} features.")
        if self.model is None:
            raise ValueError("Model is not trained.")
        if self._X_train is None or self._y_train is None:
            raise ValueError("Training data must be available for permutation importance.")

        result = permutation_importance(
            self.model,
            self._X_train,
            self._y_train,
            n_repeats=5,
            random_state=42,
        )
        importances = result.importances_mean
        features = self.used_features
        top_idxs = importances.argsort()[::-1][:top_n]
        return [(features[i], importances[i]) for i in top_idxs]

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk using joblib serialization.

        Args:
            path (str): Path where the model will be saved.

        Raises:
            ValueError: If there is no trained model.
        """
        if self.model is None:
            error_msg = "No model to save."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}.")

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path (str): Path from which to load the model.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(path):
            error_msg = f"File not found: {path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}.")
