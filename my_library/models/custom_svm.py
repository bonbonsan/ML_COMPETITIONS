import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC, SVR

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.svm_configs import SVMConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomSVM(CustomModelInterface, BaseEstimator):
    """
    Sklearn-based SVM model wrapper with unified interface.

    This class provides a unified API for both classification and regression tasks by
    wrapping sklearn's SVC and SVR. It integrates with FitConfig for configurable
    training, supports probability estimation for classification, feature importance
    extraction via permutation importance, and model persistence via joblib.
    """

    def __init__(self, config: SVMConfig):
        """
        Initialize the CustomSVM model with the given configuration.

        Args:
            config (SVMConfig): Configuration object containing:
                - model_name (str): Name for the model instance.
                - task_type (Literal['classification','regression']): Specifies SVC or SVR.
                - params (Dict): Keyword arguments for SVC or SVR.
                - save_log (bool): If True, training logs will be saved to file.
        """
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params or {}
        self.logger = Logger(self.__class__.__name__, save_to_file=config.save_log).get_logger()
        self.logger.info(
            f"Initialized {self.__class__.__name__} with model_name={self.model_name}."
            )

        self.model = None
        self.used_features = None
        self.target_name = "target"
        self._X_train = None
        self._y_train = None

    def build_model(self):
        """
        Build the sklearn SVM model based on the specified task type.

        For 'classification', initializes an SVC model with probability=True.
        For 'regression', initializes an SVR model.

        Raises:
            ValueError: If self.task_type is not 'classification' or 'regression'.
        """
        self.logger.info("Building the SVM model based on task type.")
        params = self.params.copy()
        if self.task_type == "classification":
            # Enable probability estimates for classification
            params.setdefault("probability", True)
            self.model = SVC(**params)
        elif self.task_type == "regression":
            self.model = SVR(**params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, fit_config: FitConfig) -> None:
        """
        Train the SVM model on the provided dataset.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.
            fit_config (FitConfig): Configuration for fitting process.
        """
        self.target_name = y.name or self.target_name
        self.logger.info("Starting training of the SVM model.")

        # Feature selection
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_train = X

        # Store data for permutation importance
        self._X_train = X_train.copy()
        self._y_train = y.copy()

        self.build_model()
        self.model.fit(X_train, y)
        self.logger.info("Model training completed successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions from the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.Series: Predicted values or labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        X_eval = X[self.used_features] if self.used_features else X
        preds = self.model.predict(X_eval)
        return pd.Series(preds, index=X.index, name=self.target_name)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate class probability estimates for classification tasks.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.DataFrame: Probability estimates for each class.

        Raises:
            NotImplementedError: If called for regression tasks or
            if model does not support predict_proba.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba() is only available for classification tasks.")
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(f"{type(self.model)} does not support predict_proba().")

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
            ValueError: If model is not trained or training data not available.
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
        Save the trained model to disk using joblib.

        Args:
            path (str): File path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}.")

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path (str): File path to load the model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}.")
