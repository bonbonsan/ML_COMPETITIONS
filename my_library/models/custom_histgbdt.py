import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.histgbdt_configs import HistGBDTConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomHistGBDT(CustomModelInterface, BaseEstimator):
    """HistGradientBoosting model wrapper.

    This class implements a scikit-learn HistGradientBoosting model that conforms
    to the CustomModelInterface. It supports both
    classification and regression tasks, using CPU computation only.

    Attributes:
        model_name (str): Name of the model.
        task_type (str): 'classification' or 'regression'.
        params (dict): Model hyperparameters.
        use_gpu (bool): Should always be False (not supported).
        model: The underlying sklearn model instance.
        logger: Logger instance for tracking model events.
    """

    def __init__(self, config: HistGBDTConfig):
        if config.use_gpu:
            raise ValueError("HistGradientBoosting does not support GPU acceleration.")
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params or {}
        self.use_gpu = config.use_gpu

        self.logger = Logger(self.__class__.__name__, save_to_file=config.save_log).get_logger()
        self.logger.info(
            f"Initialized {self.__class__.__name__} with model_name={self.model_name}."
            )

        self.model = None
        self.used_features = None

        self.target_name: str = "target"

    def build_model(self):
        """Construct the internal sklearn HistGradientBoosting model."""
        self.logger.info("Building the model.")
        if self.task_type == "classification":
            self.model = HistGradientBoostingClassifier(**self.params)
        elif self.task_type == "regression":
            self.model = HistGradientBoostingRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the HistGradientBoosting model using settings from FitConfig.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Training target vector.
            fitconfig (FitConfig): Configuration for fitting, including:
                - feats: optional list of feature columns to use
        """
        # set target name
        self.target_name = y.name or self.target_name
        self.logger.info("Starting HistGBDT training.")

        # select features if specified
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_train = X

        # build and fit model
        self.build_model()
        self.model.fit(X_train, y)
        self.logger.info("HistGBDT training completed.")

        # Save training data for permutation importance
        self.X_train_ = X_train.copy()
        self.y_train_ = y.copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Predicted values.
        """
        if self.model is None:
            error_message = "Model has not been trained or loaded."
            self.logger.error(error_message)
            raise ValueError(error_message)
        return pd.Series(self.model.predict(X), index=X.index, name=self.target_name)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities for classification tasks.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Class probabilities with one column per class.
                        Column names follow the format: target_0, target_1, ...
                        For binary classification, column 0 = class 0 (negative),
                        column 1 = class 1 (positive).
        """
        if self.model is None:
            error_message = "Model has not been trained or loaded."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if self.task_type != "classification":
            error_message = "predict_proba() is only available for classification tasks."
            self.logger.error(error_message)
            raise NotImplementedError(error_message)

        if not hasattr(self.model, "predict_proba"):
            error_message = f"{type(self.model)} does not support predict_proba()."
            self.logger.error(error_message)
            raise NotImplementedError(error_message)

        proba = self.model.predict_proba(X)

        # NOTE:
        # - For binary classification:
        #     - proba[:, 0] = probability of class 0 (negative)
        #     - proba[:, 1] = probability of class 1 (positive)
        # - For multiclass classification:
        #     - proba[:, i] = probability of class i

        class_names = [f"{self.target_name}_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=class_names)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """Return top-n features using permutation importance.

        Args:
            top_n (int): Number of top features to retrieve.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance) tuples sorted by importance.
        """
        self.logger.info(f"Getting top {top_n} features.")
        if self.model is None:
            raise ValueError("Model is not trained.")
        if self.used_features is None:
            raise ValueError("Used features are not recorded.")
        if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
            raise ValueError("Training data is required for permutation importance.")

        result = permutation_importance(
            self.model,
            self.X_train_,
            self.y_train_,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        importances = result.importances_mean
        pairs = list(zip(self.used_features, importances, strict=False))
        top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

        return top_pairs

    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            error_message = "No model to save."
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        dir_path = os.path.dirname(path)
        if dir_path:  # 空でなければ
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}.")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        if not os.path.exists(path):
            error_message = f"File not found: {path}"
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}.")
