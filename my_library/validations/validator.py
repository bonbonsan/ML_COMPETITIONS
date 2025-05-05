"""
validator.py

This script defines a Validator class that wraps a gradient boosting
model with consistent training, prediction, and evaluation methods.

It supports early stopping, logging, and compatibility with scikit-learn
datasets. Designed to work with custom gradient boosting models such as
CustomLightGBM, CustomXGBoost, and CustomCatBoost.

"""

from typing import Optional

import pandas as pd

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger

logger = Logger(__name__, save_to_file=False).get_logger()

class Validator:
    """Validator for training, predicting, and evaluating models."""

    def __init__(self, model: CustomModelInterface):
        """Initialize the Validator class.

        Args:
            model (CustomModelInterface): Custom model object.
        """
        self.model = model
        logger.info(f"Initialized Validator for model: {model.__class__.__name__}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fit_config: FitConfig
    ) -> None:
        """
        Train the model using an externally provided FitConfig.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            fit_config (FitConfig): Configuration for fitting, including:
                - feats: optional feature list
                - eval_set: optional validation set
                - early_stopping_rounds: optional early stopping
                - epochs: optional epochs
                - batch_size: optional batch size
        """
        # call custom fit with unified interface
        logger.info(
            f"Training start | model={self.model.__class__.__name__} "
            f"samples={len(X_train)} features={X_train.shape[1]}"
        )
        self.model.fit(X_train, y_train, fit_config=fit_config)
        logger.info("Training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using the model.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Model predictions.
        """
        return self.model.predict(X)
    
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
        return self.model.predict_proba(X)

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_fn: Optional[callable] = None
    ) -> float:
        """Evaluate model performance.

        Args:
            y_true (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.
            metric_fn (Optional[callable]): Custom evaluation function.

        Returns:
            float: Evaluation score.
        """
        if metric_fn is not None:
            return metric_fn(y_true, y_pred)

        if self.model.task_type == "classification":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_true, y_pred)
        else:
            from sklearn.metrics import root_mean_squared_error
            return root_mean_squared_error(y_true, y_pred)
        
    def get_params(self) -> dict:
        """Return model parameters.

        Returns:
            dict: Model configuration and hyperparameters.
        """
        return self.model.params
