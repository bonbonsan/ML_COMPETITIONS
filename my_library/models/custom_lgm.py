import os

import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping
from sklearn.base import BaseEstimator

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.lightgbm_configs import LightGBMConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomLightGBM(CustomModelInterface, BaseEstimator):
    """LightGBM model wrapper with a standard GBDT interface.

    This class extends CustomModelInterface and integrates
    LightGBM's classifier and regressor models, supporting both CPU and GPU
    computations based on user preference.

    Attributes:
        model_name (str): Name of the model.
        task_type (str): Type of task - 'classification' or 'regression'.
        params (dict): Parameters for the LightGBM model.
        use_gpu (bool): Flag to indicate GPU usage.
        model: The underlying LightGBM model instance.
        logger: Logger instance for logging events.
    """

    def __init__(self, config: LightGBMConfig):
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
        """Build the LightGBM model based on task type and parameters."""
        self.logger.info("Building the model.")
        params = self.params.copy()
        if self.use_gpu:
            params.update({"device": "gpu"})

        if self.task_type == "classification":
            self.model = lgb.LGBMClassifier(**params)
        elif self.task_type == "regression":
            self.model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the LightGBM model using FitConfig settings.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            fit_config (FitConfig): Configuration for fitting:
                - feats: optional list of feature columns to use
                - eval_set: optional validation set
                - early_stopping_rounds: optional rounds for early stopping
        """
        # assign target name
        self.target_name = y.name or self.target_name
        self.logger.info("Starting LightGBM training.")

        # feature selection
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[fit_config.feats]
            eval_set = [(
                df[fit_config.feats],
                lbl
            ) for df, lbl in (fit_config.eval_set or [])]
        else:
            self.used_features = X.columns.tolist()
            X_train = X
            eval_set = fit_config.eval_set

        # build model
        self.build_model()

        # setup callbacks
        callbacks = []
        if fit_config.early_stopping_rounds:
            callbacks.append(early_stopping(fit_config.early_stopping_rounds))

        # fit
        self.model.fit(
            X_train,
            y,
            eval_set=eval_set,
            callbacks=callbacks or None,
        )
        self.logger.info("LightGBM training completed.")

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
        """Return top-n important features using LightGBM gain importance.

        Args:
            top_n (int): Number of top features to retrieve.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance) pairs.
        """
        self.logger.info(f"Getting top {top_n} features.")
        if self.model is None:
            raise ValueError("Model is not trained.")

        booster = self.model.booster_
        importances = booster.feature_importance(importance_type="gain")
        feature_names = booster.feature_name()

        top_pairs = sorted(
            zip(feature_names, importances, strict=False),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

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
