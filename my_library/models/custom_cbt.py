import os

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.base import BaseEstimator

from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomCatBoost(CustomModelInterface, BaseEstimator):
    """CatBoost model wrapper with a standard GBDT interface.

    This class implements a CatBoost model that conforms to the interface
    defined by CustomModelInterface, supporting both
    classification and regression tasks with optional GPU acceleration.
    """

    def __init__(self, config: CatBoostConfig):
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
        """Build the CatBoost model based on task type and parameters."""
        self.logger.info("Building the model.")
        params = self.params.copy()
        if self.use_gpu:
            params.update({"task_type": "GPU"})

        if self.task_type == "classification":
            self.model = CatBoostClassifier(**params)
        elif self.task_type == "regression":
            self.model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the CatBoost model using settings from FitConfig.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Training target vector.
            config (FitConfig): Configuration for fitting, including:
                - feats: optional list of features to use
                - eval_set: optional validation set for early stopping
                - early_stopping_rounds: rounds to wait before stopping
        """
        # set target name
        self.target_name = y.name or self.target_name
        self.logger.info("Starting CatBoost training.")

        # select features
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

        # build underlying model
        self.build_model()

        # prepare eval pool
        eval_pool = None
        if eval_set:
            X_val, y_val = eval_set[0]
            eval_pool = Pool(X_val, y_val)

        # fit model
        self.model.fit(
            X_train,
            y,
            eval_set=eval_pool,
            early_stopping_rounds=fit_config.early_stopping_rounds,
            verbose=False,
        )
        self.logger.info("CatBoost training completed.")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Predicted values.

        Example:
            >>> preds = model.predict(X_test)
            >>> print(preds.shape)  # (n_samples,)

        Notes:
            - For classification, CatBoost may return:
                [[0], [1], [0], ..., [2]]  # shape: (n, 1)
            which needs to be flattened to:
                [0, 1, 0, ..., 2]  # shape: (n,)
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        preds = self.model.predict(X)

        # If predictions are 2D (e.g., shape: (n, 1)), flatten them to 1D
        if hasattr(preds, "ndim") and preds.ndim > 1:
            preds = preds.ravel()

        return pd.Series(preds, index=X.index, name=self.target_name)
    
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
        """Get top-n features using CatBoost PredictionValuesChange importance.

        Args:
            top_n (int): Number of top features to retrieve.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance) pairs.
        """
        self.logger.info(f"Getting top {top_n} features.")
        if self.model is None:
            raise ValueError("Model is not trained.")

        importances = self.model.get_feature_importance(type="PredictionValuesChange")

        top_pairs = sorted(
            zip(self.used_features, importances, strict=False),
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
