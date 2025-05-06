import os

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.tabn_configs import TabNetConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.env_loader import get_device
from my_library.utils.logger import Logger


class CustomTabNetModel(CustomModelInterface, BaseEstimator):
    """Custom TabNet model wrapper for regression and classification tasks.

    This class integrates TabNetClassifier and TabNetRegressor using a shared
    interface compatible with the CustomDeepLearningModel abstraction.

    Attributes:
        model_name (str): Name of the model.
        task_type (str): 'classification' or 'regression'.
        params (dict): Model and training hyperparameters.
        model: Internal TabNet model instance.
    """

    def __init__(self, config: TabNetConfig):
        if config.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params
        self.use_gpu = config.use_gpu
        self.save_log = config.save_log
        
        self.device = get_device(self.use_gpu)
        self.logger = Logger(self.__class__.__name__, save_to_file=self.save_log).get_logger()
        self.logger.info(f"Initialized {self.__class__.__name__} on device={self.device}.")

        self.model = None
        self.used_features: list[str] = []

        self.target_name: str = "target"

    def build_model(self):
        """Build TabNet model based on task type and hyperparameters."""
        tabnet_class = TabNetRegressor if self.task_type == "regression" else TabNetClassifier
        self.model = tabnet_class(
            optimizer_params=self.params.get(
                "optimizer_params", {"lr": self.params.get("lr", 2e-2)}
                ),
            n_d=self.params.get("n_d", 64),
            n_a=self.params.get("n_a", 64),
            n_steps=self.params.get("n_steps", 5),
            gamma=self.params.get("gamma", 1.5),
            lambda_sparse=self.params.get("lambda_sparse", 1e-3),
            mask_type=self.params.get("mask_type", "entmax"),
            verbose=self.params.get("verbose", 0),
            seed=self.params.get("seed", 42),
            device_name=self.device
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the TabNet model using FitConfig settings.

        Args:
            X (pd.DataFrame|np.ndarray): Training features.
            y (pd.Series|np.ndarray): Training targets.
            fit_config (FitConfig): Configuration for fitting:
                - feats: optional list of feature names to select (DataFrame only)
                - eval_set: optional validation set [(X_val, y_val)]
                - epochs: optional number of max_epochs
                - batch_size: optional batch_size
        """
        # set target name
        self.target_name = getattr(y, "name", None) or self.target_name

        # feature selection for DataFrame
        if fit_config.feats and isinstance(X, pd.DataFrame):
            self.used_features = fit_config.feats
            X = X[fit_config.feats]
            if fit_config.eval_set:
                fit_config.eval_set = [(
                    df[fit_config.feats], lbl
                ) for df, lbl in fit_config.eval_set]
        elif isinstance(X, pd.DataFrame):
            self.used_features = X.columns.tolist()

        # convert to numpy arrays
        X_np = np.array(X)
        y_np = np.array(y)
        if self.task_type == "regression" and y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        # record training data for importance
        self.X_train_ = X_np
        self.y_train_ = y_np

        # determine training params
        p = self.params
        max_epochs = fit_config.epochs or p.get("max_epochs", 200)
        batch_size = fit_config.batch_size or p.get("batch_size", 1024)
        patience = p.get("patience", 20)
        virtual_batch_size = p.get(
            "virtual_batch_size", max(1, batch_size // 4)
        )
        eval_metric = ("rmse" if self.task_type == "regression" else "accuracy")

        # prepare eval set
        if fit_config.eval_set:
            eval_X = [np.array(df) for df, _ in fit_config.eval_set]
            eval_y = [(
                np.array(lbl).reshape(-1, 1)
                if self.task_type == "regression"
                else np.array(lbl)
            ) for _, lbl in fit_config.eval_set]
            eval_name = ["valid"]
        else:
            eval_X, eval_y, eval_name = [], [], []

        # build and train
        self.build_model()
        self.model.fit(
            X_train=X_np,
            y_train=y_np,
            eval_set=list(zip(eval_X, eval_y, strict=False)),
            eval_name=eval_name,
            eval_metric=[eval_metric],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False
        )
        self.logger.info("TabNet training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict using the trained TabNet model.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.Series: Predictions with index matching X.
        """
        if self.model is None:
            raise ValueError("Model is not built or trained.")
        X_np = np.array(X)
        preds = self.model.predict(X_np).reshape(-1)

        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(preds))
        return pd.Series(preds, index=index, name=self.target_name)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities (classification only).

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.DataFrame: Predicted probabilities per class.
        """
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        probs = self.model.predict_proba(np.array(X))
        class_names = [f"{self.target_name}_{i}" for i in range(probs.shape[1])]
        return pd.DataFrame(probs, index=X.index, columns=class_names)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """
        Compute top-n important features using permutation importance.

        Args:
            top_n (int): Number of top features to retrieve.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance) tuples.
        """
        # Ensure training data is available
        if not hasattr(self, 'X_train_') or not hasattr(self, 'y_train_'):
            raise ValueError(
                "Training data not found. Fit must be called before getting feature importances.")

        # Choose scoring method based on task
        if self.task_type == "classification":
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'

        # Compute permutation importance
        result = permutation_importance(
            self.model,
            self.X_train_,
            self.y_train_,
            scoring=scoring,
            n_repeats=self.params.get('n_repeats', 5),
            random_state=self.params.get('random_state', 42),
            n_jobs=-1
        )
        importances = result.importances_mean
        pairs = list(zip(self.used_features, importances, strict=False))
        # Sort by importance
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

    def save_model(self, path: str):
        """Save model weights and configuration.

        Args:
            path (str): Path to save model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        self.logger.info(f"Model saved to {path}.")

    def load_model(self, path: str):
        """Load model from file.

        Args:
            path (str): Path prefix where model was saved.
        """
        tabnet_class = TabNetRegressor if self.task_type == "regression" else TabNetClassifier
        self.model = tabnet_class()
        self.model.load_model(path + ".zip")
        self.logger.info(f"Model loaded from {path}.")
