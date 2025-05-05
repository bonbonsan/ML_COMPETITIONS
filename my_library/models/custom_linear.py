import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.linear_configs import LinearConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomLinear(CustomModelInterface, BaseEstimator):
    """
    Sklearn-based linear model wrapper with a standard interface.

    This class provides a unified API for both regression and classification
    tasks by wrapping sklearn's LinearRegression and LogisticRegression models.
    It integrates with FitConfig for configurable training, supports probability
    estimation for classification, feature importance extraction via permutation
    importance (recommended) or coefficient magnitude (deprecated),
    and model persistence via joblib.

    Attributes:
        model_name (str): Identifier for the model instance.
        task_type (str): Task type, either 'classification' or 'regression'.
        params (dict): Hyperparameters passed to the sklearn model constructor.
        model (LinearRegression or LogisticRegression): The underlying sklearn model.
        used_features (list[str] or None): List of feature names used in training.
        target_name (str): Name of the target column for prediction outputs.
        logger (logging.Logger): Logger instance for recording training and inference events.
    """

    def __init__(self, config: LinearConfig):
        """
        Initialize the CustomLinear model with the given configuration.

        Args:
            config (LinearConfig): Configuration object containing:
                - model_name (str): Name for the model instance.
                - task_type (Literal['classification', 'regression']):
                  Specifies whether to use LogisticRegression or LinearRegression.
                - params (dict): Keyword arguments for the sklearn model.
                - save_log (bool): If True, training logs will be saved to file.

        Raises:
            ValueError: If config.task_type is not 'classification' or 'regression'.
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
        self.target_name: str = "target"
        # store training data for permutation importance
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None

    def build_model(self):
        """
        Build the sklearn linear model based on the specified task type.

        For 'classification', initializes a LogisticRegression model.
        For 'regression', initializes a LinearRegression model.

        Raises:
            ValueError: If self.task_type is not 'classification' or 'regression'.
        """
        self.logger.info("Building the linear model based on task type.")
        if self.task_type == "classification":
            self.model = LogisticRegression(**self.params)
        elif self.task_type == "regression":
            self.model = LinearRegression(**self.params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, fit_config: FitConfig) -> None:
        """
        Train the linear model on the provided dataset.

        Selects features according to FitConfig and fits the underlying sklearn model.
        Stores training data for later permutation importance calculation.

        Args:
            X (pd.DataFrame): Training features, where rows correspond to samples
                and columns to feature variables.
            y (pd.Series): Target values for training with the same index as X.
            fit_config (FitConfig): Configuration for fitting process, which may
                include:
                  - feats (list[str], optional): Subset of feature names to use.

        Raises:
            ValueError: If the underlying model cannot be fit (e.g., invalid data).
        """
        self.target_name = y.name or self.target_name
        self.logger.info("Starting training of the linear model.")

        # Feature selection logic
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_train = X

        # Store training data for importance
        self._X_train = X_train.copy()
        self._y_train = y.copy()

        # Build and fit the model
        self.build_model()
        self.model.fit(X_train, y)
        self.logger.info("Model training completed successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions or class labels from the trained model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.Series: Predicted values (regression) or labels
                (classification) with the same index as X.

        Raises:
            ValueError: If the model is not yet trained or loaded.
        """
        if self.model is None:
            error_message = "Model has not been trained or loaded."
            self.logger.error(error_message)
            raise ValueError(error_message)

        X_eval = X[self.used_features] if self.used_features else X
        preds = self.model.predict(X_eval)
        return pd.Series(preds, index=X.index, name=self.target_name)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate class probability estimates for classification tasks.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.DataFrame: Probability estimates for each class with columns
                named '{target_name}_0', '{target_name}_1', etc., matching X.index.

        Raises:
            ValueError: If the model is not trained or loaded.
            NotImplementedError: If called for regression tasks or if the
                underlying model does not support probability prediction.
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

        X_eval = X[self.used_features] if self.used_features else X
        proba = self.model.predict_proba(X_eval)
        class_names = [f"{self.target_name}_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=class_names)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """
        Retrieve the top-N most important features using permutation importance.

        This method computes feature importance by measuring the decrease in
        model performance when each feature's values are randomly shuffled.
        A larger drop in performance indicates a higher importance.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            list[tuple[str, float]]: A list of tuples (feature_name, importance_score)
                sorted by importance_score descending.

        Raises:
            ValueError: If the model is not trained or training data is unavailable.
        """
        self.logger.info(f"Computing permutation importance for top {top_n} features.")
        if self.model is None:
            raise ValueError("Model is not trained.")
        if self._X_train is None or self._y_train is None:
            raise ValueError("Training data must be available for permutation importance.")

        # Use stored training data for evaluation
        X_eval = self._X_train
        y_eval = self._y_train

        # Compute permutation importance
        result = permutation_importance(
            self.model,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=42,
        )
        importances = result.importances_mean
        features = self.used_features
        top_idxs = importances.argsort()[::-1][:top_n]
        top_features = [(features[i], importances[i]) for i in top_idxs]
        return top_features

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk using joblib serialization.

        Args:
            path (str): File system path where the model will be saved.

        Raises:
            ValueError: If there is no trained model to save.
            OSError: If the directory cannot be created.
        """
        if self.model is None:
            error_message = "No model to save."
            self.logger.error(error_message)
            raise ValueError(error_message)

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}.")

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path (str): File system path from which to load the model.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If the model file cannot be loaded.
        """
        if not os.path.exists(path):
            error_message = f"File not found: {path}"
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}.")
