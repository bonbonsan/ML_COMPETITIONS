from abc import ABC, abstractmethod

import pandas as pd

from my_library.configs.model_configs.fit_configs import FitConfig


class CustomModelInterface(ABC):
    """Abstract base class for custom machine learning models.

    All models should inherit from this base class to ensure compatibility
    with the training, prediction, and persistence ecosystem.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, fit_config: FitConfig) -> None:
        """Train the model on the provided data.

        Args:
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.
            fit_config (FitConfig): Configuration for the fitting process.

        Example:
            model.fit(X_train, y_train)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions from the model.

        Args:
            X (pd.DataFrame): Feature matrix for prediction.

        Returns:
            pd.Series: Predictions with the same index as X.

        Example:
            y_pred = model.predict(X_test)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities for classification tasks.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Class probabilities with one column per class.
                        Column names follow the format: target_0, target_1, ...
                        For binary classification, column 0 = class 0 (negative),
                        column 1 = class 1 (positive).

        Raises:
            NotImplementedError: If not supported by the specific model.
        """
        pass

    @abstractmethod
    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """Return top-n important features.

        Args:
            top_n (int): Number of top features to retrieve.

        Returns:
            list[tuple[str, float]]: List of (feature_name, importance) pairs.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model to a file.

        Args:
            path (str): File path where the model should be saved.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model from a file.

        Args:
            path (str): File path from which the model should be loaded.
        """
        pass

    def get_params(self) -> dict:
        """Return model parameters.

        Returns:
            dict: Model configuration and hyperparameters.
        """
        return self.params

    def set_params(self, **params) -> None:
        """Update model parameters."""
        if params == self.params:
            self.logger.info("No parameter changes detected.")
            return
        
        self.params.update(params)
        self.logger.info(f"Model parameters updated: {params}")
        self.logger.warning(
            "Model parameters have been updated, but the model itself has NOT been rebuilt. "
            "Call `build_model()` or `fit()` to apply the changes."
        )

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(params={self.params})"

    def __repr__(self) -> str:
        return self.__str__()
