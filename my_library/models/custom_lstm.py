import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.lstm_configs import LSTMConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.logger import Logger


class CustomLSTMModel(CustomModelInterface):
    """
    PyTorch-based LSTM model for regression and classification tasks.

    This model accepts input in 2D format (samples x features), and internally reshapes it
    into 3D format (samples x time_steps x input_size) for LSTM compatibility.

    For time-series forecasting tasks, it is recommended to transform the raw series into
    supervised learning format using `create_sequences()` from
    `my_library.utils.preprocessing_utils`.

    Example for time-series (e.g. Airline):
        from my_library.utils.preprocessing_utils import create_sequences

        X_seq, y_seq = create_sequences(df, col="Passengers", time_steps=5)
        model = CustomLSTMModel(config)
        model.fit(X_seq, y_seq)

    For tabular data (e.g. Iris or Diabetes), this is handled automatically by reshaping.

    Example for tabular:
        model = CustomLSTMModel(config)
        model.fit(X, y)
    """

    def __init__(self, config: LSTMConfig):
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params
        self.use_gpu = config.use_gpu
        self.save_log = config.save_log

        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        self.logger = Logger(self.__class__.__name__, save_to_file=self.save_log).get_logger()
        self.logger.info(f"Initialized {self.__class__.__name__} on device={self.device}.")

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.target_name: str = "target"
        self.used_features: list[str] = []

    def build_model(self) -> None:
        """Build LSTM model and related components."""
        p = self.params
        self.model = LSTMNet(
            input_size=p["input_size"],
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            output_size=p["output_size"],
            activation=p.get("activation"),
            dropout=p.get("dropout", 0.0)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p["lr"])
        self.criterion = (
            nn.MSELoss() if self.task_type == "regression" else nn.CrossEntropyLoss()
        )

    def _prepare_input_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        """Convert 2D DataFrame to 3D tensor for LSTM input.

        Args:
            X (pd.DataFrame): Input of shape (N, F)

        Returns:
            torch.Tensor: 3D tensor of shape (N, T, F/T)
        """
        p = self.params
        time_steps = p.get("time_steps", 1)
        n_samples, n_features = X.shape

        if n_features % time_steps != 0:
            raise ValueError(
                f"Number of features ({n_features}) must be divisible by time_steps ({time_steps})"
            )

        input_size = n_features // time_steps
        X_array = X.values.astype(np.float32)
        X_3d = X_array.reshape(n_samples, time_steps, input_size)  # shape: (N, T, F/T)
        return torch.tensor(X_3d, dtype=torch.float32).to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the LSTM model using FitConfig settings.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training targets.
            fit_config (FitConfig): Configuration for fitting:
                - feats: optional feature selection list
                - epochs: optional training epochs
                - batch_size: optional batch size
        """
        # set target name
        self.target_name = y.name or self.target_name
        # feature selection
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_train_df = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_train_df = X
        # record training data for importance
        self.X_train_ = X_train_df.copy()
        self.y_train_ = y.copy()

        # determine hyperparameters
        p = self.params
        epochs = fit_config.epochs or p.get("epochs", 100)
        batch_size = fit_config.batch_size or p.get("batch_size", 32)

        # build and train
        self.build_model()
        X_tensor = self._prepare_input_tensor(X_train_df)
        y_arr = y.values
        if self.task_type == "regression":
            y_arr = y_arr.reshape(-1)
        y_tensor = torch.tensor(
            y_arr.astype(int) if self.task_type == "classification" else y_arr,
            dtype=torch.long if self.task_type == "classification" else torch.float32,
        ).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                out = self.model(xb)
                if self.task_type == "regression":
                    out = out.view(-1)
                    yb = yb.view(-1)
                loss = self.criterion(out, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.Series: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model is not built or trained.")

        self.model.eval()
        X_tensor = self._prepare_input_tensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == "classification":
                preds = outputs.argmax(dim=1).cpu().numpy()
            else:
                preds = outputs.view(-1).cpu().numpy()

        return pd.Series(preds.ravel(), index=X.index, name=self.target_name)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predicted class probabilities (classification only).

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Predicted probabilities per class.
        """
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba() is only for classification tasks.")

        self.model.eval()
        X_tensor = self._prepare_input_tensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        class_names = [f"{self.target_name}_{i}" for i in range(probs.shape[1])]
        return pd.DataFrame(probs, index=X.index, columns=class_names)

    def get_top_features(self, top_n: int) -> list[tuple[str, float]]:
        """
        Return top-n features using permutation-based importance.
        """
        # Ensure training data exists
        if not hasattr(self, 'X_train_') or not hasattr(self, 'y_train_'):
            raise ValueError("Fit must be called before get_top_features.")

        # baseline performance
        X_df = self.X_train_
        y_true = self.y_train_
        # choose metric
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score
            baseline_preds = self.predict(X_df)
            baseline_score = accuracy_score(y_true, baseline_preds)
            def score_fn(y, y_pred):
                return accuracy_score(y, y_pred)
            # importance = drop in accuracy
            def importance_fn(orig, perm):
                return orig - perm
        else:
            from sklearn.metrics import mean_squared_error
            baseline_preds = self.predict(X_df)
            baseline_score = mean_squared_error(y_true, baseline_preds)
            def score_fn(y, y_pred):
                return mean_squared_error(y, y_pred)
            # importance = increase in error
            def importance_fn(orig, perm):
                return perm - orig

        importances = []
        for feat in self.used_features:
            X_perm = X_df.copy()
            X_perm[feat] = np.random.permutation(X_perm[feat].values)
            perm_preds = self.predict(X_perm)
            perm_score = score_fn(y_true, perm_preds)
            imp_val = importance_fn(baseline_score, perm_score)
            importances.append((feat, imp_val))

        # sort features by importance descending
        sorted_pairs = sorted(importances, key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]


    def save_model(self, path: str) -> None:
        """Save model weights to a file."""
        if self.model is None:
            raise ValueError("No model to save.")

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model weights saved to {path}.")

    def load_model(self, path: str) -> None:
        """Load model weights from a file."""
        self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model weights loaded from {path}.")


class LSTMNet(nn.Module):
    """Vanilla LSTM network for sequence modeling.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of LSTM layers.
        output_size (int): Number of output units.
        activation (str): Optional activation ('relu', 'tanh', 'sigmoid').
        dropout (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size,
                 num_layers, output_size, activation=None, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = self._get_activation(activation)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        if self.activation:
            out = self.activation(out)
        return out

    def _get_activation(self, name):
        if name is None:
            return None
        nl = name.lower()
        return {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }.get(nl, None)   
