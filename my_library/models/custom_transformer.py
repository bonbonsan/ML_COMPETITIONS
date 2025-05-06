import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.transformer_configs import TransformerConfig
from my_library.models.interface import CustomModelInterface
from my_library.utils.env_loader import get_device
from my_library.utils.logger import Logger


class CustomTransformerModel(CustomModelInterface):
    """
    PyTorch-based Transformer model for regression and classification tasks.

    This model accepts input in 2D format (samples x features), and internally reshapes it
    into 3D format (samples x time_steps x input_size) for Transformer compatibility.

    For time-series forecasting tasks, it is recommended to transform the raw series into
    supervised learning format using `create_sequences()` from
    `my_library.utils.preprocessing_utils`.

    Example for time-series (e.g. Airline):
        from my_library.utils.preprocessing_utils import create_sequences

        X_seq, y_seq = create_sequences(df, col="Passengers", time_steps=5)
        model = CustomTransformerModel(config)
        model.fit(X_seq, y_seq)

    For tabular data (e.g. Iris or Diabetes), this is handled automatically by reshaping.

    Example for tabular:
        model = CustomTransformerModel(config)
        model.fit(X, y)
    """

    def __init__(self, config: TransformerConfig):
        self.model_name = config.model_name
        self.task_type = config.task_type
        self.params = config.params
        self.use_gpu = config.use_gpu
        self.save_log = config.save_log

        self.device = get_device(self.use_gpu)
        self.logger = Logger(self.__class__.__name__, save_to_file=self.save_log).get_logger()
        self.logger.info(f"Initialized {self.__class__.__name__} on device={self.device}.")

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.target_name: str = "target"
        self.used_features: list[str] = []

    def build_model(self) -> None:
        """Build Transformer model and related components."""
        p = self.params
        self.model = TransformerNet(
            input_size=p["input_size"],
            output_size=p["output_size"],
            d_model=p["d_model"],
            nhead=p["nhead"],
            num_layers=p["num_layers"],
            dim_feedforward=p["dim_feedforward"],
            dropout=p.get("dropout", 0.0),
            activation=p.get("activation", "relu"),
            task_type=self.task_type
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p["lr"])
        self.criterion = (
            nn.MSELoss() if self.task_type == "regression" else nn.CrossEntropyLoss()
        )

    def _prepare_input_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        """Convert 2D DataFrame to 3D tensor for Transformer input.

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
        X_3d = X_array.reshape(n_samples, time_steps, input_size)
        return torch.tensor(X_3d, dtype=torch.float32).to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_config: FitConfig,
    ) -> None:
        """
        Train the Transformer model using FitConfig settings.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training targets.
            fit_config (FitConfig): contains:
                - feats: optional list[str]
                - epochs: optional int
                - batch_size: optional int
        """
        # set target name
        self.target_name = y.name or self.target_name
        # feature selection
        if fit_config.feats:
            self.used_features = fit_config.feats
            X_df = X[fit_config.feats]
        else:
            self.used_features = X.columns.tolist()
            X_df = X
        # record training data for importance
        self.X_train_ = X_df.copy()
        self.y_train_ = y.copy()

        # hyperparameters
        p = self.params
        epochs = fit_config.epochs or p.get('epochs', 100)
        batch_size = fit_config.batch_size or p.get('batch_size', 32)

        # build and train
        self.build_model()
        X_tensor = self._prepare_input_tensor(X_df)
        y_arr = y.values
        if self.task_type == 'regression':
            y_tensor = torch.tensor(y_arr.reshape(-1), dtype=torch.float32, device=self.device)
        else:
            y_tensor = torch.tensor(y_arr.astype(int), dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                out = self.model(xb)
                if self.task_type == 'regression':
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
        Return top-n features using permutation importance.
        """
        if not hasattr(self, 'X_train_') or not hasattr(self, 'y_train_'):
            raise ValueError("Fit must be called before get_top_features.")
        # baseline score
        X_df = self.X_train_
        y_true = self.y_train_
        if self.task_type == 'classification':
            from sklearn.metrics import accuracy_score
            base_preds = self.predict(X_df)
            base_score = accuracy_score(y_true, base_preds)
            score_fn = accuracy_score
            def imp_fn(o, p):
                return o - p
        else:
            from sklearn.metrics import mean_squared_error
            base_preds = self.predict(X_df)
            base_score = mean_squared_error(y_true, base_preds)
            score_fn = mean_squared_error
            def imp_fn(o, p):
                return p - o

        importances = []
        for feat in self.used_features:
            Xp = X_df.copy()
            Xp[feat] = np.random.permutation(Xp[feat].values)
            preds = self.predict(Xp)
            score = score_fn(y_true, preds)
            importances.append((feat, imp_fn(base_score, score)))

        return sorted(importances, key=lambda x: x[1], reverse=True)[:top_n]

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



class PositionalEncoding(nn.Module):
    """Injects positional information into input sequences."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerNet(nn.Module):
    """PyTorch Transformer architecture for sequence modeling.

    Args:
        input_size (int): Number of input features.
        output_size (int): Output dimension.
        d_model (int): Dimension of embedding.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Size of hidden layer.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        task_type (str): 'classification' or 'regression'.
    """

    def __init__(self, input_size, output_size, d_model, nhead, num_layers,
                 dim_feedforward, dropout, activation="relu", task_type="regression"):
        super().__init__()
        self.task_type = task_type
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_size)
        self.output_activation = self._get_output_activation(task_type)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        out = self.output_layer(x[:, -1, :])
        if self.output_activation:
            out = self.output_activation(out)
        return out

    def _get_output_activation(self, task_type):
        if task_type == "regression":
            return None
        elif task_type == "classification":
            return nn.Sigmoid()  # for binary classification
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
