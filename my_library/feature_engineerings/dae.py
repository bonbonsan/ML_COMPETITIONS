"""
dae.py

This module implements a denoising autoencoder (DAE) using PyTorch,
designed for robust feature extraction from tabular numerical data.

Main components:
- `DenoisingAutoencoder`: A simple feed-forward autoencoder model.
- `DAETrainer`: Utility class for training the DAE with Gaussian noise
                and extracting latent representations.

This module is especially useful for preprocessing steps in machine learning pipelines,
such as dimensionality reduction, denoising, and unsupervised representation learning.

Example usage:
    X = np.random.rand(1000, 20)
    trainer = DAETrainer(input_dim=20)
    trainer.fit(X)
    X_encoded = trainer.transform(X)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class DenoisingAutoencoder(nn.Module):
    """
    A simple denoising autoencoder with one hidden layer.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the latent representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Suitable for inputs clipped in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Reconstructed tensor of shape (batch_size, input_dim).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """
        Returns the encoded (latent) representation of the input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Encoded tensor.
        """
        return self.encoder(x)


class DAETrainer:
    """
    Trainer class for fitting a denoising autoencoder with noise injection and early stopping.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the latent representation.
        noise_std (float): Standard deviation of Gaussian noise added during training.
        device (str): Device to use ('cpu' or 'cuda').
        patience (int): Early stopping patience based on validation loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        noise_std: float = 0.1,
        device: str = "cpu",
        patience: int = 5,
    ):
        self.model = DenoisingAutoencoder(input_dim, hidden_dim).to(device)
        self.scaler = StandardScaler()
        self.noise_std = noise_std
        self.device = device
        self.patience = patience

    def _validate_input(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a NumPy ndarray.")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Input array must be numeric. Detected dtype: {X.dtype}")

        if np.isnan(X).any():
            raise ValueError("Input contains NaN values. Please impute or remove them.")


    def fit(self, X: np.ndarray, val_split: float = 0.1, epochs: int = 100, batch_size: int = 64):
        """
        Fit the autoencoder on noisy inputs and minimize reconstruction loss.

        Args:
            X (np.ndarray): Input training data (num_samples, num_features).
            val_split (float): Proportion of data to use for validation.
            epochs (int): Maximum number of training epochs.
            batch_size (int): Mini-batch size for training.
        """
        self._validate_input(X)

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, 0.0, 1.0)

        n_val = int(len(X_scaled) * val_split)
        X_train, X_val = X_scaled[n_val:], X_scaled[:n_val]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self.device)
                noise = torch.randn_like(x_batch) * self.noise_std
                x_noisy = x_batch + noise

                output = self.model(x_noisy)
                loss = criterion(output, x_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_recon = self.model(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()

            print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping.")
                    break

        self.model.load_state_dict(self.best_model_state)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the trained encoder to transform input into latent features.

        Args:
            X (np.ndarray): Input data to transform.

        Returns:
            np.ndarray: Encoded latent representation of shape (num_samples, hidden_dim).
        """
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, 0.0, 1.0)
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            features = self.model.encode(x_tensor).cpu().numpy()
        return features


if __name__ == "__main__":
    """
    Example usage for training and feature extraction with the DAE.
    """
    X = np.random.rand(1000, 20)

    trainer = DAETrainer(input_dim=20, hidden_dim=16, noise_std=0.2, device="cpu", patience=5)
    trainer.fit(X)

    X_encoded = trainer.transform(X)
    print(X_encoded.shape)
    print(X_encoded[:10])
