import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class BaseEncoder(ABC):
    """
    Abstract base class for label encoders.
    """

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Learns encoding from the given series (if applicable)."""
        pass

    @abstractmethod
    def transform(self, series: Union[pd.Series, List[Any]]) -> Union[pd.Series, List[int]]:
        """Encodes values in the given series using internal mapping."""
        pass

    @abstractmethod
    def inverse_transform(self, series: Union[pd.Series, List[int]]) -> Union[pd.Series, List[Any]]:
        """Decodes integer labels back to original values."""
        pass

    @abstractmethod
    def get_mapping(self) -> Dict[Any, int]:
        """Returns the mapping from original values to encoded integers."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Saves encoder state to file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseEncoder":
        """Loads encoder state from file."""
        pass


class DictLabelEncoder(BaseEncoder):
    """
    Label encoder using a user-specified dictionary.
    """

    def __init__(self, mapping: Dict[Any, int], unknown_label: int = -1):
        self.mapping = mapping
        self.unknown_label = unknown_label
        self.inverse = {v: k for k, v in mapping.items()}

    def fit(self, series: pd.Series) -> None:
        pass  # Not needed

    def transform(self, series: pd.Series) -> pd.Series:
        if isinstance(series, list):
            return [self.mapping.get(val, self.unknown_label) for val in series]
        return series.map(self.mapping).fillna(self.unknown_label).astype(int)

    def inverse_transform(self, series: Union[pd.Series, List[int]]) -> Union[pd.Series, List[Any]]:
        if isinstance(series, list):
            return [self.inverse.get(val, None) for val in series]
        return series.map(self.inverse)

    def get_mapping(self) -> Dict[Any, int]:
        return self.mapping

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                "mapping": self.mapping,
                "inverse": self.inverse,
                "unknown_label": self.unknown_label
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "DictLabelEncoder":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        encoder = cls(mapping=data["mapping"], unknown_label=data.get("unknown_label", -1))
        encoder.inverse = data["inverse"]
        return encoder


class AutoLabelEncoder(BaseEncoder):
    """
    Label encoder that learns mapping from data automatically.
    """

    def __init__(self, unknown_label: int = -1):
        self.mapping: Dict[Any, int] = {}
        self.inverse: Dict[int, Any] = {}
        self.unknown_label = unknown_label
        self.fitted: bool = False

    def fit(self, series: pd.Series) -> None:
        uniques = sorted(series.dropna().unique())
        self.mapping = {val: idx for idx, val in enumerate(uniques)}
        self.inverse = {idx: val for val, idx in self.mapping.items()}
        self.fitted = True

    def transform(self, series: Union[pd.Series, List[Any]]) -> Union[pd.Series, List[int]]:
        if not self.fitted:
            raise ValueError("AutoLabelEncoder must be fit before transform.")
        if isinstance(series, list):
            return [self.mapping.get(val, self.unknown_label) for val in series]
        elif isinstance(series, pd.Series):
            return series.map(self.mapping).fillna(self.unknown_label).astype(int)
        else:
            raise TypeError("Input to transform() must be a pandas Series or a list.")

    def inverse_transform(self, series: Union[pd.Series, List[int]]) -> Union[pd.Series, List[Any]]:
        if not self.fitted:
            raise ValueError("AutoLabelEncoder must be fit before inverse_transform.")
        if isinstance(series, list):
            return [self.inverse.get(val, None) for val in series]
        elif isinstance(series, pd.Series):
            return series.map(self.inverse)
        else:
            raise TypeError("Input to inverse_transform() must be a pandas Series or a list.")

    def get_mapping(self) -> Dict[Any, int]:
        return self.mapping

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                "mapping": self.mapping,
                "inverse": self.inverse,
                "unknown_label": self.unknown_label
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "AutoLabelEncoder":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        encoder = cls(unknown_label=data.get("unknown_label", -1))
        encoder.mapping = data["mapping"]
        encoder.inverse = data["inverse"]
        encoder.fitted = True
        return encoder


class TargetMeanEncoder(BaseEncoder):
    """
    Target mean encoder with optional smoothing and noise injection.
    Useful for encoding high-cardinality categorical variables.

    Args:
        k (float): Regularization parameter. Higher k gives more weight to prior.
        f (float): Scaling parameter for smoothing.
        noise_level (float): Magnitude of Gaussian noise to add.
        random_state (int, optional): Random seed for reproducibility.
    """

    def __init__(self, k: float = 1.0, f: float = 1.0,
                 noise_level: float = 0.0, random_state: int = None):
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.random_state = random_state

        self.encodings: Dict[str, Dict[Any, float]] = {}
        self.prior: float = 0.0
        self.columns: List[str] = []
        self.fitted = False

    def fit(self, series: pd.Series, target: pd.Series = None) -> None:
        if target is None:
            raise ValueError("TargetMeanEncoder requires a target variable during fit.")

        self.columns = [series.name]
        df = pd.DataFrame({"cat": series, "target": target})
        self.prior = target.mean()

        stats = df.groupby("cat")["target"].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(stats["count"] - self.k) / self.f))
        smoothed = self.prior * (1 - smoothing) + stats["mean"] * smoothing

        self.encodings[series.name] = smoothed.to_dict()
        self.fitted = True

    def transform(self, series: Union[pd.Series, List[Any]]) -> Union[pd.Series, List[float]]:
        if not self.fitted:
            raise RuntimeError("You must fit the encoder before calling transform().")

        mapping = self.encodings[self.columns[0]]
        if isinstance(series, list):
            encoded = [mapping.get(x, self.prior) for x in series]
        else:
            encoded = series.map(lambda x: mapping.get(x, self.prior)).astype(float)

        if self.noise_level > 0:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            noise = np.random.randn(len(encoded)) * self.noise_level
            encoded = encoded + noise

        return encoded

    def inverse_transform(
            self, series: Union[pd.Series, List[float]]) -> Union[pd.Series, List[Any]]:
        raise NotImplementedError("TargetMeanEncoder does not support inverse_transform().")

    def get_mapping(self) -> Dict[Any, float]:
        if not self.fitted:
            raise RuntimeError("You must fit the encoder before calling get_mapping().")
        return self.encodings[self.columns[0]]

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                "k": self.k,
                "f": self.f,
                "noise_level": self.noise_level,
                "random_state": self.random_state,
                "prior": self.prior,
                "encodings": self.encodings,
                "columns": self.columns,
                "fitted": self.fitted,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "TargetMeanEncoder":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        encoder = cls(
            k=data["k"],
            f=data["f"],
            noise_level=data["noise_level"],
            random_state=data["random_state"]
        )
        encoder.prior = data["prior"]
        encoder.encodings = data["encodings"]
        encoder.columns = data["columns"]
        encoder.fitted = data["fitted"]
        return encoder
