from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import pickle
import os


class BaseEncoder(ABC):
    """
    Abstract base class for label encoders.
    """

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Learns encoding from the given series (if applicable)."""
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """Encodes values in the given series using internal mapping."""
        pass

    @abstractmethod
    def inverse_transform(self, series: pd.Series) -> pd.Series:
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
        return series.map(self.mapping).fillna(self.unknown_label).astype(int)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
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

    def transform(self, series: pd.Series) -> pd.Series:
        if not self.fitted:
            raise ValueError("AutoLabelEncoder must be fit before transform.")
        return series.map(self.mapping).fillna(self.unknown_label).astype(int)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        if not self.fitted:
            raise ValueError("AutoLabelEncoder must be fit before inverse_transform.")
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
