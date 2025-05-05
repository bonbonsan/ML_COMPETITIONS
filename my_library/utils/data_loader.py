import os

import pandas as pd


def load_sample_data(name: str, task: str = "classification") -> pd.DataFrame:
    """Load sample dataset from local storage.

    Args:
        name (str): The dataset file name (without .csv extension).
        task (str): 'classification' or 'regression'.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'samples'))
    file_path = os.path.join(root, task, f"{name}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)
