import numpy as np
import polars as pl
import pytest

from my_library.utils.data_loader import load_sample_data
from my_library.validations.adversarial_validator import AdversarialValidator


# pytest my_library/tests/unit/validations/test_adversarial_validator.py -v
@pytest.mark.parametrize("name,task", [
    ("iris", "classification"),
    ("diabetes", "regression"),
])
def test_adversarial_validator_on_sample_data(name, task):
    # Load sample dataset
    df = load_sample_data(name=name, task=task)

    # Drop target column if exists
    feature_cols = [col for col in df.columns if col != "target"]
    real_X = pl.from_pandas(df[feature_cols])

    # Generate dummy data with same shape and similar ranges
    np.random.seed(42)
    dummy_data = {
        col: np.random.uniform(
            low=real_X[col].min(),
            high=real_X[col].max(),
            size=real_X.height
        )
        for col in real_X.columns
    }
    dummy_X = pl.DataFrame(dummy_data)

    # Instantiate and run validator
    validator = AdversarialValidator(scoring="roc_auc")
    validator.fit(train_X=real_X, test_X=dummy_X)
    score = validator.validate()

    # Assert AUC is reasonably high (indicating distinguishable)
    assert score > 0.8, f"Expected high AUC, got {score:.4f} for dataset: {name}"

    # Feature importance check
    importance_df = validator.get_feature_importance()
    assert importance_df.shape[0] > 0
    assert set(importance_df.columns) == {"feature", "importance"}
