# pytest my_library/tests/unit/models/test_custom_knn.py -v

from pathlib import Path

import pandas as pd
import pytest

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.knn_configs import KNNConfig
from my_library.models.custom_knn import CustomKNN
from my_library.utils.data_loader import load_sample_data


# pytest my_library/tests/unit/models/test_custom_knn.py -v
@pytest.mark.parametrize("task_type,dataset_name,target_column", [
    ("classification", "iris",    "target"),
    ("regression",    "diabetes", "target"),
])
def test_custom_knn_model(task_type, dataset_name, target_column, tmp_path: Path):
    # Load sample data
    df = load_sample_data(name=dataset_name, task=task_type)
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Prepare fit configuration (no feature subset, no early stopping)
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

    # Instantiate model configuration and model
    config = KNNConfig(task_type=task_type, save_log=False)
    model = CustomKNN(config=config)

    # Fit the model
    model.fit(X, y, fit_config=fit_config)

    # ---- Test predict ----
    preds = model.predict(X)
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == X.shape[0]
    assert preds.name == y.name or preds.name == "target"

    # ---- Test predict_proba ----
    if task_type == "classification":
        proba = model.predict_proba(X)
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape[0] == X.shape[0]
        # Each row probabilities sum to ~1
        assert abs(proba.sum(axis=1) - 1).max() < 1e-6
    else:
        with pytest.raises(NotImplementedError):
            _ = model.predict_proba(X)

    # ---- Test feature importance ----
    top_feats = model.get_top_features(top_n=5)
    assert isinstance(top_feats, list)
    assert len(top_feats) <= 5
    assert all(
        isinstance(f, tuple) and isinstance(f[0], str) and isinstance(f[1], (int, float))
        for f in top_feats
    )

    # ---- Test save/load ----
    path = tmp_path / f"custom_knn_{dataset_name}_{task_type}.pkl"
    model.save_model(str(path))
    assert path.exists()

    new_model = CustomKNN(config=config)
    new_model.load_model(str(path))
    new_preds = new_model.predict(X)
    assert isinstance(new_preds, pd.Series)
    assert new_preds.shape[0] == X.shape[0]
    assert new_preds.name == preds.name
