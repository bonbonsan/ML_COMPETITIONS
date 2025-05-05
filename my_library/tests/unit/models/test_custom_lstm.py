# my_library/tests/unit/models/test_custom_lstm.py

from pathlib import Path

import pandas as pd
import pytest

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.lstm_configs import LSTMConfig
from my_library.models.custom_lstm import CustomLSTMModel
from my_library.utils.data_loader import load_sample_data


# pytest my_library/tests/unit/models/test_custom_lstm.py -v
@pytest.mark.parametrize("task_type,dataset_name,target_column", [
    ("classification", "iris", "target"),
    ("regression", "diabetes", "target"),
])
def test_custom_lstm_model(task_type, dataset_name, target_column, tmp_path: Path):
    # Load sample data
    df = load_sample_data(name=dataset_name, task=task_type)
    X = df.drop(columns=target_column)
    y = df[target_column]
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

    # Define config
    config = LSTMConfig(
        model_name="LSTMModelTest",
        task_type=task_type,
        use_gpu=False,
        save_log=False,
        params={
            "input_size": X.shape[1],
            "hidden_size": 16,
            "num_layers": 1,
            "output_size": len(y.unique()) if task_type == "classification" else 1,
            "activation": None,
            "dropout": 0.0,
            "lr": 0.01,
            "epochs": 5,
            "batch_size": 16,
        }
    )

    model = CustomLSTMModel(config=config)
    model.fit(X, y, fit_config=fit_config)

    # Predict
    preds = model.predict(X)
    assert isinstance(preds, pd.Series)
    assert preds.shape[0] == X.shape[0]
    assert preds.name == y.name or preds.name == "target"

    # Predict probabilities for classification
    if task_type == "classification":
        proba = model.predict_proba(X)
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape[0] == X.shape[0]
        assert abs(proba.sum(axis=1) - 1).max() < 1e-6
    else:
        with pytest.raises(NotImplementedError):
            _ = model.predict_proba(X)

    # Get top features
    top_feats = model.get_top_features(top_n=5)
    assert isinstance(top_feats, list)
    assert len(top_feats) <= 5
    assert all(
        isinstance(f, tuple)
        and isinstance(f[0], str)
        and isinstance(f[1], (int, float))
        for f in top_feats
    )

    # Save/load
    path = tmp_path / f"model_{dataset_name}_{task_type}.pt"
    model.save_model(str(path))
    assert path.exists()

    new_model = CustomLSTMModel(config=config)
    new_model.load_model(str(path))

    new_preds = new_model.predict(X)
    assert isinstance(new_preds, pd.Series)
    assert new_preds.shape[0] == X.shape[0]
    assert new_preds.name == preds.name
