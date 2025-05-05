import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from my_library.ensembles.stacking import StackingEnsembler


# pytest my_library/tests/unit/ensembles/test_stacking.py -v
def test_regression_without_meta_features():
    # Prepare base predictions and target
    y = np.array([1.0, 2.0, 3.0])
    preds1 = y.copy()                       # first feature equals y
    preds2 = np.zeros_like(y)               # second feature zeros

    # Use LinearRegression without intercept to ensure exact recovery
    meta = LinearRegression(fit_intercept=False)
    ensembler = StackingEnsembler(meta_model=meta)
    out = ensembler.ensemble([preds1, preds2], y_true=y)

    # Output should match the true target exactly
    assert isinstance(out, np.ndarray)
    assert out.shape == y.shape
    assert np.allclose(out, y)

    # evaluate should compute zero MSE
    mse = ensembler.evaluate(y, mean_squared_error)
    assert pytest.approx(0.0) == mse

def test_regression_with_meta_features_various_input_types(tmp_path):
    # Base prediction from a single model
    preds_np = np.array([1.0, 2.0, 3.0])

    # Meta features in pandas DataFrame
    meta_pd = pd.DataFrame({"feat": [2.0, 4.0, 6.0]})
    y_pd = preds_np + meta_pd["feat"].values

    # Meta features in Polars DataFrame
    meta_pl = pl.DataFrame({"feat": [2.0, 4.0, 6.0]})
    y_pl = preds_np + meta_pl["feat"].to_numpy()

    meta = LinearRegression(fit_intercept=False)

    # Test with pandas meta features
    ensembler_pd = StackingEnsembler(meta_model=meta)
    out_pd = ensembler_pd.ensemble([preds_np], X_meta_features=meta_pd, y_true=y_pd)
    assert isinstance(out_pd, np.ndarray)
    assert out_pd.shape == (3,)
    assert np.allclose(out_pd, y_pd)

    # Test with Polars meta features
    ensembler_pl = StackingEnsembler(meta_model=meta)
    out_pl = ensembler_pl.ensemble([preds_np], X_meta_features=meta_pl, y_true=y_pl)
    assert isinstance(out_pl, np.ndarray)
    assert out_pl.shape == (3,)
    assert np.allclose(out_pl, y_pl)

    # Test save_prediction writes correct CSV
    out_path = tmp_path / "stacking.csv"
    ensembler_pl.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    # Expect one column of predictions
    assert df.shape == (3, 1)
    assert list(df.iloc[:, 0]) == pytest.approx(list(y_pl))

def test_error_conditions_and_usage_before_ensemble():
    # Missing y_true should raise ValueError
    ensembler = StackingEnsembler()
    with pytest.raises(ValueError):
        ensembler.ensemble([np.array([1.0, 2.0, 3.0])])

    # evaluate and save_prediction before calling ensemble should raise ValueError
    ensembler2 = StackingEnsembler(meta_model=LinearRegression())
    with pytest.raises(ValueError):
        ensembler2.evaluate(np.array([0, 1]), mean_squared_error)
    with pytest.raises(ValueError):
        ensembler2.save_prediction("no.csv")
