import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error

from my_library.ensembles.ranking_average import RankAveragingEnsembler


# pytest my_library/tests/unit/ensembles/test_ranking_average.py -v
def test_regression_rank_averaging_numpy_pandas_polars(tmp_path):
    # Prepare three 1D predictions in different ArrayLike formats
    preds_np = np.array([1.0, 2.0, 3.0])
    preds_pd = pd.Series([3.0, 2.0, 1.0], name="pred")
    preds_pl = pl.Series("pred", [2.0, 3.0, 1.0])

    ensembler = RankAveragingEnsembler()
    out = ensembler.ensemble([preds_np, preds_pd, preds_pl])

    # Expected ranks per model:
    # np -> [1,2,3], pd -> [3,2,1], pl -> [2,3,1]
    # Averaged: [(1+3+2)/3, (2+2+3)/3, (3+1+1)/3]
    expected = np.array([2.0, 7/3, 5/3])

    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, expected)

    # evaluate should work (MSE with itself = 0)
    mse = ensembler.evaluate(expected, mean_squared_error)
    assert pytest.approx(0.0) == mse

    # save_prediction writes a single-column CSV
    out_path = tmp_path / "reg.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    assert df.shape == (3, 1)
    assert list(df.iloc[:, 0]) == pytest.approx(expected.tolist())


def test_classification_rank_averaging_flattened(tmp_path):
    # Prepare 2D predictions (e.g., class probabilities) in various formats
    preds_np = np.array([[1, 2], [3, 4]])
    preds_pd = pd.DataFrame([[4, 3], [2, 1]], columns=["c0", "c1"])
    preds_pl = pl.DataFrame({"c0": [2, 4], "c1": [3, 1]})

    ensembler = RankAveragingEnsembler()
    out = ensembler.ensemble([preds_np, preds_pd, preds_pl])

    # Flattened ranks per model:
    # model1: [1,2,3,4]
    # model2: [4,3,2,1]
    # model3: [2,3,4,1]
    # Averaged: [(1+4+2)/3, (2+3+3)/3, (3+2+4)/3, (4+1+1)/3]
    expected = np.array([7/3, 8/3, 3.0, 2.0])

    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert np.allclose(out, expected)

    # save_prediction should write the flattened array
    out_path = tmp_path / "cls.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    # one column, four rows
    assert df.shape == (4, 1)
    assert list(df.iloc[:, 0]) == pytest.approx(expected.tolist())


def test_error_conditions():
    # Mismatched lengths should raise ValueError
    ensembler = RankAveragingEnsembler()
    with pytest.raises(ValueError):
        ensembler.ensemble([np.array([1, 2]), np.array([1, 2, 3])])

    # evaluate and save_prediction before calling ensemble should raise
    ensembler2 = RankAveragingEnsembler()
    with pytest.raises(ValueError):
        ensembler2.evaluate([0, 1], mean_squared_error)
    with pytest.raises(ValueError):
        ensembler2.save_prediction("no.csv")
