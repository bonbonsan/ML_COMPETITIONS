import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from my_library.ensembles.blending import BlendingEnsembler


# pytest my_library/tests/unit/ensembles/test_blending.py -v
def test_regression_with_mixed_arraylike(tmp_path):
    # Prepare holdout ground truth and two base model predictions in different ArrayLike formats
    y_true = np.array([1.0, 2.0, 3.0])
    preds_pd = pd.Series(y_true, name="pred1")
    preds_pl = pl.Series("pred2", y_true.tolist())

    ensembler = BlendingEnsembler()
    # Fit the meta-model on the holdout set and get final predictions
    y_pred = ensembler.ensemble([preds_pd, preds_pl], y_true)

    # The meta-model should learn to recover y_true exactly
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_true.shape
    assert np.allclose(y_pred, y_true)

    # evaluate() should compute zero MSE on perfect predictions
    mse = ensembler.evaluate(y_true, mean_squared_error)
    assert pytest.approx(0.0) == mse

    # save_prediction() should write a single-column CSV matching y_true
    out_path = tmp_path / "blend_regression.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    assert df.shape == (3, 1)
    assert list(df.iloc[:, 0]) == pytest.approx(y_true.tolist())


def test_binary_classification_with_logistic_meta(tmp_path):
    # Prepare a simple binary classification holdout set
    y_true = np.array([0, 1, 0, 1])
    # Two base-model score arrays that correlate perfectly with the labels
    preds1 = np.array([0.1, 0.9, 0.2, 0.8])
    preds2 = np.array([0.2, 0.8, 0.1, 0.9])

    # Use a logistic regression meta-model for classification
    meta = LogisticRegression(solver="lbfgs", random_state=0)
    ensembler = BlendingEnsembler(meta_model=meta)
    y_pred = ensembler.ensemble([preds1, preds2], y_true)

    # The meta-model should classify perfectly
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_true.shape
    assert np.array_equal(y_pred, y_true)

    # evaluate() should compute 100% accuracy
    acc = ensembler.evaluate(y_true, accuracy_score)
    assert pytest.approx(1.0) == acc

    # save_prediction() should write a single-column CSV of class labels
    out_path = tmp_path / "blend_classification.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    assert df.shape == (4, 1)
    assert list(df.iloc[:, 0]) == pytest.approx(y_true.tolist())


def test_error_before_ensemble(tmp_path):
    ensembler = BlendingEnsembler()
    # Calling evaluate or save_prediction before ensemble() should raise ValueError
    with pytest.raises(ValueError):
        ensembler.evaluate(np.array([1, 2]), mean_squared_error)
    with pytest.raises(ValueError):
        ensembler.save_prediction(str(tmp_path / "noop.csv"))
