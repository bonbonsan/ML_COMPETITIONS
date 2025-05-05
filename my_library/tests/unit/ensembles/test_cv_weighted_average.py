import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import log_loss, mean_squared_error

from my_library.ensembles.cv_weighted_average import CVWeightAveragingEnsembler


# pytest my_library/tests/unit/ensembles/test_cv_weighted_average.py -v
def test_regression_equal_scores():
    # Two predictors with identical MSE -> equal weights
    preds1 = np.array([1.0, 2.0, 3.0])
    preds2 = np.array([3.0, 2.0, 1.0])
    y_true = np.array([2.0, 2.0, 2.0])

    ensembler = CVWeightAveragingEnsembler()
    out = ensembler.ensemble([preds1, preds2], y_true, mean_squared_error)

    # Weights should be equal (0.5, 0.5)
    assert isinstance(ensembler.weights, np.ndarray)
    assert ensembler.weights.shape == (2,)
    assert pytest.approx(ensembler.weights.tolist(), rel=1e-6) == [0.5, 0.5]

    # Output should be average of predictions: [2, 2, 2]
    expected = np.array([2.0, 2.0, 2.0])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, expected)

    # final_prediction attribute should match returned array
    assert np.allclose(ensembler.final_prediction, out)


def test_classification_equal_scores():
    # Two predictors with identical log loss -> equal weights
    preds1 = np.array([[0.5, 0.5], [0.5, 0.5]])
    preds2 = np.array([[0.5, 0.5], [0.5, 0.5]])
    y_true = np.array([0, 1])

    ensembler = CVWeightAveragingEnsembler()
    out = ensembler.ensemble([preds1, preds2], y_true, log_loss)

    # Weights should be equal (0.5, 0.5)
    assert pytest.approx(ensembler.weights.tolist(), rel=1e-6) == [0.5, 0.5]

    # Output should be identical to any input predictor
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert np.allclose(out, preds1)


def test_arraylike_variants_equal_mse():
    # Test using list, pandas Series, and polars Series
    preds_list = [1.0, 2.0, 3.0]
    preds_pd = pd.Series([3.0, 2.0, 1.0])
    preds_pl = pl.Series("p", [3.0, 2.0, 1.0])
    y_true_list = [2.0, 2.0, 2.0]

    ensembler = CVWeightAveragingEnsembler()
    out = ensembler.ensemble([preds_list, preds_pd, preds_pl], y_true_list, mean_squared_error)

    # All predictors have same MSE -> weights equal (1/3 each)
    expected_weight = 1.0 / 3.0
    assert pytest.approx(ensembler.weights.tolist(), rel=1e-6) == [expected_weight] * 3
    # Weighted average of the three prediction‐arrays
    expected_out = (np.array(preds_list)
                    + preds_pd.values
                    + preds_pl.to_numpy()) / 3
    assert np.allclose(out, expected_out)

def test_shape_mismatch_error():
    # Predictions and truth length mismatch should raise an error
    preds = [np.array([1, 2, 3]), np.array([1, 2])]
    y_true = np.array([1, 2, 3])
    ensembler = CVWeightAveragingEnsembler()
    with pytest.raises(ValueError):
        ensembler.ensemble(preds, y_true, mean_squared_error)

def test_empty_predictions_error():
    # No predictors provided should raise an error
    y_true = np.array([1.0, 2.0, 3.0])
    ensembler = CVWeightAveragingEnsembler()
    with pytest.raises(ValueError):
        ensembler.ensemble([], y_true, mean_squared_error)

def test_non_callable_metric_error():
    # Passing a non-callable as metric_fn should raise a TypeError
    preds1 = np.array([1.0, 2.0, 3.0])
    preds2 = np.array([3.0, 2.0, 1.0])
    y_true = np.array([2.0, 2.0, 2.0])
    ensembler = CVWeightAveragingEnsembler()
    with pytest.raises(TypeError):
        ensembler.ensemble([preds1, preds2], y_true, None)
