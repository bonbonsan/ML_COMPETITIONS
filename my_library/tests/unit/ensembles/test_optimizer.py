import numpy as np
import pandas as pd
import polars as pl
import pytest

from my_library.ensembles.optimizer import WeightOptimizer


# pytest my_library/tests/unit/ensembles/test_optimizer.py -v
# A simple mean squared error metric (lower is better).
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def test_to_numpy_with_numpy_array():
    """_to_numpy should return the same numpy array."""
    opt = WeightOptimizer(metric_fn=mse)
    arr = np.array([1, 2, 3])
    out = opt._to_numpy(arr)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr)

def test_to_numpy_with_pandas_series_and_df():
    """_to_numpy should handle pandas Series and DataFrame."""
    opt = WeightOptimizer(metric_fn=mse)
    ser = pd.Series([1, 2, 3])
    df = pd.DataFrame({'a': [4, 5, 6]})
    assert np.array_equal(opt._to_numpy(ser), ser.values)
    assert np.array_equal(opt._to_numpy(df), df.values)

def test_to_numpy_with_polars_series_and_df():
    """_to_numpy should handle polars Series and DataFrame."""
    opt = WeightOptimizer(metric_fn=mse)
    ser = pl.Series([1, 2, 3])
    df = pl.DataFrame({'a': [4, 5, 6]})
    assert np.array_equal(opt._to_numpy(ser), ser.to_numpy())
    assert np.array_equal(opt._to_numpy(df), df.to_numpy())

def test_to_numpy_with_list_and_tuple():
    """_to_numpy should convert Python lists and tuples to numpy arrays."""
    opt = WeightOptimizer(metric_fn=mse)
    lst = [1, 2, 3]
    tpl = (4, 5, 6)
    assert np.array_equal(opt._to_numpy(lst), np.array(lst))
    assert np.array_equal(opt._to_numpy(tpl), np.array(tpl))

def test_to_numpy_raises_type_error_for_unsupported():
    """_to_numpy should raise TypeError for unsupported types."""
    opt = WeightOptimizer(metric_fn=mse)
    with pytest.raises(TypeError):
        opt._to_numpy("unsupported")

def test_optimize_equal_predictions():
    """
    When all model predictions are identical,
    the optimized weights should be equal.
    """
    y_true = np.array([1, 2, 3, 4])
    pred = np.array([2, 2, 2, 2])
    opt = WeightOptimizer(metric_fn=mse)
    weights = opt.optimize(predictions=[pred, pred, pred], y_true=y_true)
    assert weights == pytest.approx([1/3, 1/3, 1/3], rel=1e-6)

def test_optimize_prefers_better_model():
    """
    When one model predicts perfectly and the other is poor,
    the optimizer should assign higher weight to the perfect model.
    """
    y_true = np.array([1, 2, 3, 4])
    perfect = y_true
    bad = np.zeros_like(y_true)
    opt = WeightOptimizer(metric_fn=mse)
    weights = opt.optimize(predictions=[perfect, bad], y_true=y_true)
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0, rel=1e-6)

def test_optimize_without_sum_constraint():
    """
    When constraint_sum_to_one=False and default bounds,
    the optimizer will push the perfect model to its upper bound (1.0)
    and leave the other model's weight at its initial value (1/n_models).
    """
    y_true = np.array([1, 2, 3])
    pred1 = y_true
    pred2 = np.zeros_like(y_true)
    opt = WeightOptimizer(metric_fn=mse)
    weights = opt.optimize(
        predictions=[pred1, pred2],
        y_true=y_true,
        constraint_sum_to_one=False
    )
    # perfect model w1 -> 1.0, bad model w2 -> initial 0.5 (since n_models=2)
    assert weights[0] == pytest.approx(1.0, rel=1e-6)
    assert weights[1] == pytest.approx(0.5, rel=1e-6)
    # and since no sum-to-one constraint, sum should be > 1.0
    assert sum(weights) == pytest.approx(1.5, rel=1e-6)

def test_optimize_with_bounds():
    """
    Bounds should restrict each weight to the specified interval,
    and sum-to-one constraint forces [0.5, 0.5] in this symmetric case.
    """
    y_true = np.array([1, 2])
    pred1 = np.array([1, 2])
    pred2 = np.array([1, 2])
    opt = WeightOptimizer(metric_fn=mse)
    weights = opt.optimize(
        predictions=[pred1, pred2],
        y_true=y_true,
        bounds=[(0.0, 0.5), (0.5, 1.0)]
    )
    # Only feasible solution is w1=0.5, w2=0.5 under sum-to-one
    assert weights == pytest.approx([0.5, 0.5], rel=1e-6)

def test_loss_greater_is_better():
    """
    If loss_greater_is_better=True, the optimizer maximizes the metric_fn.
    """
    # Define a "score" metric where higher is better (negative MSE).
    def score(y_true, y_pred):
        return -np.mean((y_true - y_pred) ** 2)

    y_true = np.array([1, 2, 3])
    perfect = y_true
    bad = np.zeros_like(y_true)
    opt = WeightOptimizer(metric_fn=score, loss_greater_is_better=True)
    weights = opt.optimize(predictions=[perfect, bad], y_true=y_true)
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0, rel=1e-6)
