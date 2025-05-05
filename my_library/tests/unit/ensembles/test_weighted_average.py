import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import log_loss, mean_squared_error

from my_library.ensembles.weighted_average import WeightedAverageEnsembler


# pytest my_library/tests/unit/ensembles/test_weighted_average.py -v
def test_regression_numpy_pandas_polars(tmp_path):
    # Prepare three predictions in different ArrayLike フォーマット
    preds_np = np.array([1.0, 2.0, 3.0])
    preds_pd = pd.Series([3.0, 2.0, 1.0], name="pred")
    preds_pl = pl.Series("pred", [2.0, 2.0, 2.0])

    # Equal weight ensemble
    ensembler = WeightedAverageEnsembler(n_models=3)
    out = ensembler.ensemble([preds_np, preds_pd, preds_pl])
    # 全部平均すると [2,2,2]
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, [2.0, 2.0, 2.0])

    # evaluate も動く
    mse = ensembler.evaluate(np.array([2.0, 2.0, 2.0]), mean_squared_error)
    assert pytest.approx(0.0) == mse

    # save_prediction
    out_path = tmp_path / "reg.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    assert df.shape == (3, 1)  # 1 列
    assert list(df.iloc[:, 0]) == pytest.approx([2.0, 2.0, 2.0])


def test_classification_numpy_pandas_polars(tmp_path):
    # 2クラス分類の確率予測
    preds_np = np.array([[0.2, 0.8], [0.6, 0.4]])
    preds_pd = pd.DataFrame([[0.8, 0.2], [0.4, 0.6]], columns=["c0", "c1"])
    preds_pl = pl.DataFrame({"c0": [0.5, 0.5], "c1": [0.5, 0.5]})

    # カスタム重みで ensemble
    ensembler = WeightedAverageEnsembler(weights=[0.2, 0.3, 0.5])
    out = ensembler.ensemble([preds_np, preds_pd, preds_pl])
    # 期待値を手計算
    expected = 0.2 * preds_np + 0.3 * preds_pd.values + 0.5 * preds_pl.to_numpy()
    assert out.shape == (2, 2)
    assert np.allclose(out, expected)

    # log_loss も動く
    # 正解ラベル (one-hotではなくクラスID): [1, 0]
    ll = ensembler.evaluate(np.array([1, 0]), log_loss)
    assert ll >= 0.0

    # save_prediction (2列)
    out_path = tmp_path / "clf.csv"
    ensembler.save_prediction(str(out_path))
    df = pd.read_csv(out_path)
    assert df.shape == (2, 2)


def test_error_conditions():
    # 重みの長さ不一致
    ensembler = WeightedAverageEnsembler(weights=[0.5])
    with pytest.raises(ValueError):
        ensembler.ensemble([np.array([0, 1]), np.array([1, 0])])

    # ensemble 前の evaluate/save_prediction はエラー
    ensembler2 = WeightedAverageEnsembler(n_models=2)
    with pytest.raises(ValueError):
        ensembler2.evaluate([0, 1], mean_squared_error)
    with pytest.raises(ValueError):
        ensembler2.save_prediction("no.csv")
