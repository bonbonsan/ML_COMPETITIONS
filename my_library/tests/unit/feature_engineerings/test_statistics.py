import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from my_library.feature_engineerings.statistics import (
    compute_basic_stats_features,
    compute_change_rate_features,
    compute_grouped_momentum_ratio,
    compute_time_diff_stats,
    compute_trend_features,
)


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Helper to convert polars output to pandas for assertion stability."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def test_compute_basic_stats_features_returns_expected_metrics():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1, 3, 2, 4],
        }
    )

    result = compute_basic_stats_features(df, ["group"], "value", prefix="value")

    assert isinstance(result, pl.DataFrame)
    result_pd = _to_pandas(result).set_index("group")

    assert result_pd.loc["A", "value_mean"] == pytest.approx(2.0)
    assert result_pd.loc["A", "value_min"] == 1
    assert result_pd.loc["A", "value_max"] == 3
    assert result_pd.loc["A", "value_median"] == 2.0
    assert result_pd.loc["B", "value_std"] == pytest.approx(np.std([2, 4], ddof=1))


def test_compute_grouped_momentum_ratio_maintains_input_order_with_pandas_input():
    df = pd.DataFrame(
        {
            "store": ["s1", "s1", "s2", "s2"],
            "sales": [10.0, 20.0, 5.0, 15.0],
        }
    )

    result = compute_grouped_momentum_ratio(df, ["store"], "sales")

    assert isinstance(result, pl.DataFrame)
    result_pd = _to_pandas(result)

    momentum_col = "sales_momentum_store"
    assert momentum_col in result_pd.columns

    group_means = df.groupby("store")["sales"].transform("mean")
    expected = df["sales"] / group_means
    np.testing.assert_allclose(result_pd[momentum_col], expected)


def test_compute_time_diff_stats_resets_between_users():
    df = pd.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2"],
            "event_time": [
                datetime.datetime(2024, 1, 1, 0, 0, 0),
                datetime.datetime(2024, 1, 1, 0, 5, 0),
                datetime.datetime(2024, 1, 2, 0, 0, 0),
                datetime.datetime(2024, 1, 2, 0, 10, 0),
            ],
        }
    )

    result = compute_time_diff_stats(df, user_col="user", time_col="event_time")

    assert isinstance(result, pl.DataFrame)
    result_pd = _to_pandas(result).set_index("user")

    assert result_pd.loc["u1", "mean_interval"] == pytest.approx(300.0)
    assert result_pd.loc["u2", "mean_interval"] == pytest.approx(600.0)
    # Ensure statistics ignore cross-user gaps
    assert result_pd.loc["u1", "max_interval"] < 600.0


def test_compute_change_rate_features_handles_multi_group_data():
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "value": [10.0, 12.0, 18.0, 5.0, 15.0],
        }
    )

    result = compute_change_rate_features(df, ["group"], "value")

    assert isinstance(result, pl.DataFrame)
    result_pd = _to_pandas(result).set_index("group")

    assert result_pd.loc["A", "value_mean_abs_diff"] == pytest.approx(4.0)
    assert result_pd.loc["A", "value_mean_change_rate"] == pytest.approx(0.35)
    assert result_pd.loc["B", "value_mean_abs_diff"] == pytest.approx(10.0)
    assert result_pd.loc["B", "value_mean_change_rate"] == pytest.approx(2.0)


def test_compute_trend_features_returns_per_group_slopes():
    dates = [
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 1, 2),
        datetime.datetime(2024, 1, 3),
    ]
    df = pl.DataFrame(
        {
            "store": ["A", "A", "A", "B"],
            "date": dates + [datetime.datetime(2024, 1, 1)],
            "value": [10.0, 13.0, 16.0, 5.0],
        }
    )

    result = compute_trend_features(df, ["store"], "value", "date")

    assert isinstance(result, pl.DataFrame)
    result_pd = _to_pandas(result).set_index("store")

    # Expected slope for store A computed through linear regression for stability
    timestamps = np.array([dt.timestamp() for dt in dates]).reshape(-1, 1)
    lr = LinearRegression().fit(timestamps, np.array([10.0, 13.0, 16.0]))
    expected_slope = lr.coef_[0]

    assert result_pd.loc["A", "value_trend"] == pytest.approx(expected_slope)
    assert np.isnan(result_pd.loc["B", "value_trend"])
