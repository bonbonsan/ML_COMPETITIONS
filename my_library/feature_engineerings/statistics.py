from typing import List, Optional

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from my_library.utils.df_utils import df_io_polars


@df_io_polars(return_type="polars")
def compute_time_diff_stats(
    df: pl.DataFrame,
    user_col: str,
    time_col: str
) -> pl.DataFrame:
    """
    Compute statistical summaries of time intervals between actions for each user.

    Parameters:
        df: Input log data.
        user_col: Column name for user ID.
        time_col: Column name for timestamp (datetime type).

    Returns:
        Time interval statistics (mean, median, std, max).
    """
    df = df.sort([user_col, time_col])
    df = df.with_columns([
        (
            pl.col(time_col)
            .cast(pl.Datetime)
            .diff()
            .over(user_col)
            .dt.total_seconds()
        ).alias("time_diff_sec")
    ])
    return df.group_by(user_col).agg([
        pl.col("time_diff_sec").mean().alias("mean_interval"),
        pl.col("time_diff_sec").median().alias("median_interval"),
        pl.col("time_diff_sec").std().alias("std_interval"),
        pl.col("time_diff_sec").max().alias("max_interval"),
    ])


@df_io_polars(return_type="polars")
def compute_grouped_momentum_ratio(
    df: pl.DataFrame,
    group_cols: List[str],
    value_col: str,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute momentum as the ratio of the current value to the group-wise mean.

    Args:
        df: Input DataFrame.
        group_cols: Columns to group by (e.g., store_id, month).
        value_col: Value column (e.g., sales, price).
        output_col: Optional output column name.

    Returns:
        DataFrame with added momentum column.
    """
    if output_col is None:
        output_col = f"{value_col}_momentum_" + "_".join(group_cols)

    return df.with_columns([
        (pl.col(value_col) / pl.col(value_col).mean().over(group_cols)).alias(output_col)
    ])


@df_io_polars(return_type="polars")
def compute_basic_stats_features(
    df: pl.DataFrame,
    group_cols: List[str],
    value_col: str,
    prefix: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute basic statistical summaries (mean, std, min, max, median, skew, kurtosis, quantiles)
    per group.

    Args:
        df: Input DataFrame.
        group_cols: Columns to group by.
        value_col: Target value column.
        prefix: Optional prefix for output columns.

    Returns:
        DataFrame with statistical features.
    """
    if prefix is None:
        prefix = value_col

    return df.group_by(group_cols).agg([
        pl.col(value_col).mean().alias(f"{prefix}_mean"),
        pl.col(value_col).std().alias(f"{prefix}_std"),
        pl.col(value_col).min().alias(f"{prefix}_min"),
        pl.col(value_col).max().alias(f"{prefix}_max"),
        pl.col(value_col).median().alias(f"{prefix}_median"),
        pl.col(value_col).skew().alias(f"{prefix}_skew"),
        pl.col(value_col).kurtosis().alias(f"{prefix}_kurtosis"),
        pl.col(value_col).quantile(0.01).alias(f"{prefix}_q01"),
        pl.col(value_col).quantile(0.05).alias(f"{prefix}_q05"),
        pl.col(value_col).quantile(0.25).alias(f"{prefix}_q25"),
        pl.col(value_col).quantile(0.75).alias(f"{prefix}_q75"),
        pl.col(value_col).quantile(0.95).alias(f"{prefix}_q95"),
        pl.col(value_col).quantile(0.99).alias(f"{prefix}_q99"),
    ])


@df_io_polars(return_type="polars")
def compute_change_rate_features(
    df: pl.DataFrame,
    group_cols: List[str],
    value_col: str,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute mean absolute change and mean rate of change per group.

    Args:
        df: Input DataFrame.
        group_cols: Columns to group by.
        value_col: Target column.
        output_col: Optional prefix for output columns.

    Returns:
        DataFrame with change-related features.
    """
    if output_col is None:
        output_col = value_col

    df = df.sort(group_cols)
    diff_col = f"{output_col}_diff"
    rate_col = f"{output_col}_rate"

    df = df.with_columns([
        pl.col(value_col).diff().over(group_cols).alias(diff_col),
        (
            pl.col(value_col).diff().over(group_cols)
            / pl.col(value_col).shift(1).over(group_cols)
        ).alias(rate_col),
    ])

    return df.group_by(group_cols).agg([
        pl.col(diff_col).abs().mean().alias(f"{output_col}_mean_abs_diff"),
        pl.col(rate_col).mean().alias(f"{output_col}_mean_change_rate"),
    ])


@df_io_polars(return_type="polars")
def compute_trend_features(
    df: pl.DataFrame,
    group_cols: List[str],
    value_col: str,
    time_col: str,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Fit a linear trend (slope) to value vs. time per group using sklearn LinearRegression.

    Args:
        df: Input DataFrame.
        group_cols: Grouping columns.
        value_col: Column to model.
        time_col: Time column (must be datetime).
        output_col: Optional column name for output.

    Returns:
        DataFrame with trend (slope) feature per group.
    """
    if output_col is None:
        output_col = f"{value_col}_trend"

    results = []
    for group_key, group_df in df.group_by(group_cols):
        ts = group_df[time_col].cast(pl.Datetime).to_numpy()
        y = group_df[value_col].to_numpy()

        # 時間をUNIX秒に変換
        X = np.array([t.timestamp() for t in ts]).reshape(-1, 1)

        if len(X) < 2:
            slope = np.nan
        else:
            lr = LinearRegression()
            lr.fit(X, y)
            slope = lr.coef_[0]

        results.append((*group_key, slope))

    return pl.DataFrame(results, schema=group_cols + [output_col])
