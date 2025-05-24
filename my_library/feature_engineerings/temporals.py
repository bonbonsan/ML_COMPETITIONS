
from typing import List, Literal, Optional

import numpy as np
import polars as pl

from my_library.utils.df_utils import DataFrameType, df_io_polars


@df_io_polars(return_type="polars")
def compute_entity_time_gaps(
    df: DataFrameType,
    entity_col: str,
    timestamp_col: str,
    subgroup_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Compute time differences (in hours) between consecutive timestamps for:
        - each entity (e.g., user)
        - or each entity + subgroup combination (e.g., user x category X brand)

    Args:
        df: Input DataFrame.
        entity_col: Column name for the primary entity.
        timestamp_col: Column name for timestamp values (must be pl.Datetime).
        subgroup_cols: Optional list of columns to group additionally (nested subgrouping).
  
    Returns:
        pl.DataFrame: With one new column:
            - time_diff_hours[_subgroup1_subgroup2...]
    """
    if subgroup_cols is None:
        group_keys = [entity_col]
        suffix = ""
    else:
        group_keys = [entity_col] + subgroup_cols
        suffix = "_" + "_".join(subgroup_cols)

    output_col = f"time_diff_hours{suffix}"

    return df.with_columns([
        ((pl.col(timestamp_col) - pl.col(timestamp_col).shift(1)).dt.total_seconds() // 3600)
        .over(group_keys)
        .alias(output_col)
    ])


@df_io_polars(return_type="polars")
def compute_event_recency(
    df: DataFrameType,
    timestamp_col: str
) -> pl.DataFrame:
    """
    Compute recency of events (distance from latest timestamp) in:
        - days
        - hours

    Args:
        df: Input DataFrame.
        timestamp_col: Column containing timestamps.

    Returns:
        pl.DataFrame: With two new columns:
            - days_from_latest
            - hours_from_latest
    """
    ts_max = df[timestamp_col].max()
    return df.with_columns([
        (ts_max - pl.col(timestamp_col)).dt.total_days().alias("days_from_latest"),
        ((ts_max - pl.col(timestamp_col)).dt.total_seconds() // 3600).alias("hours_from_latest"),
    ])


@df_io_polars(return_type="polars")
def extract_timepoint_rows(
    df: pl.DataFrame,
    group_keys: List[str],
    time_col: str,
    mode: Literal["first", "last"] = "first"
) -> pl.DataFrame:
    """
    Extract rows corresponding to the earliest or latest time point within each group.

    Args:
        df (pl.DataFrame): Input dataframe.
        group_keys (List[str]): Columns to group by (e.g., ["顧客CD"]).
        time_col (str): Timestamp column name (e.g., "ts").
        mode (Literal["first", "last"]): Whether to extract "first" or "last" timepoint rows.

    Returns:
        pl.DataFrame: Subset of df where each group has only rows at its min or max time.
    """
    # Get first or last timestamp per group
    agg_expr = pl.col(time_col).min() if mode == "first" else pl.col(time_col).max()
    target_times = df.group_by(group_keys).agg([
        agg_expr.alias("__target_time__")
    ])

    # Join to original and filter
    joined = df.join(target_times, on=group_keys, how="inner")
    return joined.filter(pl.col(time_col) == pl.col("__target_time__")).drop("__target_time__")


@df_io_polars(return_type="polars")
def extract_weekday(
    df: pl.DataFrame, datetime_col: str, output_col: str = "weekday"
    ) -> pl.DataFrame:
    """
    Extract weekday (0=Monday, 6=Sunday) from a datetime column.

    Args:
        df: Polars DataFrame
        datetime_col: Name of datetime column
        output_col: Name of the new weekday column

    Returns:
        Polars DataFrame with appended weekday column
    """
    return df.with_columns([
        pl.col(datetime_col).dt.weekday().alias(output_col)
    ])


@df_io_polars(return_type="polars")
def is_weekend(
    df: pl.DataFrame, weekday_col: str = "weekday", output_col: str = "is_weekend"
    ) -> pl.DataFrame:
    """
    Add a boolean column indicating if the weekday is Saturday (5) or Sunday (6).

    Args:
        df: Polars DataFrame
        weekday_col: Column with weekday integers (0=Mon, 6=Sun)
        output_col: Output boolean column name

    Returns:
        DataFrame with added weekend flag
    """
    return df.with_columns([
        (pl.col(weekday_col).is_in([5, 6])).alias(output_col)
    ])


@df_io_polars(return_type="polars")
def add_weekday_cyclical_features(df: pl.DataFrame, datetime_col: str) -> pl.DataFrame:
    """
    Add sine and cosine cyclical encoding of weekday (0=Monday, 6=Sunday) to a Polars DataFrame.

    This is useful for capturing cyclical patterns in time-series models,
    especially when the day of the week has a periodic effect.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        datetime_col (str): Column name containing datetime values.

    Returns:
        pl.DataFrame: DataFrame with 'weekday_sin' and 'weekday_cos' columns added.
    
    Raises:
        ValueError: If the datetime column is not of dtype `pl.Datetime`.
    """
    if df[datetime_col].dtype != pl.Datetime:
        raise ValueError(f"Column '{datetime_col}' must be of type pl.Datetime")

    cycle = 7
    weekday_expr = df[datetime_col].dt.weekday()  # returns Int8 from 0 (Monday) to 6 (Sunday)

    df = df.with_columns([
        (2 * np.pi * weekday_expr / cycle).sin().alias("weekday_sin"),
        (2 * np.pi * weekday_expr / cycle).cos().alias("weekday_cos")
    ])

    return df


@df_io_polars(return_type="polars")
def compute_first_last_features(
    df: pl.DataFrame,
    user_col: str,
    time_col: str,
    event_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute the first and last timestamps (and optionally event types) per user.

    Parameters:
        df (pl.DataFrame): Input log data.
        user_col (str): Column name representing the user ID.
        time_col (str): Column name representing the timestamp (datetime type).
        event_col (str, optional): Column name for event type. If None, event is not processed.

    Returns:
        pl.DataFrame: Aggregated features including first/last timestamp and duration range.
    """
    sort_df = df.sort(time_col)
    aggs = [
        pl.col(time_col).first().alias("first_timestamp"),
        pl.col(time_col).last().alias("last_timestamp"),
        (pl.col(time_col).last() - pl.col(time_col).first()).dt.total_seconds()\
            .alias("time_range_sec"),
    ]
    if event_col:
        aggs.extend([
            pl.col(event_col).first().alias("first_event"),
            pl.col(event_col).last().alias("last_event"),
        ])
    return sort_df.group_by(user_col).agg(aggs)


@df_io_polars(return_type="polars")
def compute_time_diff_stats(
    df: pl.DataFrame,
    user_col: str,
    time_col: str
) -> pl.DataFrame:
    """
    Compute statistical summaries of time intervals between actions for each user.

    Parameters:
        df (pl.DataFrame): Input log data.
        user_col (str): Column name for user ID.
        time_col (str): Column name for timestamp (datetime type).

    Returns:
        pl.DataFrame: Time interval statistics (mean, median, std, max).
    """
    df = df.sort([user_col, time_col])
    df = df.with_columns([
        pl.col(time_col).cast(pl.Datetime).diff().dt.total_seconds().alias("time_diff_sec")
    ])
    return df.group_by(user_col).agg([
        pl.col("time_diff_sec").mean().alias("mean_interval"),
        pl.col("time_diff_sec").median().alias("median_interval"),
        pl.col("time_diff_sec").std().alias("std_interval"),
        pl.col("time_diff_sec").max().alias("max_interval"),
    ])


@df_io_polars(return_type="polars")
def compute_event_lag_features(
    df: pl.DataFrame,
    user_col: str,
    time_col: str,
    event_col: str,
    target_event: str
) -> pl.DataFrame:
    """
    Compute the time elapsed since the last occurrence of a target event.

    Parameters:
        df (pl.DataFrame): Input log data.
        user_col (str): Column name for user ID.
        time_col (str): Column name for timestamp (datetime type).
        event_col (str): Column name for event type.
        target_event (str): Specific event to track for lag computation.

    Returns:
        pl.DataFrame: Log with additional column 'since_last_event_sec'.
    """
    df = df.sort([user_col, time_col])
    df = df.with_columns([
        pl.when(pl.col(event_col) == target_event)
          .then(pl.col(time_col))
          .otherwise(None)
          .alias("event_time")
    ])
    df = df.with_columns([
        pl.when(pl.col(event_col) == target_event)
        .then(pl.col(time_col))
        .otherwise(None)
        .alias("event_time")
    ])

    # 「last_event_time」を forward fill 後に、新しい df として分けて保持
    df = df.with_columns([
        pl.col("event_time").fill_null(strategy="forward").alias("last_event_time")
    ])

    # 「last_event_time」ができた後に使う
    df = df.with_columns([
        (pl.col(time_col) - pl.col("last_event_time")).dt.total_seconds()\
            .alias("since_last_event_sec")
    ])

    return df.select([user_col, time_col, event_col, "since_last_event_sec"])


@df_io_polars(return_type="polars")
def compute_consecutive_days_features(
    df: pl.DataFrame,
    user_col: str,
    time_col: str
) -> pl.DataFrame:
    """
    Compute maximum streak of consecutive login days per user.

    Parameters:
        df (pl.DataFrame): Input log data.
        user_col (str): Column name for user ID.
        time_col (str): Column name for timestamp (datetime type).

    Returns:
        pl.DataFrame: User-wise maximum login streak.
    """
    df = df.with_columns([
        pl.col(time_col).dt.date().alias("date")
    ]).unique([user_col, "date"])

    df = df.sort([user_col, "date"])
    df = df.with_columns([
        (pl.col("date").cast(pl.Int32) - pl.col("date").cast(pl.Int32).shift(1)).alias("gap")
    ])

    def count_max_streak(gaps: List[int]) -> int:
        streak = max_streak = 1
        for g in gaps[1:]:
            if g == 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        return max_streak

    result = []
    for user_id, group in df.group_by([user_col]):
        gap_list = group["gap"].to_list()
        max_streak = count_max_streak(gap_list)
        result.append((user_id, max_streak))

    return pl.DataFrame(result, schema=[user_col, "max_login_streak"])


@df_io_polars(return_type="polars")
def compute_grouped_momentum_ratio(
    df: pl.DataFrame,
    group_cols: List[str],
    value_col: str,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute momentum as the ratio of the current value to the group-wise mean.
    Useful for expressing local trends such as price momentum relative to monthly or yearly average.

    Args:
        df: Polars DataFrame.
        group_cols: List of columns to group by (e.g., ["store_id", "item_id", "month"]).
        value_col: Name of the column to compute momentum on (e.g., "sell_price").
        output_col: Optional name for the output column.
                    Defaults to "<value_col>_momentum_<grouping>".

    Returns:
        DataFrame with added momentum column.
    """
    if output_col is None:
        output_col = f"{value_col}_momentum_" + "_".join(group_cols)

    return df.with_columns([
        (pl.col(value_col) / pl.col(value_col).mean().over(group_cols)).alias(output_col)
    ])


@df_io_polars(return_type="polars")
def compute_lag_features(
    df: pl.DataFrame,
    group_cols: List[str],
    target_col: str,
    lags: List[int]
) -> pl.DataFrame:
    """
    Compute lag features (e.g., previous day's value) over a specified group.

    Args:
        df: Input dataframe.
        group_cols: Columns to group by (e.g., entity, user_id).
        target_col: Column to apply lag on (e.g., "sales").
        lags: List of lag offsets (e.g., [1, 7, 14]).

    Returns:
        DataFrame with new lag columns.
    """
    df = df.with_columns([
        pl.col(target_col).shift(lag).over(group_cols).alias(f"{target_col}_lag_{lag}")
        for lag in lags
    ])
    return df


@df_io_polars(return_type="polars")
def compute_rolling_stats_features(
    df: pl.DataFrame,
    group_cols: List[str],
    target_col: str,
    window_sizes: List[int],
    shift: int = 1,
    stats: List[str] = None
) -> pl.DataFrame:
    """
    Efficient computation of rolling stats (mean, std, etc.) using Polars native rolling functions.

    Only supports fast built-in functions: mean, std, min, max, sum, median, q1, q3.

    Args:
        df: Input dataframe.
        group_cols: Columns to group by.
        target_col: Column to compute on.
        window_sizes: Rolling window sizes (e.g., [7, 14, 30]).
        shift: Shift before rolling to avoid leakage (e.g., 1 for prediction-safe).
        stats: List of stats to compute. Supported:
               "mean", "std", "min", "max", "sum", "median", "q1", "q3".

    Returns:
        DataFrame with new rolling feature columns.
    """
    if stats is None:
        stats = ["mean", "std"]
        
    supported_stats = {
        "mean": lambda s, w: s.rolling_mean(w),
        "std": lambda s, w: s.rolling_std(w),
        "min": lambda s, w: s.rolling_min(w),
        "max": lambda s, w: s.rolling_max(w),
        "sum": lambda s, w: s.rolling_sum(w),
        "median": lambda s, w: s.rolling_median(w),
        "q1": lambda s, w: s.rolling_quantile(
            window_size=w, quantile=0.25, interpolation="nearest"
            ),
        "q3": lambda s, w: s.rolling_quantile(
            window_size=w, quantile=0.75, interpolation="nearest"
            ),
    }

    exprs = []
    for win in window_sizes:
        shifted = pl.col(target_col).shift(shift)
        for stat in stats:
            if stat not in supported_stats:
                raise ValueError(f"Unsupported stat: {stat}")
            expr = supported_stats[stat](shifted, win).over(group_cols).alias(
                f"{target_col}_rolling_{stat}_{win}"
            )
            exprs.append(expr)

    return df.with_columns(exprs)


if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta

    # --- サンプルデータの作成 ---
    random.seed(42)
    user_ids = ["A", "B", "C"]
    events = ["view", "cart", "buy"]

    base_time = datetime(2024, 1, 1, 8, 0, 0)

    data = []
    for uid in user_ids:
        for _ in range(7):
            timestamp = base_time + timedelta(hours=random.randint(0, 72))
            event = random.choice(events)
            data.append((uid, timestamp, event))

    df = pl.DataFrame(data, schema=["user_id", "timestamp", "event"])
    df = df.sort(["user_id", "timestamp"])

    print("🔹元データ")
    print(df)

    # --- 特徴量作成 ---
    print("\n🔹First/Last Features")
    print(compute_first_last_features(
        df, user_col="user_id", time_col="timestamp", event_col="event")
        )

    print("\n🔹Time Interval Stats")
    print(compute_time_diff_stats(df, user_col="user_id", time_col="timestamp"))

    print("\n🔹Lag from Last 'cart' Event")
    print(compute_event_lag_features(
        df, user_col="user_id", time_col="timestamp", event_col="event", target_event="cart")
        )

    print("\n🔹Max Consecutive Login Days")
    print(compute_consecutive_days_features(df, user_col="user_id", time_col="timestamp"))

    print("\n🔹Weighted Recent History Score (example)")
    history = [0, 1, 1, 0, 1]  # 仮のユーザー履歴
    from my_library.utils.array_utils import apply_weighted_decay
    score = apply_weighted_decay(history, base=0.1)
    print(f"Input: {history}, Weighted Score: {score:.5f}")
