
from typing import List, Literal, Optional

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
