
from typing import List, Optional

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
