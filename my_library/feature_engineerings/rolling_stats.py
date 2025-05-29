from typing import List, Optional

import polars as pl

from my_library.utils.df_utils import df_io_polars


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
    Efficient computation of rolling stats (mean, std, min, max, sum, median, quantiles).

    Supported stats:
        - "mean", "std", "min", "max", "sum", "median"
        - "q1", "q3" (25%, 75%)
        - "q01", "q05", "q95", "q99"

    Args:
        df: Input dataframe.
        group_cols: Columns to group by.
        target_col: Column to compute on.
        window_sizes: Rolling window sizes (e.g., [7, 14, 30]).
        shift: Shift before rolling to avoid leakage (e.g., 1).
        stats: List of stats to compute.

    Returns:
        DataFrame with rolling feature columns.
    """
    if stats is None:
        stats = ["mean", "std"]

    def rolling_quantile_func(q: float):
        return lambda s, w: s.rolling_quantile(
            window_size=w,
            quantile=q,
            interpolation="nearest"
        )

    supported_stats = {
        "mean": lambda s, w: s.rolling_mean(w),
        "std": lambda s, w: s.rolling_std(w),
        "min": lambda s, w: s.rolling_min(w),
        "max": lambda s, w: s.rolling_max(w),
        "sum": lambda s, w: s.rolling_sum(w),
        "median": lambda s, w: s.rolling_median(w),
        "q1": rolling_quantile_func(0.25),
        "q3": rolling_quantile_func(0.75),
        "q01": rolling_quantile_func(0.01),
        "q05": rolling_quantile_func(0.05),
        "q95": rolling_quantile_func(0.95),
        "q99": rolling_quantile_func(0.99),
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


@df_io_polars(return_type="polars")
def compute_ewm_features(
    df: pl.DataFrame,
    group_cols: List[str],
    target_col: str,
    spans: List[int],
    output_prefix: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute exponentially weighted moving averages (EWMA) for specified spans.

    Args:
        df: Input DataFrame.
        group_cols: Grouping columns.
        target_col: Value column.
        spans: List of span values for EWMA (e.g., [7, 14, 30]).
        output_prefix: Optional prefix for output columns.

    Returns:
        DataFrame with EWMA columns added.
    """
    if output_prefix is None:
        output_prefix = f"{target_col}_ewm"

    exprs = [
        pl.col(target_col)
        .ewm_mean(span=span)
        .over(group_cols)
        .alias(f"{output_prefix}_{span}")
        for span in spans
    ]

    return df.with_columns(exprs)


@df_io_polars(return_type="polars")
def compute_bollinger_bands(
    df: pl.DataFrame,
    group_cols: List[str],
    target_col: str,
    window: int,
    num_std: float = 2.0
) -> pl.DataFrame:
    """
    Compute Bollinger Bands (upper, lower, width) from rolling mean and std.

    Args:
        df: Input dataframe.
        group_cols: Grouping columns.
        target_col: Value column.
        window: Rolling window size.
        num_std: Number of standard deviations (typically 2).

    Returns:
        DataFrame with Bollinger Band columns.
    """
    mean_col = f"{target_col}_bb_mean_{window}"
    std_col = f"{target_col}_bb_std_{window}"
    upper_col = f"{target_col}_bb_upper_{window}"
    lower_col = f"{target_col}_bb_lower_{window}"
    width_col = f"{target_col}_bb_width_{window}"

    df = df.with_columns([
        pl.col(target_col).shift(1).rolling_mean(window).over(group_cols).alias(mean_col),
        pl.col(target_col).shift(1).rolling_std(window).over(group_cols).alias(std_col),
    ])

    return df.with_columns([
        (pl.col(mean_col) + num_std * pl.col(std_col)).alias(upper_col),
        (pl.col(mean_col) - num_std * pl.col(std_col)).alias(lower_col),
        ((pl.col(upper_col) - pl.col(lower_col)) / pl.col(mean_col)).alias(width_col),
    ])
