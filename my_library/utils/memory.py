import numpy as np
import polars as pl

from my_library.utils.df_utils import df_io_polars


@df_io_polars(return_type="polars")
def reduce_df_memory_usage(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Reduce memory usage of a Polars DataFrame by downcasting numeric columns.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        verbose (bool): If True, print memory usage before and after.

    Returns:
        pl.DataFrame: Memory-optimized DataFrame.
    """
    if verbose:
        print(f"[Memory] Before optimization: {round(df.estimated_size('mb'), 4)} MB")

    int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
    float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in int_types + float_types:
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min is None or c_max is None:
                continue

            if col_type in int_types:
                df = df.with_columns(df[col].cast(_smallest_int_dtype(c_min, c_max)))
            elif col_type in float_types:
                if _can_cast_to_float32(c_min, c_max):
                    df = df.with_columns(df[col].cast(pl.Float32))

    if verbose:
        print(f"[Memory] After optimization: {round(df.estimated_size('mb'), 4)} MB")

    return df


def _smallest_int_dtype(c_min, c_max):
    """Return the smallest suitable integer dtype for the given min/max values."""
    if c_min >= 0:
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if c_max <= np.iinfo(dtype).max:
                return pl.datatypes.dtype(dtype)
    else:
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                return pl.datatypes.dtype(dtype)
    return pl.Int64  # fallback


def _can_cast_to_float32(c_min, c_max):
    """Check if a float column can safely be cast to Float32."""
    return (
        c_min > np.finfo(np.float32).min
        and c_max < np.finfo(np.float32).max
    )
