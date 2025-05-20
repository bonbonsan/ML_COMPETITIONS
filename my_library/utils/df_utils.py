import datetime
from functools import wraps
from typing import Any, Callable, List, Literal, Optional, TypeVar, Union, cast

import pandas as pd
import polars as pl

DataFrameType = Union[pd.DataFrame, pl.DataFrame]
ReturnTypeLiteral = Literal["pandas", "polars"]

# Generic type for the decorator
# This allows us to specify that the decorator can be used on any function
# that takes any number of arguments and returns a DataFrame
F = TypeVar("F", bound=Callable[..., pl.DataFrame])


def df_io_polars(
        return_type: ReturnTypeLiteral = "polars"
        ) -> Callable[[F], Callable[..., DataFrameType]]:
    """
    Decorator for feature engineering functions that operate using Polars internally,
    while allowing flexible input/output in either pandas or polars DataFrame format.

    Args:
        return_type (str, optional): Desired output format.
            - "pandas": Return pandas.DataFrame (default)
            - "polars": Return polars.DataFrame

    Returns:
        A wrapped function that:
        - Accepts pandas or polars DataFrame as first argument.
        - Converts input to polars internally.
        - Executes the original function (expects polars.DataFrame).
        - Converts output to the specified return_type.
    
    Raises:
        TypeError: If input is neither pandas nor polars DataFrame.
        ValueError: If return_type is invalid.
    """

    def decorator(func: F) -> Callable[..., DataFrameType]:
        @wraps(func)
        def wrapper(df: DataFrameType, *args, **kwargs) -> DataFrameType:
            if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
                raise TypeError(
                    f"[df_io_polars] Expected pandas or \
                      polars DataFrame as first argument, got {type(df)}"
                )

            # Convert to polars if needed
            df_polars = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

            # Execute core logic
            result_polars = func(df_polars, *args, **kwargs)

            # Output conversion
            if return_type == "pandas":
                return to_pandas(result_polars)
            elif return_type == "polars":
                return result_polars
            else:
                raise ValueError(
                    f"[df_io_polars] return_type must be 'pandas' or 'polars', got '{return_type}'"
                    )

        return cast(Callable[..., DataFrameType], wrapper)

    return decorator


def to_polars(df: DataFrameType) -> pl.DataFrame:
    """
    Converts a pandas or polars DataFrame to polars.DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    elif isinstance(df, pl.DataFrame):
        return df
    else:
        raise TypeError(f"Expected pd.DataFrame or pl.DataFrame, got {type(df)}")


def to_pandas(df: DataFrameType) -> pd.DataFrame:
    """
    Converts a pandas or polars DataFrame to pandas.DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise TypeError(f"Expected pd.DataFrame or pl.DataFrame, got {type(df)}")


@df_io_polars(return_type="polars")
def filter_by_date_range(
    df: pl.DataFrame,
    date_col: str,
    start_date: datetime,
    end_date: datetime,
    start_inclusive: bool = True,
    end_inclusive: bool = True
) -> pl.DataFrame:
    """
    Filters a DataFrame using a flexible date range on the specified column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        date_col (str): Column to filter.
        start_date (datetime): Start of range.
        end_date (datetime): End of range.
        start_inclusive (bool): If True, start_date is inclusive (>=). Else, (>).
        end_inclusive (bool): If True, end_date is inclusive (<=). Else, (<).

    Returns:
        pl.DataFrame: Filtered DataFrame.
    """
    if df.schema.get(date_col) not in [pl.Datetime, pl.Date]:
        raise ValueError(f"[filter_by_date_range] '{date_col}' must be of type Date or Datetime.")

    # Build dynamic conditions
    start_cond = (
        pl.col(date_col) >= pl.lit(start_date)
        if start_inclusive else
        pl.col(date_col) > pl.lit(start_date)
    )

    end_cond = (
        pl.col(date_col) <= pl.lit(end_date)
        if end_inclusive else
        pl.col(date_col) < pl.lit(end_date)
    )

    return df.filter(start_cond & end_cond)


@df_io_polars(return_type="polars")
def filter_all_conditions(
    df: pl.DataFrame,
    *conditions: pl.Expr
) -> pl.DataFrame:
    """
    Filters a Polars DataFrame by applying all given conditions (logical AND).

    Args:
        df (pl.DataFrame): The input DataFrame.
        *conditions (pl.Expr): Any number of Polars expression conditions (e.g., pl.col("x") > 0).

    Returns:
        pl.DataFrame: A filtered DataFrame where all conditions are True.

    Raises:
        ValueError: If no conditions are provided.
    """
    if not conditions:
        raise ValueError("[filter_all_conditions] At least one condition must be provided.")

    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition &= cond

    return df.filter(combined_condition)


@df_io_polars(return_type="polars")
def rename_columns_with_tag(
    df: pl.DataFrame,
    cols: List[str],
    tag: str
) -> pl.DataFrame:
    """
    Rename specified columns in a Polars DataFrame by appending a tag.

    Args:
        df (pl.DataFrame): Input DataFrame.
        cols (List[str]): List of column names to rename.
        tag (str): Suffix tag to append.

    Returns:
        pl.DataFrame: DataFrame with renamed columns.
    """
    rename_map = {col: f"{col}_{tag}" for col in cols if col in df.columns}
    return df.rename(rename_map)


def memory_efficient_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: Union[str, List[str]],
    how: Literal["left", "inner", "outer", "semi", "anti"] = "left",
    right_cols: Optional[List[str]] = None,
    cast_categorical: bool = True
) -> pl.DataFrame:
    """
    Efficiently joins two Polars DataFrames with memory-saving strategies.

    This function is optimized for joining large datasets in Polars. It supports:
    - selecting only necessary columns from the right table
    - optionally casting join keys to categorical types for performance
    - avoiding unnecessary memory consumption by handling join order and column filtering

    Args:
        left (pl.DataFrame): The left-hand side DataFrame (typically larger).
        right (pl.DataFrame): The right-hand side DataFrame (typically smaller).
        on (Union[str, List[str]]): Column(s) to join on.
        how (str): Type of join. One of: 'left', 'inner', 'outer', 'semi', 'anti'.
                   Default is 'left'.
        right_cols (Optional[List[str]]): If specified, only these columns from `right` will be kept
                                          (in addition to the join keys). Default is None (all).
        cast_categorical (bool): Whether to cast join keys in both DataFrames to categorical.
                                 Useful when joining on high-cardinality strings. Default is True.

    Returns:
        pl.DataFrame: The joined DataFrame.
    
    Raises:
        ValueError: If provided join columns are not found in either DataFrame.
    """
    # Ensure join columns are list
    if isinstance(on, str):
        on = [on]

    # Validate join columns
    missing_left = [col for col in on if col not in left.columns]
    missing_right = [col for col in on if col not in right.columns]
    if missing_left or missing_right:
        raise ValueError(f"Missing join keys - Left: {missing_left}, Right: {missing_right}")

    # Optionally cast join keys to categorical
    if cast_categorical:
        for col in on:
            if left.schema[col] == pl.Utf8:
                left = left.with_columns(pl.col(col).cast(pl.Categorical))
            if right.schema[col] == pl.Utf8:
                right = right.with_columns(pl.col(col).cast(pl.Categorical))

    # Reduce right table to necessary columns
    if right_cols is not None:
        right = right.select(list(set(on + right_cols)))

    # Perform the join (Polars is optimized to bring right side in memory)
    return left.join(right, on=on, how=how)


def get_common_columns(
        df1: DataFrameType, df2: DataFrameType
        ) -> List[str]:
    """
    Returns the list of column names that are common to both DataFrames.
    """
    cols1 = set(to_pandas(df1).columns)
    cols2 = set(to_pandas(df2).columns)
    return sorted(list(cols1 & cols2))


def get_unique_values(df: DataFrameType, column: str) -> List[Any]:
    """
    Returns a list of unique values from a specified column of a pandas or polars DataFrame.

    Args:
        df (pd.DataFrame | pl.DataFrame): Input DataFrame (pandas or polars).
        column (str): Column name to extract unique values from.

    Returns:
        List[Any]: List of unique values in the column.

    Raises:
        TypeError: If input is not a DataFrame.
        ValueError: If the specified column does not exist.
    """
    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        return df[column].dropna().unique().tolist()

    elif isinstance(df, pl.DataFrame):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        return df.select(pl.col(column).unique()).to_series().to_list()

    else:
        raise TypeError(f"Expected pandas or polars DataFrame, got {type(df)}")
