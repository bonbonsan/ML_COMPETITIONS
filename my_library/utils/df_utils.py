from functools import wraps
from typing import Any, Callable, List, Literal, TypeVar, Union, cast

import pandas as pd
import polars as pl

ReturnTypeLiteral = Literal["pandas", "polars"]

# Generic type for the decorator
# This allows us to specify that the decorator can be used on any function
# that takes any number of arguments and returns a DataFrame
F = TypeVar("F", bound=Callable[..., pl.DataFrame])

DataFrameType = Union[pd.DataFrame, pl.DataFrame]


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
