from functools import reduce
from itertools import product
from typing import Callable, Dict, List

import pandas as pd
import polars as pl

from my_library.utils.df_utils import DataFrameType, df_io_polars, get_unique_values


@df_io_polars(return_type="polars")
def aggregate_multi_category_long(
    df: DataFrameType,
    group_cols: List[str],
    agg_func_map: Dict[str, Callable[[pl.Series], float]],
    default_value: float = -10000.0
) -> pl.DataFrame:
    """
    Aggregates statistics over all combinations of unique values in multiple categorical columns,
    returning the result in long format with multi-column grouping.

    This function is particularly useful when you want to compute statistics across **combinations
    of multiple categorical variables** (e.g., ["store", "gender", "weekday"]) and then join the
    resulting long-format aggregated data back to your original dataset on those multiple keys.

    Unlike wide-format pivot aggregations (where categorical values become column names), this
    function returns each combination of categorical values as a **row**, making it ideal for
    multi-column joins and complex grouping logic.

    Args:
        df (pd.DataFrame or pl.DataFrame): Input DataFrame (internally converted to polars).
        group_cols (List[str]): List of categorical columns to group by (e.g., ["店舗名", "性別"]).
        agg_func_map (Dict[str, Callable]): Mapping of aggregation function names to
                                            Polars expressions.
                                            Example: {"mean": pl.col("売上金額").mean()}
        default_value (float): Value to fill for missing combinations. Defaults to -10000.0.

    Returns:
        pl.DataFrame: Aggregated statistics in long format (one row per category combination).
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    # ---- Validation: Category type or not ----
    non_cat_cols = []
    for col in group_cols:
        dtype = df.schema.get(col)
        if dtype not in (pl.Utf8, pl.Categorical, pl.Boolean):
            non_cat_cols.append((col, str(dtype)))

    if non_cat_cols:
        msgs = [f"{col}: {dtype}" for col, dtype in non_cat_cols]
        raise ValueError("The following columns are not categorical:\n" + "\n".join(msgs))

    # ---- Generate unique value combinations ----
    unique_lists = [get_unique_values(df, col) for col in group_cols]
    combinations = list(product(*unique_lists))  # list of tuples

    # ---- Actual aggregation ----
    agg_exprs = []
    for func_name, func in agg_func_map.items():
        expr_name = "_".join(group_cols) + f"_{func_name}"
        agg_exprs.append(func.alias(expr_name))

    grouped_df = df.group_by(group_cols).agg(agg_exprs)

    # ---- Covering all combinations ----
    full_index_df = pl.DataFrame(
        {col: [row[i] for row in combinations] for i, col in enumerate(group_cols)}
        )

    joined_df = full_index_df.join(grouped_df, on=group_cols, how="left")

    # ---- Fill missing values with default value ----
    fill_nulls = joined_df.fill_null(default_value)

    return fill_nulls


@df_io_polars(return_type="polars")
def aggregate_by_category_pivot(
    df: DataFrameType,
    index_col: str,
    category_col: str,
    agg_func_map: Dict[str, pl.Expr],
    default_value: float = 0.0
) -> pl.DataFrame:
    """
    Performs wide-format aggregation using Polars. Supports both built-in aggregation functions
    (e.g. 'mean', 'sum') via native pivot, and unsupported ones (e.g. 'std', 'n_unique') via
    manual groupby-reshape fallback.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Input dataframe.
        index_col (str): Key column to aggregate by (e.g., user_id).
        category_col (str): Categorical column to pivot (e.g., product category).
        agg_func_map (Dict[str, pl.Expr]): Mapping from function name (e.g., "mean")
                                           to Polars expression (e.g., pl.col("target").mean()).
        default_value (float): Fill value for missing combinations.

    Returns:
        pl.DataFrame: Wide-format dataframe, one row per index_col,
                      with pivoted statistical features.
    """
    SUPPORTED = {"sum", "min", "max", "mean", "first"}
    result = pl.DataFrame({index_col: df[index_col].unique()})

    for func_name, expr in agg_func_map.items():
        root_names = expr.meta.root_names()
        if not root_names:
            print(
                f"[WARN] Expression for '{func_name}' does not reference a column. "
                f"Using fallback column name "". Expression: {expr}"
            )
            col_name = ""
        else:
            col_name = root_names[0]

        if func_name in SUPPORTED and col_name != "":
            # safe to use pivot only if column exists
            pivot = (
                df.pivot(
                    values=col_name,
                    index=index_col,
                    columns=category_col,
                    aggregate_function=func_name
                )
                .fill_null(default_value)
            )

            # pivot columns like '20s_target_mean'
            pivot = pivot.rename({
                col: f"{col}_{col_name}_{func_name}" if col != index_col else col
                for col in pivot.columns
            })

        else:
            # use manual reshaping for count, std, etc.
            pivot = custom_pivot_with_expr(
                df=df,
                index_col=index_col,
                category_col=category_col,
                expr=expr,
                func_name=func_name,
                default_value=default_value
            )

        # 安全に index_col 由来の余分な列（e.g. gender_right）を削除
        conflicting_index_like_cols = [
            col for col in result.columns if col.startswith(f"{index_col}_right")
            ]
        if conflicting_index_like_cols:
            print(f"[INFO] Removing previously joined index_col variants: \
                  {conflicting_index_like_cols}"
                  )
            result = result.drop(conflicting_index_like_cols)

        # 重複チェック
        conflict_cols = set(result.columns) & set(pivot.columns) - {index_col}
        if conflict_cols:
            raise ValueError(f"[ERROR] Column conflict detected before join: {conflict_cols}")

        # 正常なjoin
        result = result.join(pivot, on=index_col, how="left")
        result = result.drop(
            [col for col in result.columns if col.startswith(f"{index_col}_right")]
        )

    return result


def custom_pivot_with_expr(
    df: pl.DataFrame,
    index_col: str,
    category_col: str,
    expr: pl.Expr,
    func_name: str,
    default_value: float = 0.0
) -> pl.DataFrame:
    """
    Performs manual wide-format reshaping when aggregation function is not supported
    by Polars pivot().
    Works by grouping and reshaping per category value.

    Args:
        df (pl.DataFrame): Input Polars dataframe.
        index_col (str): Column to group by (e.g., user_id).
        category_col (str): Categorical column to pivot on (e.g., product category).
        expr (pl.Expr): Polars aggregation expression (e.g., pl.col("target").std()).
        func_name (str): Function name used for naming output columns (e.g., "std").
        default_value (float): Value to fill for missing combinations.

    Returns:
        pl.DataFrame: Manually reshaped wide-format dataframe.
    """
    col_name = expr.meta.root_names()[0] if expr.meta.root_names() else ""

    agg_df = (
        df.group_by([index_col, category_col])
        .agg(expr.alias("value"))
    )

    category_values = df[category_col].unique().to_list()
    reshaped_dfs = []
    for val in category_values:
        suffix = (
            f"{val}_{func_name}" if col_name == "" else f"{val}_{col_name}_{func_name}"
        )
        extracted = (
            agg_df.filter(pl.col(category_col) == val)
            .select([index_col, pl.col("value").alias(suffix)])
        )
        reshaped_dfs.append(extracted)

    result = reduce(lambda left, right: left.join(right, on=index_col, how="outer"), reshaped_dfs)
    return result.fill_null(default_value)


@df_io_polars(return_type="polars")
def compute_category_presence_flags(
    df: pl.DataFrame,
    index_col: str,
    category_col: str,
    value_col: str,
    target_categories: list[str],
    flag_suffix: str = "presence_flag"
) -> pl.DataFrame:
    """
    Create binary flags indicating whether a target category appeared (value > 0)
    for each index (e.g., session_id, user_id).

    Args:
        df: Input DataFrame.
        index_col: Column to group by (e.g., session_id).
        category_col: Column representing category membership.
        value_col: Numeric column indicating quantity to check presence (e.g., quantity).
        target_categories: List of category values to check presence for.
        flag_suffix: Suffix for the generated flag columns (default: "presence_flag").

    Returns:
        pl.DataFrame: With binary flag columns like "<category>_<flag_suffix>".
    """
    pivot = (
        df.filter(pl.col(category_col).is_in(target_categories))
        .pivot(values=value_col, index=index_col, columns=category_col, aggregate_function="sum")
        .fill_null(0)
    )

    return pivot.with_columns([
        pl.when(pl.col(col) > 0).then(1).otherwise(0).alias(f"{col}_{flag_suffix}")
        for col in target_categories
    ]).select(
        [index_col] + [f"{col}_{flag_suffix}" for col in target_categories]
    )


@df_io_polars(return_type="polars")
def flag_group_combination_existence(
    df: pl.DataFrame,
    group_keys: List[str],
    value_col: str,
    flag_col_name: str = None
) -> pl.DataFrame:
    """
    Create binary flag column indicating whether each combination in group_keys
    appears in df with value_col > 0.

    Args:
        df: Polars DataFrame
        group_keys: List of column names to group by (e.g., ["gender", "age_group"])
        value_col: Column to test for positive existence (e.g., "target")
        flag_col_name: Optional name of the output flag column. If None,
                       defaults to "{g1}_{g2}_flag".

    Returns:
        pl.DataFrame with columns: group_keys + flag
    """
    # Step 1: Get all combinations
    unique_vals = [df[col].unique().to_list() for col in group_keys]
    all_combinations = list(product(*unique_vals))

    # Step 2: Build flag rows
    rows = []
    for comb in all_combinations:
        condition = pl.col(value_col) > 0
        for col, val in zip(group_keys, comb, strict=False):
            condition &= (pl.col(col) == val)

        flag = int(df.filter(condition).height > 0)
        row = {col: val for col, val in zip(group_keys, comb, strict=False)}
        row[flag_col_name or f"{'_'.join(group_keys)}_flag"] = flag
        rows.append(row)

    return pl.DataFrame(rows)


@df_io_polars(return_type="polars")
def compute_category_ratio(
    df: pl.DataFrame,
    group_key: str,
    category_col: str,
    count_col: str = "count",
    output_suffix: str = "_count_ratio"
) -> pl.DataFrame:
    """
    For each group_key, compute ratio of each category's count to total count.

    Args:
        df: Polars DataFrame
        group_key: Column to group by (e.g., 顧客CD)
        category_col: Column with categorical values
        count_col: Name of count column (default: "count")
        output_suffix: Suffix for ratio columns

    Returns:
        Wide-format DataFrame with group_key and ratio columns per category
    """
    count_df = (
        df.group_by([group_key, category_col])
        .agg(pl.len().alias(count_col))
    )

    total_df = (
        count_df.group_by(group_key)
        .agg(pl.col(count_col).sum().alias("total"))
    )

    merged = count_df.join(total_df, on=group_key)
    merged = merged.with_columns([
        (pl.col(count_col) / pl.col("total")).alias("ratio")
    ])

    pivot = merged.pivot(
        values="ratio",
        index=group_key,
        columns=category_col
    ).rename({
        col: str(col) + output_suffix for col in merged[category_col].unique().to_list()
    })

    return pivot


if __name__ == "__main__":

    def generate_test_df() -> pd.DataFrame:
        """
        Generates a test DataFrame with 3 categorical columns (2 values each),
        and 3 rows per unique combination with target values [1, 2, 3].
        Total: 8 combinations × 3 = 24 rows.
        
        Returns:
            pd.DataFrame: DataFrame with columns: gender, age_group, married, target
        """
        genders = ["male", "female"]
        age_groups = ["20s", "30s"]
        married_status = [True, False]

        combinations = list(product(genders, age_groups, married_status))

        rows = []
        for gender, age, married in combinations:
            for i in range(3):  # i=0,1,2 → target=1,2,3
                rows.append({
                    "gender": gender,
                    "age_group": age,
                    "married": married,
                    "target": i + 1
                })

        return pd.DataFrame(rows)

    df = generate_test_df()
    print(df)

    agg_funcs = {
        "mean": pl.col("target").mean(),
        "max": pl.col("target").max(),
        "min": pl.col("target").min(),
        "median": pl.col("target").median(),
        "std": pl.col("target").std(),
        "count": pl.len(),
        "n_unique": pl.col("target").n_unique(),
        "exists": (pl.len() > 0).cast(pl.Int8),
        "skew": pl.col("target").skew(),
        "kurt": pl.col("target").kurtosis(),
        "q1": pl.col("target").quantile(0.25, "nearest"),
        "q3": pl.col("target").quantile(0.75, "nearest")
    }

    print("=== Test for aggregate_multi_category_long ===")
    result = aggregate_multi_category_long(
        df,
        group_cols=["gender", "age_group", "married"],
        agg_func_map=agg_funcs,
        default_value=-9999
    )

    print(result)

    print("=== Test for aggregate_by_category_pivot ===")
    pivot_result = aggregate_by_category_pivot(
        df=df,
        index_col="gender",
        category_col="age_group",
        agg_func_map=agg_funcs,
        default_value=0.0
    )

    print(pivot_result)

    print("=== Test for flag_group_combination_existence ===")
    flag_result = flag_group_combination_existence(
        df=df,
        group_keys=["gender", "age_group"],
        value_col="target",
        flag_col_name=None
    )

    print(flag_result)

    print("=== Test for compute_category_ratio ===")
    ratio_result = compute_category_ratio(
        df=df,
        group_key="gender",
        category_col="age_group"
    )

    print(ratio_result)
