from itertools import product
from typing import Callable, Dict, List, Union

import pandas as pd
import polars as pl

from my_library.utils.df_utils import df_io_polars, get_unique_values


@df_io_polars(return_type="polars")
def aggregate_by_category_combinations(
    df: Union[pd.DataFrame, pl.DataFrame],
    group_cols: List[str],
    agg_func_map: Dict[str, Callable[[pl.Series], float]],
    default_value: float = -10000.0
) -> pl.DataFrame:
    """
    Aggregates statistics over all combinations of unique category values
    for the given group_cols using provided aggregation functions.

    Args:
        df (pd.DataFrame or pl.DataFrame): Input DataFrame (converted internally to polars).
        group_cols (List[str]): List of categorical column names to group by.
        agg_func_map (Dict[str, Callable]): Dict mapping function names to aggregation functions.
                                            Example: {"mean": pl.col("target").mean()}
        default_value (float): Value to fill for missing category combinations.
                               Defaults to -10000.0.

    Returns:
        pl.DataFrame: Aggregated statistics per category combination.
    """
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

    grouped_df = df.groupby(group_cols).agg(agg_exprs)

    # ---- 全組み合わせを網羅する補完処理 ----
    full_index_df = pl.DataFrame(
        {col: [row[i] for row in combinations] for i, col in enumerate(group_cols)}
        )

    joined_df = full_index_df.join(grouped_df, on=group_cols, how="left")

    # ---- Fill missing values with default value ----
    fill_nulls = joined_df.fill_null(default_value)

    return fill_nulls


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

    agg_funcs = {
        "mean": pl.col("target").mean(),
        "max": pl.col("target").max(),
        "min": pl.col("target").min(),
        "median": pl.col("target").median(),
        "std": pl.col("target").std()
    }

    result = aggregate_by_category_combinations(
        df,
        group_cols=["gender", "age_group", "married"],
        agg_func_map=agg_funcs,
        default_value=-9999
    )

    print(result)
