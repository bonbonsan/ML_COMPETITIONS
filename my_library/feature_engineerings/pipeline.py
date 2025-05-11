from typing import Callable, List

import pandas as pd


def apply_pipeline(
        df: pd.DataFrame, funcs: List[Callable[[pd.DataFrame], pd.DataFrame]]
        ) -> pd.DataFrame:
    for func in funcs:
        df = func(df)
    return df

# from my_library.feature_engineerings.datetime import extract_hour
# from my_library.feature_engineerings.categorical import count_encoding
# from my_library.feature_engineerings.pipeline import apply_feature_functions

# df = apply_feature_functions(df, [
#     lambda df: extract_hour(df, "purchase_time"),
#     lambda df: count_encoding(df, "store_id")
# ])
