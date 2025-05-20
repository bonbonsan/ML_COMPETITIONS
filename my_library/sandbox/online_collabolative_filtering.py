import polars as pl

from my_library.feature_engineerings.itemcf import (
    aggregate_user_item_scores,
    compute_chunked_cf_scores,
)
from my_library.utils.df_utils import rename_columns_with_tag


# --------------------
# Sample Data Creation
# --------------------
def assign_age_group(age: int) -> str:
    return f"{(age // 10) * 10}s"

click_log = pl.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 4],
    "page_id": ["P1", "P2", "P1", "P3", "P2", "P4", "P1"],
    "ts": [10, 20, 15, 25, 12, 22, 20],
    "age": [25, 25, 32, 32, 45, 45, 60]
})

contract_log = pl.DataFrame({
    "user_id": [1, 2, 2, 3, 3, 4],
    "product_id": ["F1", "F2", "F1", "F2", "F3", "F1"],
    "ts": [23, 28, 40, 33, 36, 38],
    "age": [25, 32, 32, 45, 45, 60]
})

click_log = click_log.with_columns([
    pl.col("age").map_elements(assign_age_group).alias("age_group")
])
contract_log = contract_log.with_columns([
    pl.col("age").map_elements(assign_age_group).alias("age_group")
])

# --------------------
# User-based ItemCF Features
# --------------------
cf_scores = compute_chunked_cf_scores(
    df_left=click_log,
    df_right=contract_log,
    user_col="user_id",
    left_item_col="page_id",
    right_item_col="product_id",
    timestamp_col="ts",
    chunk_size=10,
    direction_weight=0.8
)

click_df_for_join = click_log.select(["user_id", "page_id", "ts"])
joined_df = click_df_for_join.join(cf_scores, left_on="page_id", right_on="item_left", how="inner")

joined_df = joined_df.with_columns([
    (1.0 / pl.col("ts").rank(method="dense", descending=True).over("user_id")).alias("pos_weight")
])
joined_df = joined_df.with_columns([
    (pl.col("score") * pl.col("pos_weight")).alias("weighted_score")
])

# --------------------
# Age-group-based ItemCF Features
# --------------------
age_cf_scores = compute_chunked_cf_scores(
    df_left=click_log.rename({"age_group": "group_id"}),
    df_right=contract_log.rename({"age_group": "group_id"}),
    user_col="group_id",
    left_item_col="page_id",
    right_item_col="product_id",
    timestamp_col="ts",
    chunk_size=10,
    direction_weight=0.8
)

click_df_for_join_age = click_log.select(["user_id", "age_group", "page_id", "ts"])
age_joined = click_df_for_join_age.join(
    age_cf_scores, left_on="page_id", right_on="item_left", how="inner"
    )

age_joined = age_joined.with_columns([
    (1.0 / pl.col("ts").rank(method="dense", descending=True).over("age_group")).alias("pos_weight")
])
age_joined = age_joined.with_columns([
    (pl.col("score") * pl.col("pos_weight")).alias("weighted_score")
])

# --------------------
# Aggregate Features
# --------------------
target_products = ["F1", "F2", "F3"]
agg_func_map = {
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "std": lambda col: pl.col(col).std(),
    "max": lambda col: pl.col(col).max(),
    "min": lambda col: pl.col(col).min()
}

user_features = aggregate_user_item_scores(
    df=joined_df,
    user_col="user_id",
    item_col="item_right",
    agg_func_map=agg_func_map,
    target_items=target_products,
    score_col="score",
    weight_col="weighted_score"
)
user_features = rename_columns_with_tag(
    user_features, [c for c in user_features.columns if c != "user_id"], "cf"
    )

age_features = aggregate_user_item_scores(
    df=age_joined,
    user_col="user_id",
    item_col="item_right",
    agg_func_map=agg_func_map,
    target_items=target_products,
    score_col="score",
    weight_col="weighted_score"
)
age_features = rename_columns_with_tag(
    age_features, [c for c in age_features.columns if c != "user_id"], "cf_age"
    )

# --------------------
# Merge All Features
# --------------------
final_df = click_log.select(["user_id"]).unique()
final_df = final_df.join(user_features, on="user_id", how="left")
final_df = final_df.join(age_features, on="user_id", how="left")
final_df = final_df.fill_null(-1)

print(final_df)
