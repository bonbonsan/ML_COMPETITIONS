from typing import Callable, Dict, List

import polars as pl

from my_library.utils.df_utils import df_io_polars


def compute_itemcf_score(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    user_col: str,
    left_item_col: str,
    right_item_col: str,
    timestamp_col: str = "ts",
    direction_weight: float = 0.8
) -> pl.DataFrame:
    """
    Compute item-item similarity scores based on user co-occurrence between two item columns.

    Parameters
    ----------
    df_left : pl.DataFrame
        Left-side interaction log (e.g., click).
    df_right : pl.DataFrame
        Right-side interaction log (e.g., purchase).
    user_col : str
        Column name for user ID.
    left_item_col : str
        Item column in df_left.
    right_item_col : str
        Item column in df_right.
    timestamp_col : str, default "ts"
        Timestamp column name.
    direction_weight : float, default 0.8
        Weight applied when left timestamp is later than right timestamp.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns [item_left, item_right, score].
    """
    df_l = df_left.rename({left_item_col: "item_left", timestamp_col: "ts_left"})
    df_r = df_right.rename({right_item_col: "item_right", timestamp_col: "ts_right"})

    pair_df = df_l.join(df_r, on=user_col, how="inner")

    if left_item_col == right_item_col:
        pair_df = pair_df.filter(pl.col("item_left") != pl.col("item_right"))

    pair_df = pair_df.unique(
        subset=[user_col, "item_left", "item_right"], keep="last", maintain_order=True
        )
    pair_df = pair_df.with_columns(pl.lit(1.0).alias("score"))

    past_df = pair_df.filter(pl.col("ts_left") <= pl.col("ts_right"))
    future_df = pair_df.filter(pl.col("ts_left") > pl.col("ts_right"))
    future_df = future_df.with_columns((pl.col("score") * direction_weight).alias("score"))

    pair_df = pl.concat([past_df, future_df])

    score_df = pair_df.group_by(["item_left", "item_right"]).agg([
        pl.col("score").sum().alias("score")
    ])

    norm_df = pair_df.group_by("item_left").agg([
        pl.len().alias("count")
    ])

    score_df = score_df.join(norm_df, on="item_left", how="left")
    score_df = score_df.with_columns(
        (pl.col("score") / (pl.col("count") + 10)).alias("score")
    )

    return score_df.select(["item_left", "item_right", "score"])


def compute_chunked_cf_scores(
    df_left: pl.DataFrame,
    df_right: pl.DataFrame,
    user_col: str,
    left_item_col: str,
    right_item_col: str,
    timestamp_col: str = "ts",
    chunk_size: int = 10000,
    direction_weight: float = 0.8
) -> pl.DataFrame:
    """
    Compute itemCF scores in user-based chunks to reduce memory usage.

    Parameters
    ----------
    df_left : pl.DataFrame
        Left-side interaction log.
    df_right : pl.DataFrame
        Right-side interaction log.
    user_col : str
        Column name for user ID.
    left_item_col : str
        Item column in df_left.
    right_item_col : str
        Item column in df_right.
    timestamp_col : str, default "ts"
        Timestamp column name.
    chunk_size : int, default 10000
        Number of users per chunk.
    direction_weight : float, default 0.8
        Directional downweight for future events.

    Returns
    -------
    pl.DataFrame
        Aggregated itemCF score matrix.
    """
    df_left = df_left.with_columns((pl.col(user_col) // chunk_size).alias("chunk_group"))
    chunk_scores = []

    for chunk_id in df_left["chunk_group"].unique():
        sub_left = df_left.filter(pl.col("chunk_group") == chunk_id)
        sub_right = df_right.filter(pl.col(user_col).is_in(sub_left[user_col])).unique(
            subset=[user_col, right_item_col], keep="last", maintain_order=True
        )
        score_df = compute_itemcf_score(
            df_left=sub_left,
            df_right=sub_right,
            user_col=user_col,
            left_item_col=left_item_col,
            right_item_col=right_item_col,
            timestamp_col=timestamp_col,
            direction_weight=direction_weight
        )
        chunk_scores.append(score_df)

    return pl.concat(chunk_scores).group_by(["item_left", "item_right"]).agg(
        [pl.col("score").sum().alias("score")]
    )


@df_io_polars(return_type="polars")
def aggregate_user_item_scores(
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    agg_func_map: Dict[str, Callable[[str], pl.Expr]],
    target_items: List[str],
    score_col: str = "score",
    weight_col: str = None
) -> pl.DataFrame:
    """
    Aggregate itemCF scores into user-item-level features using flexible aggregation functions.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with user-item-score interactions.
    user_col : str
        User ID column name.
    item_col : str
        Target item column name (e.g., item_right).
    agg_func_map : dict[str, Callable]
        Dictionary mapping suffix names to functions that return Polars expressions.
    target_items : list of str
        List of item IDs to extract features for.
    score_col : str, default "score"
        Column name of itemCF score.
    weight_col : str, optional
        Column name for weighted score, if any.

    Returns
    -------
    pl.DataFrame
        Wide-format user-level feature DataFrame (one row per user).
    """
    df = df.filter(pl.col(item_col).is_in(target_items))

    agg_exprs = []
    for suffix, func in agg_func_map.items():
        agg_exprs.append(func(score_col).alias(f"{score_col}_{suffix}"))
        if weight_col:
            agg_exprs.append(func(weight_col).alias(f"{weight_col}_{suffix}"))

    agg_df = df.group_by([user_col, item_col]).agg(agg_exprs)

    result = df.select(user_col).unique()
    for item in target_items:
        tmp = agg_df.filter(pl.col(item_col) == item).drop(item_col)
        tmp = tmp.rename({col: f"{col}_{item}" for col in tmp.columns if col != user_col})
        result = result.join(tmp, on=user_col, how="left")

    return result


if __name__ == "__main__":
    from my_library.utils.df_utils import rename_columns_with_tag

    click_df = pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "item_id": ["A", "B", "A", "D", "B", "C", "A"],
        "ts": [10, 20, 15, 25, 12, 22, 20]
    })

    purchase_df = pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "item_id": ["C", "D", "C", "A", "D", "B", "A"],
        "ts": [23, 35, 28, 40, 27, 33, 30]
    })

    print("=== Click Log ===")
    print(click_df)

    print("=== Purchase Log ===")
    print(purchase_df)

    agg_func_map = {
        "sum": lambda col: pl.col(col).sum(),
        "mean": lambda col: pl.col(col).mean(),
        "std": lambda col: pl.col(col).std(),
        "max": lambda col: pl.col(col).max(),
        "min": lambda col: pl.col(col).min(),
        "last": lambda col: pl.col(col).last()
    }

    target_items = ["A", "B", "C", "D"]
    all_features = []

    pattern_defs = [
        (click_df, click_df, "click2click"),     # 同時にクリックされる商品
        (click_df, purchase_df, "click2buy"),   # クリック後によく買われる商品
        (purchase_df, purchase_df, "buy2buy")   # 一緒に買われる商品
    ]

    for df_left, df_right, tag in pattern_defs:
        print(f"\n=== Pattern: {tag} ===")
        score_df = compute_chunked_cf_scores(
            df_left=df_left,
            df_right=df_right,
            user_col="user_id",
            left_item_col="item_id",
            right_item_col="item_id",
            timestamp_col="ts",
            chunk_size=5,
            direction_weight=0.5
        )

        df_for_join = df_left.rename({"ts": "left_ts"})

        joined_df = df_for_join.join(score_df, left_on="item_id", right_on="item_left", how="inner")

        joined_df = joined_df.with_columns(
            (1.0 / pl.col("left_ts").rank(method="dense", descending=True)\
             .over("user_id")).alias("pos_weight")
        )

        joined_df = joined_df.with_columns(
            (pl.col("score") * pl.col("pos_weight")).alias("weighted_score")
        )

        features = aggregate_user_item_scores(
            df=joined_df,
            user_col="user_id",
            item_col="item_right",
            agg_func_map=agg_func_map,
            target_items=target_items,
            score_col="score",
            weight_col="weighted_score"
        )
        features = features.fill_null(-1)

        rename_cols = [col for col in features.columns if col != "user_id"]
        features = rename_columns_with_tag(features, rename_cols, tag)
        all_features.append(features)

    base_users = click_df.select("user_id").unique()
    for f in all_features:
        base_users = base_users.join(f, on="user_id", how="left")

    print("\n=== Merged All Features ===")
    print(base_users)
    # print(base_users.columns)
    print(base_users.filter(pl.col("user_id") == 4))
