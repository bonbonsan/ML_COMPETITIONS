from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import polars as pl
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy import sparse

from my_library.utils.array_utils import compute_embedding_similarity
from my_library.utils.df_utils import df_io_polars


@df_io_polars(return_type="polars")
def generate_user_item_dot_features(
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    target_items: List[Union[str, int]],
    embedding_dim: int,
    model_type: Literal["ALS", "BPR"] = "ALS",
    output_path: str = None,
    prefix: str = "",
    random_state: int = 42,
    use_gpu: bool = False,
    als_params: Dict = None,
    bpr_params: Dict = None,
    similarity_type: Literal["dot", "cosine"] = "dot"
) -> pl.DataFrame:
    """
    Train matrix factorization model and generate dot-product or cosine features between users
    and target items using Polars.

    Args:
        df: Polars DataFrame with user-item interactions.
        user_col: Name of the user column.
        item_col: Name of the item column.
        target_items: List of item IDs to compute similarity against.
        embedding_dim: Dimension of latent vectors.
        model_type: 'ALS' or 'BPR'.
        output_path: Optional path to save the resulting features as Parquet.
        prefix: Prefix for output column names.
        random_state: Random seed.
        use_gpu: Use GPU (only applicable to BPR).
        als_params: Dict of ALS hyperparameters.
        bpr_params: Dict of BPR hyperparameters.
        similarity_type: 'dot' or 'cosine'.

    Returns:
        Polars DataFrame with user ID and features for each target item.
    """
    # Extract columns and convert to pandas for factorization
    df_pd = df.select([user_col, item_col]).to_pandas()
    df_pd["user_idx"], user_index = pd.factorize(df_pd[user_col])
    df_pd["item_idx"], item_index = pd.factorize(df_pd[item_col])

    # Sparse matrix
    interaction_matrix = sparse.csr_matrix(
        (np.ones(len(df_pd)), (df_pd["user_idx"], df_pd["item_idx"]))
    )

    # Train model
    if model_type == "ALS":
        als_params = als_params or {}
        model = AlternatingLeastSquares(
            factors=embedding_dim,
            regularization=als_params.get("regularization", 0.1),
            iterations=als_params.get("iterations", 500),
            alpha=als_params.get("alpha", 100),
            random_state=random_state,
            calculate_training_loss=True,
        )
    elif model_type == "BPR":
        bpr_params = bpr_params or {}
        model = BayesianPersonalizedRanking(
            factors=embedding_dim,
            learning_rate=bpr_params.get("learning_rate", 0.02),
            regularization=bpr_params.get("regularization", 0.01),
            iterations=bpr_params.get("iterations", 500),
            use_gpu=use_gpu,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(interaction_matrix)

    # Fix: ensure same dimension between user and item factors
    if model_type == "BPR":
        actual_dim = min(model.user_factors.shape[1], model.item_factors.shape[1])
        user_factors = model.user_factors[:, :actual_dim]
        item_factors = model.item_factors[:, :actual_dim]
    else:
        user_factors = model.user_factors
        item_factors = model.item_factors

    user_emb_dict = {i: vec for i, vec in enumerate(user_factors)}
    item_emb_dict = {i: vec for i, vec in enumerate(item_factors)} 

    default_emb = np.zeros(user_factors.shape[1], dtype=np.float32)
    item_name_to_index = df_pd.drop_duplicates(subset=[item_col, "item_idx"])\
        .set_index(item_col)["item_idx"].to_dict()

    # Create Polars output DataFrame
    unique_users = df_pd.drop_duplicates("user_idx")[[user_col, "user_idx"]]
    result_df = pl.DataFrame(unique_users)

    for item in target_items:
        item_id = item_name_to_index.get(item, None)
        if item_id is None:
            # ターゲットが未知のときはゼロで埋める
            scores = np.zeros(len(result_df), dtype=np.float32)
        else:
            scores = compute_embedding_similarity(
                user_ids=result_df["user_idx"].to_list(),
                item_ids=[item_id] * len(result_df),
                user_emb_dict=user_emb_dict,
                item_emb_dict=item_emb_dict,
                default_emb=default_emb,
                similarity_type=similarity_type
            )
        result_df = result_df.with_columns(
            pl.Series(
                name=f"{prefix}_{model_type}_{similarity_type}_{item}",
                values=scores
            )
        )

    # Drop user_idx and return
    result_df = result_df.drop("user_idx")

    if output_path:
        result_df.write_parquet(output_path)

    return result_df


if __name__ == "__main__":
    
    dummy_data = pl.DataFrame({
        "user_id": ["U0"] * 3 + ["U1"] * 2 + ["U2"] * 1 + ["U3"] * 3 + ["U4"] * 2,
        "item_id": ["A", "B", "C", "A", "C", "C", "A", "B", "C", "A", "B"]
    })
    print(dummy_data)

    target_items = ["A", "B", "C"]

    result = generate_user_item_dot_features(
        df=dummy_data,
        user_col="user_id",
        item_col="item_id",
        target_items=target_items,
        embedding_dim=8,
        model_type="ALS",  # ALS or BPR
        use_gpu=False,
        similarity_type="dot",
        prefix="sim"
    )

    print("===== ALS =====")
    print(result)

    result = generate_user_item_dot_features(
        df=dummy_data,
        user_col="user_id",
        item_col="item_id",
        target_items=target_items,
        embedding_dim=8,
        model_type="BPR",  # ALS or BPR
        use_gpu=False,
        similarity_type="dot",
        prefix="sim"
    )

    print("===== BPR =====")
    print(result)
