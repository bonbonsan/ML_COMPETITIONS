from typing import List, Tuple

import networkx as nx
import numpy as np
import polars as pl

from my_library.feature_engineerings.encoders import AutoLabelEncoder
from my_library.feature_engineerings.prone import ProNE
from my_library.utils.array_utils import compute_embedding_similarity
from my_library.utils.df_utils import df_io_polars


def encode_user_item_nodes(
    df: pl.DataFrame,
    user_col: str,
    item_col: str
) -> Tuple[pl.DataFrame, AutoLabelEncoder, AutoLabelEncoder]:
    """
    Encode user and item columns into integer node IDs using AutoLabelEncoder.

    This is used for preparing a bipartite graph representation where
    user and item nodes must share the same ID space.

    Parameters
    ----------
    df : pl.DataFrame
        Input session dataframe with user and item columns.
    user_col : str
        Column name representing user ID.
    item_col : str
        Column name representing item ID.

    Returns
    -------
    Tuple containing:
    - df_encoded: Polars DataFrame with additional columns 'user_lbl' and 'item_lbl'.
    - user_encoder: Fitted AutoLabelEncoder for users.
    - item_encoder: Fitted AutoLabelEncoder for items.
    """
    user_encoder = AutoLabelEncoder()
    item_encoder = AutoLabelEncoder()

    df_pandas = df.to_pandas()
    user_encoder.fit(df_pandas[user_col])
    item_encoder.fit(df_pandas[item_col])

    item_ids = item_encoder.transform(df_pandas[item_col])
    user_ids = user_encoder.transform(df_pandas[user_col]) + max(item_ids) + 1

    df_encoded = pl.DataFrame({
        user_col: df_pandas[user_col],
        item_col: df_pandas[item_col],
        "user_lbl": user_ids,
        "item_lbl": item_ids
    })
    return df_encoded, user_encoder, item_encoder


def build_user_item_graph(
        df: pl.DataFrame,
        user_label_col: str = "user_lbl",
        item_label_col: str = "item_lbl"
        ) -> nx.Graph:
    """
    Construct an undirected bipartite graph of users and items using specified encoded columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with encoded user and item columns.
    user_label_col : str, optional
        Column name for encoded user IDs. Default is 'user_lbl'.
    item_label_col : str, optional
        Column name for encoded item IDs. Default is 'item_lbl'.

    Returns
    -------
    nx.Graph
        Bipartite undirected graph where edges represent user-item interactions.
    """
    G = nx.Graph()
    edge_list = df.select([user_label_col, item_label_col]).to_numpy().tolist()
    G.add_edges_from(edge_list)
    return G


@df_io_polars(return_type="polars")
def generate_u2i_prone_features(
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    target_items: List[str],
    emb_size: int = 128,
    step: int = 10,
    theta: float = 0.5,
    mu: float = 0.2,
    n_iter: int = 5,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Generate user-to-item similarity features using ProNE-based graph embeddings.

    This function constructs a bipartite graph between users and items, computes embeddings
    using the ProNE algorithm, and outputs cosine similarities between each user and
    specified target items as features.

    Parameters
    ----------
    df : pl.DataFrame
        Input session log with user and item columns.
    user_col : str
        Column name for user identifiers.
    item_col : str
        Column name for item identifiers.
    target_items : list of str
        Item IDs to compute cosine similarity features against.
    emb_size : int, optional
        Embedding dimensionality. Default is 128.
    step : int, optional
        Spectral propagation step count. Default is 10.
    theta : float, optional
        Gaussian kernel theta parameter. Default is 0.5.
    mu : float, optional
        Spectral propagation shift parameter. Default is 0.2.
    n_iter : int, optional
        Number of SVD iterations. Default is 5.
    random_state : int, optional
        Random seed. Default is 42.

    Returns
    -------
    pl.DataFrame
        A dataframe with one row per user and one column per
        target item containing cosine similarity scores.
    """
    df_encoded, user_encoder, item_encoder = encode_user_item_nodes(df, user_col, item_col)
    G = build_user_item_graph(df_encoded, user_label_col="user_lbl", item_label_col="item_lbl")

    model = ProNE(G, emb_size=emb_size, step=step, theta=theta,
                  mu=mu, n_iter=n_iter, random_state=random_state)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    emb_df = model.transform()

    emb_pl = pl.from_pandas(emb_df.set_index("nodes").add_prefix("ProNE_Emb_").reset_index())
    item_max = df_encoded['item_lbl'].max()

    # ユーザー側埋め込み
    u2emb = emb_pl.filter(pl.col("nodes") > item_max).with_columns([
        (pl.col("nodes") - item_max - 1).alias("user_idx")
    ])
    u2emb = u2emb.with_columns([
        pl.Series(name=user_col, values=user_encoder\
                  .inverse_transform(u2emb['user_idx'].to_numpy().tolist()))
    ]).drop(["nodes", "user_idx"])

    # アイテム側埋め込み
    i2emb_raw = emb_pl.filter(pl.col("nodes") <= item_max)
    item_ids = i2emb_raw["nodes"].to_numpy().tolist()
    item_names = item_encoder.inverse_transform(item_ids)
    i2emb = i2emb_raw.with_columns([
        pl.Series(name=item_col, values=item_names)
    ]).drop(["nodes"])

    # 辞書に変換
    u2emb_dict = dict(zip(
        u2emb[user_col],
        u2emb.drop([user_col]).to_numpy().astype(np.float32), strict=False
    ))
    i2emb_dict = dict(zip(
        i2emb[item_col],
        i2emb.drop([item_col]).to_numpy().astype(np.float32), strict=False
    ))

    default_emb = np.zeros(emb_size, dtype=np.float32)
    user_ids = u2emb[user_col].to_list()

    output = pl.DataFrame({user_col: user_ids})

    for item in target_items:
        item_ids = [item] * len(user_ids)
        scores = compute_embedding_similarity(
            user_ids=user_ids,
            item_ids=item_ids,
            user_emb_dict=u2emb_dict,
            item_emb_dict=i2emb_dict,
            default_emb=default_emb,
            similarity_type="cosine"
        )
        output = output.with_columns([
            pl.Series(name=f"{item}_prone_u2i_cos", values=scores)
        ])

    return output



if __name__ == "__main__":
    # Create dummy data
    df = pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "item_id": ["A", "B", "A", "C", "B", "D", "A"]
    })

    target_items = ["A", "B", "C", "D"]

    # Generate features using ProNE embeddings
    features = generate_u2i_prone_features(
        df=df,
        user_col="user_id",
        item_col="item_id",
        target_items=target_items,
        emb_size=16,  # small for testing
        step=5,
        theta=0.5,
        mu=0.2,
        n_iter=3,
        random_state=42
    )

    print("=== ProNE-based User-to-Item Cosine Similarity Features ===")
    print(features)
