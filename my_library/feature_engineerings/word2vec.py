from typing import Callable, Dict, List, Literal, Union

import numpy as np
import polars as pl
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

from my_library.utils.array_utils import compute_embedding_similarity
from my_library.utils.df_utils import df_io_polars


def train_item2vec_embedding(
    sequences: List[List[Union[str, int]]],
    embedding_dim: int = 100,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 10,
    seed: int = 42,
    workers: int = 4
) -> Dict[Union[str, int], np.ndarray]:
    """
    Train a Word2Vec model on sequences of item IDs and return a dictionary
    mapping each item ID to its L2-normalized embedding vector.

    Args:
        sequences: List of item ID sequences (e.g., purchase history per user).
        embedding_dim: Dimension of the resulting embedding vectors.
        window: Context window size used by Word2Vec.
        min_count: Minimum number of occurrences required to include an item.
        epochs: Number of training iterations over the data.
        seed: Random seed for reproducibility.
        workers: Number of worker threads to use for training.

    Returns:
        Dictionary mapping item ID to its learned embedding vector (as np.ndarray).
    """
    model = Word2Vec(
        sequences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    vocab = model.wv.index_to_key
    vectors = np.array([model.wv[word] for word in vocab], dtype=np.float32)
    vectors = normalize(vectors, norm="l2")
    return {vocab[i]: vectors[i] for i in range(len(vocab))}


@df_io_polars(return_type="polars")
def compute_item_target_similarities(
    df: pl.DataFrame,
    item_col: str,
    embedding_dict: Dict[Union[str, int], np.ndarray],
    targets: List[Union[str, int]],
    embedding_dim: int,
    similarity_type: Literal["dot", "cosine"] = "dot"
) -> pl.DataFrame:
    """
    Compute similarity between each item's embedding and multiple target items.
    The similarity values are appended to the original DataFrame.

    Args:
        df: Polars DataFrame containing the item column.
        item_col: Name of the column containing item IDs.
        embedding_dict: Dictionary mapping item ID to its embedding vector.
        targets: List of target item IDs to compare similarity against.
        embedding_dim: Dimension of the embedding vectors.
        similarity_type: Type of similarity to compute: "dot" or "cosine".

    Returns:
        Polars DataFrame with additional columns for similarity to each target.
    """
    default_emb = np.zeros(embedding_dim, dtype=np.float32)
    item_ids = df[item_col].to_list()

    for target in targets:
        scores = compute_embedding_similarity(
            user_ids=item_ids,
            item_ids=[target] * len(item_ids),
            user_emb_dict=embedding_dict,
            item_emb_dict=embedding_dict,
            default_emb=default_emb,
            similarity_type=similarity_type
        )
        df = df.with_columns(
            pl.Series(name=f"{target}_w2v_i2i_{similarity_type}", values=scores)
        )

    return df


@df_io_polars(return_type="polars")
def aggregate_user_item_similarities(
    df: pl.DataFrame,
    user_col: str,
    targets: List[Union[str, int]],
    similarity_type: str = "dot",
    agg_func_map: Dict[str, Callable[[pl.Series], float]] = None
) -> pl.DataFrame:
    """
    Aggregate similarity scores per user for each target item using custom aggregation functions.

    Args:
        df: Polars DataFrame with similarity scores and user ID column.
        user_col: Column name representing the user ID.
        targets: List of target item IDs.
        similarity_type: Type of similarity used (e.g., "dot" or "cosine").
        agg_func_map: Optional dictionary mapping aggregation name to a callable function.
                      If not provided, default functions (mean, std, sum, max, min, last) are used.

    Returns:
        Polars DataFrame with aggregated features for each user and target item.
    """
    if agg_func_map is None:
        agg_func_map = {
            "mean": lambda col: col.mean(),
            "std": lambda col: col.std(),
            "sum": lambda col: col.sum(),
            "max": lambda col: col.max(),
            "min": lambda col: col.min(),
            "last": lambda col: col.tail(1).first(),
        }

    agg_exprs = []
    for t in targets:
        col_name = f"{t}_w2v_i2i_{similarity_type}"
        for name, func in agg_func_map.items():
            agg_exprs.append(func(pl.col(col_name)).alias(f"{col_name}_{name}"))

    return df.group_by(user_col).agg(agg_exprs)


@df_io_polars(return_type="polars")
def generate_item2vec_user_features(
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    targets: List[Union[str, int]],
    embedding_dim: int = 100,
    similarity_type: str = "dot",
    agg_func_map: Dict[str, Callable[[pl.Series], float]] = None
) -> pl.DataFrame:
    """
    Main pipeline to generate user-level features using Word2Vec item embeddings.
    For each user, compute similarity between purchased items and given targets,
    then aggregate these similarity scores.

    Args:
        df: Polars DataFrame with user, item, and timestamp columns.
        user_col: Column name indicating user IDs.
        item_col: Column name indicating item IDs.
        targets: List of item IDs to compare against.
        embedding_dim: Dimension of Word2Vec embeddings.
        similarity_type: Similarity metric to use ("dot" or "cosine").
        agg_func_map: Optional aggregation functions (mean, max, etc.) to apply.

    Returns:
        Polars DataFrame with user-level aggregated similarity features.
    """
    df_sorted = df.sort([user_col, "ts"])
    grouped = df_sorted.group_by(user_col).agg([pl.col(item_col).alias("item_seq")])
    sequences = grouped.get_column("item_seq").to_list()
    embedding_dict = train_item2vec_embedding(sequences, embedding_dim=embedding_dim)

    df_with_sim = compute_item_target_similarities(
        df_sorted, item_col, embedding_dict, targets, embedding_dim, similarity_type
    )

    return aggregate_user_item_similarities(
        df_with_sim, user_col, targets, similarity_type, agg_func_map
    )


if __name__ == "__main__":
    dummy_df = pl.DataFrame({
        "user_id": ["U1", "U1", "U2", "U3", "U3", "U4", "U5", "U5", "U5"],
        "item_id": ["A", "B", "A", "C", "B", "C", "A", "B", "C"],
        "ts": [1, 2, 1, 1, 2, 3, 1, 2, 3]  # Dummy timestamp
    })

    features = generate_item2vec_user_features(
        df=dummy_df,
        user_col="user_id",
        item_col="item_id",
        targets=["A", "B", "C"],
        embedding_dim=16,
        similarity_type="dot"
    )

    print(features)
