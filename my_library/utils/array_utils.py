from typing import Dict, List, Literal, Union

import numpy as np


def compute_embedding_similarity(
    user_ids: List[Union[str, int]],
    item_ids: List[Union[str, int]],
    user_emb_dict: Dict[Union[str, int], np.ndarray],
    item_emb_dict: Dict[Union[str, int], np.ndarray],
    default_emb: np.ndarray,
    similarity_type: Literal["dot", "cosine"] = "dot"
) -> np.ndarray:
    """
    Compute similarity (dot product or cosine) between user and item embeddings.

    Args:
        user_ids: List of user IDs.
        item_ids: List of item IDs.
        user_emb_dict: Mapping from user ID to embedding vector.
        item_emb_dict: Mapping from item ID to embedding vector.
        default_emb: Embedding used for missing users/items.
        similarity_type: 'dot' for inner product, 'cosine' for cosine similarity.

    Returns:
        Numpy array of similarity scores.
    """
    user_vectors = np.stack([user_emb_dict.get(uid, default_emb) for uid in user_ids])
    item_vectors = np.stack([item_emb_dict.get(iid, default_emb) for iid in item_ids])

    dot_products = np.sum(user_vectors * item_vectors, axis=1)

    if similarity_type == "dot":
        return dot_products
    elif similarity_type == "cosine":
        user_norms = np.linalg.norm(user_vectors, axis=1)
        item_norms = np.linalg.norm(item_vectors, axis=1)
        return dot_products / (user_norms * item_norms + 1e-8)
    else:
        raise ValueError(f"Unsupported similarity_type: {similarity_type}")


def apply_weighted_decay(values: List[float], base: float = 0.1) -> float:
    """
    Apply exponential decay to a list of values and return the weighted sum.

    This is useful for encoding recent actions with higher importance,
    such as [1, 0, 1] → 1 * base^0 + 0 * base^1 + 1 * base^2 = 1.0 + 0.0 + 0.01

    Parameters:
        values (List[float]): Sequence of values (e.g. binary history, scores).
                              Most recent item should be at the end of the list.
        base (float): Decay base (e.g. 0.1 for 1.0, 0.1, 0.01, ...). Default is 0.1.

    Returns:
        float: Weighted sum after applying exponential decay.

    Raises:
        ValueError: If `base` is not in (0, 1].
    """
    if not (0 < base <= 1.0):
        raise ValueError("base must be between 0 (exclusive) and 1.0 (inclusive).")
    
    return sum(v * (base ** i) for i, v in enumerate(reversed(values)))
