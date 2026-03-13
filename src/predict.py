"""Rating prediction and top-N recommendation generation."""

import numpy as np
from scipy.sparse import csr_matrix

import config


def predict_rating(
    user_idx: int,
    item_idx: int,
    interaction_matrix: csr_matrix,
    sim_matrix: csr_matrix,
) -> float | None:
    """
    Predict user u's rating for item i using item-item collaborative filtering.

    r_hat(u, i) = sum(sim(i,j) * r(u,j)) / sum(|sim(i,j)|)
    for j in N(i; u) — items similar to i that user u has rated.

    Returns predicted rating or None if prediction is not possible.
    """
    # Items the user has rated (sparse row)
    user_ratings = interaction_matrix[user_idx]
    rated_items = user_ratings.indices
    rated_values = user_ratings.data

    if len(rated_items) == 0:
        return None

    # Similarities between target item and all items the user rated
    item_sims = sim_matrix[item_idx]
    if isinstance(item_sims, csr_matrix):
        # Extract similarities for rated items
        sims = np.array(item_sims[:, rated_items].todense()).flatten()
    else:
        sims = np.asarray(item_sims[rated_items]).flatten()

    # Only consider items with positive similarity
    mask = sims > 0
    if not mask.any():
        return None

    numerator = np.dot(sims[mask], rated_values[mask])
    denominator = np.sum(np.abs(sims[mask]))

    if denominator == 0:
        return None

    return float(numerator / denominator)


def recommend_top_n(
    user_idx: int,
    interaction_matrix: csr_matrix,
    sim_matrix: csr_matrix,
    n: int = config.DEFAULT_TOP_N,
) -> list[tuple[int, float]]:
    """
    Generate top-N item recommendations for a user.

    Predicts ratings for all items the user hasn't rated, returns the
    top-n items sorted by predicted rating descending.

    Returns list of (item_idx, predicted_rating) tuples.
    """
    n_items = interaction_matrix.shape[1]

    # Items the user has already rated
    rated_items = set(interaction_matrix[user_idx].indices)

    predictions = []
    for item_idx in range(n_items):
        if item_idx in rated_items:
            continue

        pred = predict_rating(user_idx, item_idx, interaction_matrix, sim_matrix)
        if pred is not None:
            predictions.append((item_idx, pred))

    # Sort by predicted rating descending, take top n
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def get_similar_items(
    item_idx: int,
    sim_matrix: csr_matrix,
    k: int = config.DEFAULT_TOP_N,
) -> list[tuple[int, float]]:
    """
    Return the top-k most similar items to a given item.
    Returns list of (item_idx, similarity_score) tuples.
    """
    row = sim_matrix[item_idx]
    if isinstance(row, csr_matrix):
        indices = row.indices
        scores = row.data
    else:
        scores = np.asarray(row).flatten()
        indices = np.where(scores > 0)[0]
        scores = scores[indices]

    # Sort by score descending
    order = np.argsort(-scores)[:k]
    return [(int(indices[i]), float(scores[i])) for i in order]


def get_popular_items(
    interaction_matrix: csr_matrix,
    n: int = config.DEFAULT_TOP_N,
) -> list[tuple[int, float]]:
    """
    Fallback: return the most-rated items with their average rating.
    Used for cold-start users with no rating history.
    """
    n_ratings = np.diff(interaction_matrix.T.tocsr().indptr)  # ratings per item
    avg_ratings = np.zeros(interaction_matrix.shape[1])

    item_matrix = interaction_matrix.T.tocsr()
    for i in range(interaction_matrix.shape[1]):
        data = item_matrix[i].data
        if len(data) > 0:
            avg_ratings[i] = data.mean()

    # Score: combine popularity and average rating
    scores = n_ratings * avg_ratings
    top_idx = np.argsort(-scores)[:n]
    return [(int(idx), float(avg_ratings[idx])) for idx in top_idx]
