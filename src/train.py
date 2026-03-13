"""Item-item collaborative filtering model training via cosine similarity."""

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.preprocessing import load_interaction_matrix


def compute_item_similarity(interaction_matrix: csr_matrix) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all items.

    Transposes the user-item matrix to item-user, so each item is a vector
    in user-space. Cosine similarity measures the angle between these vectors.

    Returns a dense (n_items, n_items) similarity matrix.
    """
    # Transpose to items x users
    item_user_matrix = interaction_matrix.T.tocsr()

    print(f"Computing cosine similarity for {item_user_matrix.shape[0]} items...")
    sim_matrix = cosine_similarity(item_user_matrix)

    # Zero out self-similarity (diagonal)
    np.fill_diagonal(sim_matrix, 0.0)

    return sim_matrix


def truncate_similarity(sim_matrix: np.ndarray, k: int = config.K_NEIGHBORS) -> csr_matrix:
    """
    Keep only the top-k most similar items per item to reduce noise.

    For each row, zero out all values except the top-k.
    Returns a sparse CSR matrix.
    """
    n_items = sim_matrix.shape[0]
    effective_k = min(k, n_items - 1)

    print(f"Truncating to top-{effective_k} neighbors per item...")
    truncated = np.zeros_like(sim_matrix)

    for i in range(n_items):
        row = sim_matrix[i]
        if effective_k >= len(row):
            truncated[i] = row
        else:
            # argpartition is O(n) vs O(n log n) for full sort
            top_k_idx = np.argpartition(row, -effective_k)[-effective_k:]
            truncated[i, top_k_idx] = row[top_k_idx]

    sparse_sim = csr_matrix(truncated)
    save_npz(config.SIMILARITY_MATRIX_PATH, sparse_sim)
    print(f"Similarity matrix saved: {sparse_sim.nnz} non-zero entries")
    return sparse_sim


def load_similarity_matrix() -> csr_matrix:
    """Load the saved similarity matrix from disk."""
    return load_npz(config.SIMILARITY_MATRIX_PATH).tocsr()


def train_model(interaction_matrix: csr_matrix | None = None) -> csr_matrix:
    """
    Full training pipeline: compute similarity and truncate.
    Returns the truncated sparse similarity matrix.
    """
    if interaction_matrix is None:
        interaction_matrix = load_interaction_matrix()

    sim_matrix = compute_item_similarity(interaction_matrix)
    sparse_sim = truncate_similarity(sim_matrix)

    print("Training complete.")
    return sparse_sim
