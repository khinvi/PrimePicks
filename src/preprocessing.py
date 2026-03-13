"""ID translation, filtering, train/test split, and sparse matrix construction."""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz

import config


def load_triplets(filepath: str = config.TRIPLETS_PATH) -> pd.DataFrame:
    """Load the raw triplets CSV."""
    return pd.read_csv(filepath, dtype={"user_id": str, "parent_asin": str, "rating": float})


def filter_by_min_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iteratively remove users and items below minimum rating thresholds.
    Repeats until stable since removing items can drop users below threshold and vice versa.
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)

        # Filter items
        item_counts = df["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= config.MIN_RATINGS_PER_ITEM].index
        df = df[df["parent_asin"].isin(valid_items)]

        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= config.MIN_RATINGS_PER_USER].index
        df = df[df["user_id"].isin(valid_users)]

    print(f"After filtering: {len(df)} triplets, "
          f"{df['user_id'].nunique()} users, {df['parent_asin'].nunique()} items")
    return df.reset_index(drop=True)


def build_id_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create bidirectional mappings from string IDs to continuous integers.
    Returns (user_to_idx, item_to_idx).
    """
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["parent_asin"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {asin: i for i, asin in enumerate(unique_items)}

    # Save mappings
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(config.USER_MAPPING_PATH, "w") as f:
        json.dump(user_to_idx, f)
    with open(config.ITEM_MAPPING_PATH, "w") as f:
        json.dump(item_to_idx, f)

    print(f"ID mappings: {len(user_to_idx)} users, {len(item_to_idx)} items")
    return user_to_idx, item_to_idx


def load_id_mappings() -> tuple[dict, dict]:
    """Load saved ID mappings from disk."""
    with open(config.USER_MAPPING_PATH) as f:
        user_to_idx = json.load(f)
    with open(config.ITEM_MAPPING_PATH) as f:
        item_to_idx = json.load(f)
    return user_to_idx, item_to_idx


def build_sparse_matrix(
    df: pd.DataFrame, user_to_idx: dict, item_to_idx: dict
) -> csr_matrix:
    """
    Build a sparse user-item interaction matrix from triplets DataFrame.
    Shape: (n_users, n_items), values are ratings.
    """
    rows = df["user_id"].map(user_to_idx).values
    cols = df["parent_asin"].map(item_to_idx).values
    data = df["rating"].values.astype(np.float32)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    matrix = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

    # Save
    save_npz(config.INTERACTION_MATRIX_PATH, matrix)
    sparsity = 1.0 - matrix.nnz / (n_users * n_items)
    print(f"Interaction matrix: {n_users}x{n_items}, "
          f"{matrix.nnz} non-zero, sparsity={sparsity:.4%}")
    return matrix


def load_interaction_matrix() -> csr_matrix:
    """Load the sparse interaction matrix from disk."""
    return load_npz(config.INTERACTION_MATRIX_PATH).tocsr()


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user train/test split to avoid data leakage.
    For each user, hold out a fraction of their ratings for testing.
    """
    train_rows = []
    test_rows = []

    for _, user_df in df.groupby("user_id"):
        n_test = max(1, int(len(user_df) * config.TEST_RATIO))
        shuffled = user_df.sample(frac=1, random_state=42)
        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    train_df.to_csv(config.TRAIN_TRIPLETS_PATH, index=False)
    test_df.to_csv(config.TEST_TRIPLETS_PATH, index=False)

    print(f"Train/test split: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def preprocess_all() -> tuple[csr_matrix, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline. Returns matrix, mappings, and splits."""
    df = load_triplets()
    df = filter_by_min_counts(df)

    train_df, test_df = train_test_split(df)

    # Build mappings and matrix from training data only
    user_to_idx, item_to_idx = build_id_mappings(train_df)
    matrix = build_sparse_matrix(train_df, user_to_idx, item_to_idx)

    return matrix, user_to_idx, item_to_idx, train_df, test_df
