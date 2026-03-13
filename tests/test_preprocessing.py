"""Tests for the preprocessing module."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.preprocessing import (
    build_id_mappings,
    build_sparse_matrix,
    filter_by_min_counts,
    train_test_split,
)
import config


@pytest.fixture
def sample_df():
    """A small sample DataFrame of triplets."""
    return pd.DataFrame({
        "user_id": ["U1", "U1", "U1", "U2", "U2", "U2", "U3", "U3", "U3",
                     "U4", "U4", "U4", "U5", "U5", "U5"],
        "parent_asin": ["A", "B", "C", "A", "B", "D", "A", "C", "D",
                         "B", "C", "D", "A", "B", "E"],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0,
                   2.0, 3.0, 4.0, 5.0, 4.0, 1.0],
    })


@pytest.fixture
def tmp_config(tmp_path):
    """Override config paths for tests."""
    original = {
        "PROCESSED_DIR": config.PROCESSED_DIR,
        "USER_MAPPING_PATH": config.USER_MAPPING_PATH,
        "ITEM_MAPPING_PATH": config.ITEM_MAPPING_PATH,
        "INTERACTION_MATRIX_PATH": config.INTERACTION_MATRIX_PATH,
        "TRAIN_TRIPLETS_PATH": config.TRAIN_TRIPLETS_PATH,
        "TEST_TRIPLETS_PATH": config.TEST_TRIPLETS_PATH,
    }

    config.PROCESSED_DIR = str(tmp_path)
    config.USER_MAPPING_PATH = str(tmp_path / "user_to_idx.json")
    config.ITEM_MAPPING_PATH = str(tmp_path / "item_to_idx.json")
    config.INTERACTION_MATRIX_PATH = str(tmp_path / "interaction_matrix.npz")
    config.TRAIN_TRIPLETS_PATH = str(tmp_path / "train_triplets.csv")
    config.TEST_TRIPLETS_PATH = str(tmp_path / "test_triplets.csv")

    yield tmp_path

    for k, v in original.items():
        setattr(config, k, v)


class TestFilterByMinCounts:
    def test_removes_sparse_users_and_items(self):
        df = pd.DataFrame({
            "user_id": ["U1", "U1", "U1", "U2", "U3"],
            "parent_asin": ["A", "B", "C", "A", "A"],
            "rating": [5, 4, 3, 2, 1],
        })
        # With defaults MIN_RATINGS_PER_USER=3, MIN_RATINGS_PER_ITEM=5
        # Only U1 has 3+ ratings, but no item has 5+ ratings
        # So after filtering, nothing should remain
        original_min_user = config.MIN_RATINGS_PER_USER
        original_min_item = config.MIN_RATINGS_PER_ITEM

        config.MIN_RATINGS_PER_USER = 2
        config.MIN_RATINGS_PER_ITEM = 2
        result = filter_by_min_counts(df)

        # U1 has 3 ratings, U2 has 1, U3 has 1
        # Item A has 3 ratings, B has 1, C has 1
        # After filter: only U1 and A survive with min=2
        # But then U1 only has item A left (1 rating), below threshold...
        # Actually U1 rates A, B, C. Items with >=2: only A (3 ratings).
        # After item filter, U1 has 1 rating (A), U2 has 1 (A), U3 has 1 (A)
        # After user filter (>=2), none survive.
        assert len(result) == 0

        config.MIN_RATINGS_PER_USER = original_min_user
        config.MIN_RATINGS_PER_ITEM = original_min_item

    def test_keeps_dense_data(self, sample_df):
        original_min_user = config.MIN_RATINGS_PER_USER
        original_min_item = config.MIN_RATINGS_PER_ITEM

        config.MIN_RATINGS_PER_USER = 2
        config.MIN_RATINGS_PER_ITEM = 2
        result = filter_by_min_counts(sample_df)

        assert len(result) > 0
        # All users should have >= 2 ratings
        for _, group in result.groupby("user_id"):
            assert len(group) >= 2
        # All items should have >= 2 ratings
        for _, group in result.groupby("parent_asin"):
            assert len(group) >= 2

        config.MIN_RATINGS_PER_USER = original_min_user
        config.MIN_RATINGS_PER_ITEM = original_min_item


class TestBuildIdMappings:
    def test_mappings_are_bijective(self, sample_df, tmp_config):
        user_to_idx, item_to_idx = build_id_mappings(sample_df)

        # All unique users/items are mapped
        assert len(user_to_idx) == sample_df["user_id"].nunique()
        assert len(item_to_idx) == sample_df["parent_asin"].nunique()

        # Values are continuous integers starting from 0
        assert set(user_to_idx.values()) == set(range(len(user_to_idx)))
        assert set(item_to_idx.values()) == set(range(len(item_to_idx)))

        # Mappings saved to disk
        assert os.path.exists(config.USER_MAPPING_PATH)
        assert os.path.exists(config.ITEM_MAPPING_PATH)


class TestBuildSparseMatrix:
    def test_matrix_shape_and_values(self, sample_df, tmp_config):
        user_to_idx, item_to_idx = build_id_mappings(sample_df)
        matrix = build_sparse_matrix(sample_df, user_to_idx, item_to_idx)

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(user_to_idx), len(item_to_idx))
        assert matrix.nnz == len(sample_df)

        # Check a specific value
        u1_idx = user_to_idx["U1"]
        a_idx = item_to_idx["A"]
        assert matrix[u1_idx, a_idx] == 5.0


class TestTrainTestSplit:
    def test_split_preserves_all_data(self, sample_df, tmp_config):
        train, test = train_test_split(sample_df)

        assert len(train) + len(test) == len(sample_df)

    def test_every_user_has_test_data(self, sample_df, tmp_config):
        _, test = train_test_split(sample_df)
        assert test["user_id"].nunique() == sample_df["user_id"].nunique()
