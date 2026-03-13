"""Tests for the model training module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.train import compute_item_similarity, truncate_similarity
import config


@pytest.fixture
def simple_matrix():
    """
    A small user-item interaction matrix:
         Item0  Item1  Item2
    U0 [  5      3      0  ]
    U1 [  4      0      2  ]
    U2 [  0      4      5  ]
    """
    return csr_matrix(np.array([
        [5, 3, 0],
        [4, 0, 2],
        [0, 4, 5],
    ], dtype=np.float32))


class TestComputeItemSimilarity:
    def test_output_shape(self, simple_matrix):
        sim = compute_item_similarity(simple_matrix)
        assert sim.shape == (3, 3)

    def test_diagonal_is_zero(self, simple_matrix):
        sim = compute_item_similarity(simple_matrix)
        np.testing.assert_array_equal(np.diag(sim), [0, 0, 0])

    def test_symmetry(self, simple_matrix):
        sim = compute_item_similarity(simple_matrix)
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_values_between_neg1_and_1(self, simple_matrix):
        sim = compute_item_similarity(simple_matrix)
        assert np.all(sim >= -1.0 - 1e-6)
        assert np.all(sim <= 1.0 + 1e-6)

    def test_manual_cosine(self, simple_matrix):
        """Verify against hand-computed cosine similarity."""
        sim = compute_item_similarity(simple_matrix)

        # Item0 vector: [5, 4, 0], Item1 vector: [3, 0, 4]
        v0 = np.array([5, 4, 0], dtype=float)
        v1 = np.array([3, 0, 4], dtype=float)
        expected = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

        np.testing.assert_almost_equal(sim[0, 1], expected, decimal=5)


class TestTruncateSimilarity:
    def test_keeps_top_k(self):
        sim = np.array([
            [0.0, 0.9, 0.5, 0.1],
            [0.9, 0.0, 0.3, 0.8],
            [0.5, 0.3, 0.0, 0.7],
            [0.1, 0.8, 0.7, 0.0],
        ])

        original_path = config.SIMILARITY_MATRIX_PATH
        import tempfile, os
        config.SIMILARITY_MATRIX_PATH = os.path.join(tempfile.mkdtemp(), "sim.npz")

        truncated = truncate_similarity(sim, k=2)

        config.SIMILARITY_MATRIX_PATH = original_path

        # Each row should have at most 2 non-zero entries
        for i in range(4):
            row = truncated[i].toarray().flatten()
            assert np.count_nonzero(row) <= 2

    def test_preserves_top_values(self):
        sim = np.array([
            [0.0, 0.9, 0.5, 0.1],
            [0.9, 0.0, 0.3, 0.8],
            [0.5, 0.3, 0.0, 0.7],
            [0.1, 0.8, 0.7, 0.0],
        ])

        original_path = config.SIMILARITY_MATRIX_PATH
        import tempfile, os
        config.SIMILARITY_MATRIX_PATH = os.path.join(tempfile.mkdtemp(), "sim.npz")

        truncated = truncate_similarity(sim, k=2)

        config.SIMILARITY_MATRIX_PATH = original_path

        # Row 0: top-2 are 0.9 (col 1) and 0.5 (col 2)
        row0 = truncated[0].toarray().flatten()
        assert row0[1] == pytest.approx(0.9)
        assert row0[2] == pytest.approx(0.5)
