"""Tests for the prediction module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.predict import get_popular_items, get_similar_items, predict_rating, recommend_top_n


@pytest.fixture
def model():
    """
    Small model for testing predictions.

    Interaction matrix (users x items):
         I0   I1   I2   I3
    U0 [ 5    3    0    0 ]
    U1 [ 4    0    2    0 ]
    U2 [ 0    4    5    3 ]

    Similarity matrix (items x items), already truncated:
         I0   I1   I2   I3
    I0 [ 0   0.8  0.3   0 ]
    I1 [0.8   0   0.5   0 ]
    I2 [0.3  0.5   0   0.9]
    I3 [ 0    0   0.9   0 ]
    """
    interaction = csr_matrix(np.array([
        [5, 3, 0, 0],
        [4, 0, 2, 0],
        [0, 4, 5, 3],
    ], dtype=np.float32))

    similarity = csr_matrix(np.array([
        [0, 0.8, 0.3, 0],
        [0.8, 0, 0.5, 0],
        [0.3, 0.5, 0, 0.9],
        [0, 0, 0.9, 0],
    ], dtype=np.float32))

    return interaction, similarity


class TestPredictRating:
    def test_basic_prediction(self, model):
        interaction, similarity = model

        # Predict U0's rating for I2
        # U0 rated: I0=5, I1=3
        # I2's neighbors that U0 rated: I0 (sim=0.3), I1 (sim=0.5)
        # r_hat = (0.3*5 + 0.5*3) / (0.3 + 0.5) = 3.0 / 0.8 = 3.75
        pred = predict_rating(0, 2, interaction, similarity)
        assert pred == pytest.approx(3.75, abs=0.01)

    def test_returns_none_for_no_similar_items(self, model):
        interaction, similarity = model

        # Predict U0's rating for I3
        # U0 rated: I0, I1. I3 neighbors with positive sim: only I2 (sim=0.9)
        # But U0 hasn't rated I2, so no overlap -> None
        pred = predict_rating(0, 3, interaction, similarity)
        assert pred is None

    def test_returns_none_for_user_with_no_ratings(self):
        interaction = csr_matrix(np.array([[0, 0, 0]], dtype=np.float32))
        similarity = csr_matrix(np.array([[0, 0.5, 0.3], [0.5, 0, 0.8], [0.3, 0.8, 0]], dtype=np.float32))

        pred = predict_rating(0, 1, interaction, similarity)
        assert pred is None


class TestRecommendTopN:
    def test_excludes_rated_items(self, model):
        interaction, similarity = model

        recs = recommend_top_n(0, interaction, similarity, n=10)
        rec_items = {item_idx for item_idx, _ in recs}

        # U0 rated I0 and I1, so they should not appear
        assert 0 not in rec_items
        assert 1 not in rec_items

    def test_returns_sorted_by_rating(self, model):
        interaction, similarity = model

        recs = recommend_top_n(2, interaction, similarity, n=10)
        if len(recs) > 1:
            for i in range(len(recs) - 1):
                assert recs[i][1] >= recs[i + 1][1]


class TestGetSimilarItems:
    def test_returns_correct_neighbors(self, model):
        _, similarity = model

        similar = get_similar_items(2, similarity, k=2)
        item_indices = [idx for idx, _ in similar]

        # I2's top neighbors: I3 (0.9), I1 (0.5)
        assert 3 in item_indices
        assert 1 in item_indices

    def test_sorted_by_score(self, model):
        _, similarity = model

        similar = get_similar_items(2, similarity, k=3)
        for i in range(len(similar) - 1):
            assert similar[i][1] >= similar[i + 1][1]


class TestGetPopularItems:
    def test_returns_items(self, model):
        interaction, _ = model
        popular = get_popular_items(interaction, n=2)

        assert len(popular) == 2
        assert all(isinstance(idx, int) for idx, _ in popular)
        assert all(isinstance(score, float) for _, score in popular)
