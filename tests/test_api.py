"""Tests for the FastAPI endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from api.main import app
from api.routes import init_state


@pytest.fixture(autouse=True)
def setup_state():
    """Initialize the API state with a small test model."""
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

    user_to_idx = {"USER_A": 0, "USER_B": 1, "USER_C": 2}
    item_to_idx = {"ASIN_0": 0, "ASIN_1": 1, "ASIN_2": 2, "ASIN_3": 3}
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    idx_to_user = {v: k for k, v in user_to_idx.items()}

    item_metadata = {
        "ASIN_0": {"title": "Nike Air Max", "images": None},
        "ASIN_1": {"title": "Adidas Ultraboost", "images": None},
        "ASIN_2": {"title": "Jordan Retro", "images": None},
        "ASIN_3": {"title": "New Balance 990", "images": None},
    }

    init_state({
        "interaction_matrix": interaction,
        "sim_matrix": similarity,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
        "item_metadata": item_metadata,
    })


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


class TestHealthEndpoint:
    def test_returns_stats(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["n_users"] == 3
        assert data["n_items"] == 4


class TestRecommendEndpoint:
    def test_known_user(self, client):
        resp = client.post("/recommend", json={"user_id": "USER_A", "n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "USER_A"
        assert len(data["recommendations"]) <= 2

    def test_unknown_user_gets_popular(self, client):
        resp = client.post("/recommend", json={"user_id": "UNKNOWN", "n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) > 0


class TestPredictEndpoint:
    def test_known_user_and_item(self, client):
        resp = client.post("/predict", json={"user_id": "USER_A", "asin": "ASIN_2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_rating" in data

    def test_unknown_user(self, client):
        resp = client.post("/predict", json={"user_id": "UNKNOWN", "asin": "ASIN_0"})
        assert resp.status_code == 404

    def test_unknown_item(self, client):
        resp = client.post("/predict", json={"user_id": "USER_A", "asin": "UNKNOWN"})
        assert resp.status_code == 404


class TestSimilarEndpoint:
    def test_returns_similar_items(self, client):
        resp = client.get("/similar/ASIN_2?k=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asin"] == "ASIN_2"
        assert len(data["similar_items"]) <= 2

    def test_unknown_asin(self, client):
        resp = client.get("/similar/UNKNOWN")
        assert resp.status_code == 404
