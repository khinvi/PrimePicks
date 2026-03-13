"""API endpoints for the PrimePicks recommendation engine."""

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    SimilarItem,
    SimilarItemsResponse,
)
from src.predict import get_popular_items, get_similar_items, predict_rating, recommend_top_n

router = APIRouter()

# These are populated by the lifespan handler in main.py
_state = {}


def init_state(state: dict):
    _state.update(state)


def _item_info(item_idx: int) -> dict:
    """Look up ASIN and metadata for an item index."""
    idx_to_item = _state["idx_to_item"]
    metadata = _state.get("item_metadata", {})
    asin = idx_to_item[item_idx]
    meta = metadata.get(asin, {})
    return {
        "asin": asin,
        "title": meta.get("title"),
        "image_url": meta.get("images"),
    }


@router.get("/health", response_model=HealthResponse)
def health():
    matrix = _state["interaction_matrix"]
    n_users, n_items = matrix.shape
    n_interactions = matrix.nnz
    sparsity = 1.0 - n_interactions / (n_users * n_items) if (n_users * n_items) > 0 else 1.0
    return HealthResponse(
        status="healthy",
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions,
        sparsity=round(sparsity, 6),
    )


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    user_to_idx = _state["user_to_idx"]
    interaction_matrix = _state["interaction_matrix"]
    sim_matrix = _state["sim_matrix"]

    if req.user_id not in user_to_idx:
        # Cold start fallback: popular items
        popular = get_popular_items(interaction_matrix, n=req.n)
        items = []
        for item_idx, avg_rating in popular:
            info = _item_info(item_idx)
            items.append(RecommendedItem(predicted_rating=round(avg_rating, 2), **info))
        return RecommendationResponse(user_id=req.user_id, recommendations=items)

    user_idx = user_to_idx[req.user_id]
    recs = recommend_top_n(user_idx, interaction_matrix, sim_matrix, n=req.n)

    items = []
    for item_idx, pred in recs:
        info = _item_info(item_idx)
        items.append(RecommendedItem(predicted_rating=round(pred, 2), **info))

    return RecommendationResponse(user_id=req.user_id, recommendations=items)


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    user_to_idx = _state["user_to_idx"]
    item_to_idx = _state["item_to_idx"]
    interaction_matrix = _state["interaction_matrix"]
    sim_matrix = _state["sim_matrix"]

    if req.user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown user: {req.user_id}")
    if req.asin not in item_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown item: {req.asin}")

    user_idx = user_to_idx[req.user_id]
    item_idx = item_to_idx[req.asin]

    pred = predict_rating(user_idx, item_idx, interaction_matrix, sim_matrix)
    if pred is None:
        raise HTTPException(status_code=422, detail="Cannot predict: insufficient data")

    return PredictionResponse(
        user_id=req.user_id, asin=req.asin, predicted_rating=round(pred, 2)
    )


@router.get("/similar/{asin}", response_model=SimilarItemsResponse)
def similar(asin: str, k: int = Query(default=10, ge=1, le=100)):
    item_to_idx = _state["item_to_idx"]
    sim_matrix = _state["sim_matrix"]

    if asin not in item_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown item: {asin}")

    item_idx = item_to_idx[asin]
    similar_items = get_similar_items(item_idx, sim_matrix, k=k)

    items = []
    for idx, score in similar_items:
        info = _item_info(idx)
        items.append(SimilarItem(similarity_score=round(score, 4), **info))

    return SimilarItemsResponse(asin=asin, similar_items=items)
