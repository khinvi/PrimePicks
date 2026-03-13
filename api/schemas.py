"""Pydantic request/response models for the PrimePicks API."""

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str
    n: int = Field(default=10, ge=1, le=100)


class RecommendedItem(BaseModel):
    asin: str
    title: str | None = None
    predicted_rating: float
    image_url: str | None = None


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedItem]


class PredictionRequest(BaseModel):
    user_id: str
    asin: str


class PredictionResponse(BaseModel):
    user_id: str
    asin: str
    predicted_rating: float


class SimilarItem(BaseModel):
    asin: str
    title: str | None = None
    similarity_score: float
    image_url: str | None = None


class SimilarItemsResponse(BaseModel):
    asin: str
    similar_items: list[SimilarItem]


class HealthResponse(BaseModel):
    status: str
    n_users: int
    n_items: int
    n_interactions: int
    sparsity: float
