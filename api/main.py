"""FastAPI application for the PrimePicks recommendation engine."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api.routes import init_state, router
from src.preprocessing import load_id_mappings, load_interaction_matrix
from src.train import load_similarity_matrix


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts into memory at startup."""
    print("Loading model artifacts...")

    interaction_matrix = load_interaction_matrix()
    sim_matrix = load_similarity_matrix()
    user_to_idx, item_to_idx = load_id_mappings()

    # Reverse mappings for lookup
    idx_to_user = {int(v): k for k, v in user_to_idx.items()}
    idx_to_item = {int(v): k for k, v in item_to_idx.items()}

    # Item metadata (optional, may not exist)
    item_metadata = {}
    try:
        with open(config.ITEM_METADATA_PATH) as f:
            item_metadata = json.load(f)
    except FileNotFoundError:
        print("Warning: item_metadata.json not found, titles will be unavailable")

    init_state({
        "interaction_matrix": interaction_matrix,
        "sim_matrix": sim_matrix,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
        "item_metadata": item_metadata,
    })

    print(f"Model loaded: {interaction_matrix.shape[0]} users, "
          f"{interaction_matrix.shape[1]} items, {sim_matrix.nnz} similarity entries")
    yield


app = FastAPI(
    title="PrimePicks",
    description="Athletic sneaker recommendation engine powered by item-item collaborative filtering.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
