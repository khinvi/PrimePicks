#!/usr/bin/env python3
"""End-to-end pipeline: download -> filter -> preprocess -> train -> evaluate."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import (
    download_file,
    extract_review_triplets,
    filter_athletic_footwear_asins,
)
from src.evaluate import evaluate_model
from src.preprocessing import preprocess_all
from src.train import train_model


def main():
    # Step 1: Download raw data
    print("=" * 60)
    print("STEP 1: Download raw data")
    print("=" * 60)
    os.makedirs(config.RAW_DIR, exist_ok=True)

    if not os.path.exists(config.RAW_METADATA_PATH):
        download_file(config.METADATA_URL, config.RAW_METADATA_PATH)
    else:
        print(f"Metadata already downloaded: {config.RAW_METADATA_PATH}")

    if not os.path.exists(config.RAW_REVIEWS_PATH):
        download_file(config.REVIEWS_URL, config.RAW_REVIEWS_PATH)
    else:
        print(f"Reviews already downloaded: {config.RAW_REVIEWS_PATH}")

    # Step 2: Filter metadata for athletic footwear ASINs
    print("\n" + "=" * 60)
    print("STEP 2: Filter athletic footwear ASINs from metadata")
    print("=" * 60)
    if not os.path.exists(config.ATHLETIC_ASINS_PATH):
        valid_asins = filter_athletic_footwear_asins(config.RAW_METADATA_PATH)
    else:
        from src.data_loader import load_athletic_asins
        valid_asins = load_athletic_asins()
        print(f"Loaded {len(valid_asins)} athletic footwear ASINs from cache.")

    # Step 3: Extract review triplets
    print("\n" + "=" * 60)
    print("STEP 3: Extract review triplets for athletic footwear")
    print("=" * 60)
    if not os.path.exists(config.TRIPLETS_PATH):
        extract_review_triplets(config.RAW_REVIEWS_PATH, valid_asins)
    else:
        print(f"Triplets already extracted: {config.TRIPLETS_PATH}")

    # Step 4: Preprocess (filter, split, build matrix)
    print("\n" + "=" * 60)
    print("STEP 4: Preprocess data")
    print("=" * 60)
    matrix, user_to_idx, item_to_idx, train_df, test_df = preprocess_all()

    # Step 5: Train model
    print("\n" + "=" * 60)
    print("STEP 5: Train item-item similarity model")
    print("=" * 60)
    sim_matrix = train_model(matrix)

    # Step 6: Evaluate
    print("\n" + "=" * 60)
    print("STEP 6: Evaluate model")
    print("=" * 60)
    results = evaluate_model(test_df, matrix, sim_matrix, user_to_idx, item_to_idx)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  RMSE:     {results['rmse']:.4f}")
    print(f"  MAE:      {results['mae']:.4f}")
    print(f"  Coverage: {results['coverage']:.2%}")
    print(f"\nTo start the API server:")
    print(f"  uvicorn api.main:app --reload")
    print(f"  Then visit http://localhost:8000/docs")


if __name__ == "__main__":
    main()
