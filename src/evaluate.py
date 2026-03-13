"""Model evaluation with RMSE and MAE metrics."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import config
from src.predict import predict_rating


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - actuals)))


def evaluate_model(
    test_df: pd.DataFrame,
    interaction_matrix: csr_matrix,
    sim_matrix: csr_matrix,
    user_to_idx: dict,
    item_to_idx: dict,
) -> dict:
    """
    Evaluate the model on the test set.
    Returns dict with rmse, mae, and coverage (fraction of predictable test ratings).
    """
    predictions = []
    actuals = []
    skipped = 0

    print(f"Evaluating on {len(test_df)} test ratings...")
    for _, row in test_df.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["parent_asin"])

        # Skip if user or item not in training set
        if user_id not in user_to_idx or item_id not in item_to_idx:
            skipped += 1
            continue

        user_idx = user_to_idx[user_id]
        item_idx = item_to_idx[item_id]

        pred = predict_rating(user_idx, item_idx, interaction_matrix, sim_matrix)
        if pred is None:
            skipped += 1
            continue

        predictions.append(pred)
        actuals.append(row["rating"])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    total = len(test_df)
    evaluated = len(predictions)

    results = {
        "rmse": compute_rmse(predictions, actuals) if evaluated > 0 else None,
        "mae": compute_mae(predictions, actuals) if evaluated > 0 else None,
        "coverage": evaluated / total if total > 0 else 0.0,
        "evaluated": evaluated,
        "skipped": skipped,
        "total": total,
    }

    print(f"Results: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}, "
          f"Coverage={results['coverage']:.2%} ({evaluated}/{total})")
    return results
