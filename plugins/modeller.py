################################################################################
# AutoGluon model training module.
#
# Trains a TabularPredictor on the cleaned DataFrame and returns structured
# metadata (performance metrics, feature importance, leaderboard) that the
# Answer Agent uses to generate a business-friendly response.
#
# Configuration (environment variables):
#   AUTOGLUON_TIME_LIMIT  Training budget in seconds (default: 300)
#   AUTOGLUON_PRESETS     AutoGluon preset string (default: "medium_quality")
################################################################################

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from core.logger import log_model_metadata, log_node, logger


# Map InvestigationFindings.task_type → AutoGluon problem_type
_TASK_TYPE_MAP = {
    "binary_classification": "binary",
    "multiclass_classification": "multiclass",
    "regression": "regression",
}


def train_model(
    df: pd.DataFrame,
    user_query: str,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train an AutoGluon TabularPredictor on the cleaned data.

    Returns a metadata dict consumed by the Answer Agent. On failure, the dict
    contains an ``"error"`` key so the pipeline can continue gracefully.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if target_column is None:
        logger.warning("train_model called with no target column")
        return {"error": "No target column specified."}

    if target_column not in df.columns:
        logger.warning("Target column '%s' not in DataFrame", target_column)
        return {"error": f"Target column '{target_column}' not found in data."}

    if len(df) < 10:
        logger.warning("DataFrame too small for training (%d rows)", len(df))
        return {"error": f"Too few rows to train ({len(df)}). Need at least 10."}

    if df[target_column].nunique() < 2:
        logger.warning("Target column '%s' has < 2 unique values", target_column)
        return {"error": f"Target column '{target_column}' has only one unique value."}

    # ------------------------------------------------------------------
    # Safeguard: drop rows with NaN / inf / -inf in target column
    # ------------------------------------------------------------------
    pre_drop_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    target_invalid = df[target_column].isna()
    if target_invalid.any():
        df = df[~target_invalid].reset_index(drop=True)
        rows_dropped = pre_drop_count - len(df)
        logger.warning(
            "Dropped %d rows with NaN/inf in target column '%s' (%d → %d rows)",
            rows_dropped, target_column, pre_drop_count, len(df),
        )
        log_node("autogluon", "dropped invalid target rows",
                 target=target_column, rows_dropped=rows_dropped,
                 rows_before=pre_drop_count, rows_after=len(df))

    if len(df) < 10:
        logger.warning("DataFrame too small after dropping invalid target rows (%d rows)", len(df))
        return {"error": f"Too few rows after dropping NaN/inf in target ({len(df)}). Need at least 10."}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Normalize task_type (LLM may output "Binary Classification" etc.)
    normalized = task_type.lower().replace(" ", "_") if task_type else None
    problem_type = _TASK_TYPE_MAP.get(normalized)
    if task_type and problem_type is None:
        log_node("autogluon", "unrecognized task_type, falling back to auto-detect",
                 task_type=task_type)
    elif problem_type is None:
        log_node("autogluon", "no task_type provided, AutoGluon will auto-detect")
    time_limit = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "300"))
    presets = os.environ.get("AUTOGLUON_PRESETS", "medium_quality")
    save_path = str(Path(output_dir or "output") / "autogluon_model")

    log_node("autogluon", "starting training",
             target=target_column, problem_type=problem_type or "auto",
             time_limit=time_limit, presets=presets, rows=len(df),
             features=len(df.columns) - 1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    try:
        start = time.time()
        predictor = TabularPredictor(
            label=target_column,
            path=save_path,
            problem_type=problem_type,
        ).fit(
            train_data=df,
            time_limit=time_limit,
            presets=presets,
        )
        elapsed = round(time.time() - start, 2)

        log_node("autogluon", "training complete",
                 elapsed=elapsed, best_model=predictor.model_best)
    except Exception as e:
        logger.error("AutoGluon training failed: %s", e)
        return {
            "error": str(e),
            "target_column": target_column,
            "problem_type": problem_type,
        }

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    leaderboard = predictor.leaderboard(silent=True)

    # Use AutoGluon's internal validation scores (from the leaderboard) instead
    # of evaluating on training data, which would give inflated metrics.
    best_row = leaderboard.iloc[0] if not leaderboard.empty else {}
    eval_metrics = {"score_val": best_row.get("score_val")}
    eval_metric_name = getattr(predictor, "eval_metric", None)
    if eval_metric_name:
        eval_metrics["metric"] = str(eval_metric_name)

    # Permutation importance is expensive; sample large datasets to keep it fast.
    _MAX_FI_ROWS = 1000
    fi_df = df.sample(n=_MAX_FI_ROWS, random_state=0) if len(df) > _MAX_FI_ROWS else df
    try:
        feat_importance = predictor.feature_importance(fi_df, silent=True)
        feat_dict = feat_importance["importance"].to_dict()
    except Exception:
        feat_dict = {}

    metadata: Dict[str, Any] = {
        "target_column": target_column,
        "problem_type": predictor.problem_type,
        "eval_metrics": eval_metrics,
        "best_model": predictor.model_best,
        "leaderboard": leaderboard.head(10).to_dict(orient="records"),
        "feature_importance": feat_dict,
        "training_time_seconds": elapsed,
        "model_path": save_path,
        "num_rows_trained": len(df),
        "num_features": len(df.columns) - 1,
    }

    log_model_metadata(metadata)
    return metadata
