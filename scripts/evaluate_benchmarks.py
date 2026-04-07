"""
Run AutoDS end-to-end on labeled UCI benchmark datasets and compare the
held-out test score against published baselines.

This is a measurement script, not a framework. It:
  1. Loads a dataset from data/benchmark_data/
  2. Drops any leakage columns specified in the YAML
  3. Stratified 80/20 train/test split (random_state=42)
  4. Writes the train CSV and invokes the AutoDS LangGraph pipeline
  5. Replays the cleaning recipe on the test set (no fitted state in cleaning,
     so straight replay is methodologically sound)
  6. Scores the AutoGluon predictor on the cleaned test set with the metric
     specified in the YAML
  7. Writes results.json and prints a markdown summary table

Usage:
  AUTOGLUON_TIME_LIMIT=60 python scripts/evaluate_benchmarks.py --only ai4i_2020
  python scripts/evaluate_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so imports work when run from any CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.pipeline.graph import app  # noqa: E402
from core.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data" / "benchmark_data"
DEFAULT_YAML = REPO_ROOT / "data" / "benchmark_metadata.yaml"
OUTPUT_ROOT = REPO_ROOT / "output" / "benchmarks"
AUTOGLUON_DEFAULT_DIR = REPO_ROOT / "output" / "autogluon_model"


# ---------------------------------------------------------------------------
# Recipe replay
# ---------------------------------------------------------------------------

class RecipeReplayError(Exception):
    """Raised when a cleaning recipe step fails to execute on the test set."""


def replay_recipe(recipe, test_df: pd.DataFrame, target_col: str):
    """
    Re-execute each step of the cleaning recipe on the test DataFrame.

    AutoDS cleaning has no fitted state (no scalers, no imputers, only target
    dropna), so straight replay is correct - no train-test leakage to worry
    about.

    Returns: (X_test_cleaned, y_test_cleaned) aligned by index.

    Raises: RecipeReplayError on any step exception. Caller MUST NOT fall back.
    """
    df = test_df.copy()
    namespace = {"df": df, "pd": pd, "np": np}
    for step in recipe.steps:
        try:
            exec(step.python_code, namespace)
            df = namespace["df"]
            namespace["df"] = df
        except Exception as e:
            raise RecipeReplayError(
                f"Step {step.step_id} ({step.operation}) failed: {e}\n"
                f"Code: {step.python_code}"
            ) from e

    if target_col not in df.columns:
        raise RecipeReplayError(
            f"Target column '{target_col}' missing after replay"
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_with_metric(predictor, X_test, y_test, metric: str, task_type: str) -> float:
    """
    Score with the metric specified in YAML. Uses sklearn directly so the
    metric is exactly what we asked for, not whatever AutoGluon's eval_metric
    happens to be set to.

    Supported metrics: roc_auc, macro_f1, accuracy, rmse
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        roc_auc_score,
    )

    if metric == "roc_auc":
        proba = predictor.predict_proba(X_test)
        if task_type == "binary_classification":
            # AutoGluon returns a DataFrame with one column per class.
            positive_class = predictor.class_labels[-1]
            return float(roc_auc_score(y_test, proba[positive_class]))
        return float(roc_auc_score(y_test, proba, multi_class="ovr"))

    if metric == "macro_f1":
        preds = predictor.predict(X_test)
        return float(f1_score(y_test, preds, average="macro"))

    if metric == "accuracy":
        preds = predictor.predict(X_test)
        return float(accuracy_score(y_test, preds))

    if metric == "rmse":
        preds = predictor.predict(X_test)
        return float(mean_squared_error(y_test, preds, squared=False))

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_one(ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run AutoDS end-to-end on a single dataset and return a result row.

    On any failure, returns a result row with status != "ok" and an error
    message; never raises.
    """
    name = ds["name"]
    target_col = ds["target_column"]
    metric = ds["metric"]
    task_type = ds["task_type"]
    user_query = ds["user_query"]
    drop_columns = ds.get("drop_columns") or []

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    model_dst = out_dir / "autogluon_model"
    log_path = out_dir / f"{name}.log"

    setup_logger(log_file=str(log_path))

    result_row: Dict[str, Any] = {
        "name": name,
        "status": "ok",
        "metric": metric,
        "autods_test_score": None,
        "autogluon_val_score": None,
        "literature_baseline": ds.get("literature_baseline"),
        "literature_source": ds.get("literature_source"),
        "delta_vs_literature": None,
        "n_train": None,
        "n_test_input": None,
        "n_test_after_replay": None,
        "best_model": None,
        "training_time_seconds": None,
        "model_path": str(model_dst),
        "error": None,
    }

    try:
        # ----------------------------------------------------------------
        # 1. Load
        # ----------------------------------------------------------------
        data_path = DATA_DIR / ds["file"]
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        full_df = pd.read_parquet(data_path)
        print(f"\n[{name}] Loaded {len(full_df)} rows x {len(full_df.columns)} cols "
              f"from {data_path.name}")

        # 2. Drop leakage columns BEFORE the split
        if drop_columns:
            missing = [c for c in drop_columns if c not in full_df.columns]
            if missing:
                raise KeyError(f"drop_columns not present in dataset: {missing}")
            full_df = full_df.drop(columns=drop_columns)
            print(f"[{name}] Dropped leakage columns: {drop_columns}")

        if target_col not in full_df.columns:
            raise KeyError(f"target_column '{target_col}' not in dataset")

        # 3. Stratified 80/20 split
        stratify = (
            full_df[target_col]
            if task_type in ("binary_classification", "multiclass_classification")
            else None
        )
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )
        result_row["n_train"] = len(train_df)
        result_row["n_test_input"] = len(test_df)
        print(f"[{name}] Split: {len(train_df)} train / {len(test_df)} test")

        # 4. Write train CSV (matches scripts/run_graph.py entry path)
        train_df.to_csv(train_csv, index=False)

        # 5. Set AUTO_TARGET_COLUMN to bypass HITL prompt
        os.environ["AUTO_TARGET_COLUMN"] = target_col

        # 6. Build initial state identical to scripts/run_graph.py
        initial_state = {
            "user_query": user_query,
            "working_df": pd.read_csv(train_csv),
            "retry_count": 0,
            "tool_call_count": 0,
            "pass_count": 0,
            "target_column": None,
        }

        # 7. Invoke the graph
        print(f"[{name}] Invoking AutoDS graph...")
        graph_result = app.invoke(initial_state)

        model_meta = graph_result.get("model_metadata") or {}
        if model_meta.get("error"):
            raise RuntimeError(f"AutoGluon training failed: {model_meta['error']}")

        eval_metrics = model_meta.get("eval_metrics") or {}
        result_row["autogluon_val_score"] = eval_metrics.get("score_val")
        result_row["best_model"] = model_meta.get("best_model")
        result_row["training_time_seconds"] = model_meta.get("training_time_seconds")

        # 8. Copy the trained model out of the shared output dir
        if AUTOGLUON_DEFAULT_DIR.exists():
            if model_dst.exists():
                shutil.rmtree(model_dst)
            shutil.copytree(AUTOGLUON_DEFAULT_DIR, model_dst)
        else:
            raise FileNotFoundError(
                f"Expected AutoGluon model dir not found: {AUTOGLUON_DEFAULT_DIR}"
            )

        # 9. Replay the cleaning recipe on the test set
        recipe = graph_result.get("current_plan")
        if recipe is None:
            raise RuntimeError("Graph did not produce a cleaning recipe (current_plan)")

        try:
            X_test, y_test = replay_recipe(recipe, test_df, target_col)
        except RecipeReplayError as e:
            print(f"[{name}] Recipe replay FAILED: {e}")
            result_row["status"] = "replay_failed"
            result_row["error"] = str(e)
            return result_row

        result_row["n_test_after_replay"] = len(X_test)
        print(f"[{name}] Test set after replay: {len(X_test)} rows")

        # 10. Score using the YAML-specified metric
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(str(model_dst))
        score = score_with_metric(predictor, X_test, y_test, metric, task_type)
        result_row["autods_test_score"] = score
        if result_row["literature_baseline"] is not None:
            result_row["delta_vs_literature"] = (
                score - float(result_row["literature_baseline"])
            )
        print(f"[{name}] Test {metric} = {score:.4f}")

        return result_row

    except Exception as e:
        traceback.print_exc()
        result_row["status"] = (
            "graph_failed" if result_row["status"] == "ok" else result_row["status"]
        )
        result_row["error"] = f"{type(e).__name__}: {e}"
        return result_row
    finally:
        # Hygiene: never let AUTO_TARGET_COLUMN bleed across datasets
        os.environ.pop("AUTO_TARGET_COLUMN", None)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_score(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def print_markdown_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["Dataset", "Metric", "AutoDS", "Literature", "Delta", "Status"]
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join([
                str(r["name"]),
                str(r["metric"]),
                _fmt_score(r["autods_test_score"]),
                _fmt_score(r.get("literature_baseline")),
                _fmt_delta(r.get("delta_vs_literature")),
                str(r["status"]),
            ])
            + " |"
        )
    print(
        "\nLiterature baselines use different splits and sometimes different "
        "problem definitions. Treat deltas as ballpark, not pass/fail."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="Run a single dataset by name (matches the YAML 'name' field).",
    )
    parser.add_argument(
        "--datasets-yaml",
        type=Path,
        default=DEFAULT_YAML,
        help=f"Path to benchmark metadata YAML (default: {DEFAULT_YAML}).",
    )
    args = parser.parse_args()

    if not args.datasets_yaml.exists():
        sys.exit(f"ERROR: YAML not found at {args.datasets_yaml}")

    with open(args.datasets_yaml) as f:
        config = yaml.safe_load(f)
    datasets = config.get("datasets", [])
    if not datasets:
        sys.exit("ERROR: No datasets found in YAML")

    if args.only:
        datasets = [d for d in datasets if d["name"] == args.only]
        if not datasets:
            sys.exit(f"ERROR: Dataset '{args.only}' not found in YAML")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for ds in datasets:
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {ds['name']}")
        print("=" * 70)
        results.append(run_one(ds))

    results_path = OUTPUT_ROOT / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")

    print_markdown_table(results)


if __name__ == "__main__":
    main()
