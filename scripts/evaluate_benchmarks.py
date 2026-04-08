"""
Run AutoDS end-to-end on labeled UCI benchmark datasets and compare a held-out
test score against published baselines.

This is a measurement script, not a framework. It:
  1. Loads a dataset from data/benchmark_data/
  2. Drops any leakage columns specified in the YAML
  3. Stratified 80/10/10 train/val/test split (random_state=42)
  4. Writes the train CSV and invokes the AutoDS LangGraph pipeline with a
     short AUTOGLUON_TIME_LIMIT so the graph's training is a fast throwaway
     (we keep clean_df + the cleaning recipe, not the model)
  5. Replays the cleaning recipe on the val and test folds (no fitted state in
     cleaning, so straight replay is methodologically sound)
  6. Trains a *fresh* AutoGluon TabularPredictor with the user's full time
     budget, passing the cleaned val fold as explicit ``tuning_data`` so model
     selection / ensemble weighting never see the held-out test fold
  7. Scores that predictor on the cleaned 10% test set with the metric
     specified in the YAML — this is the headline number
  8. Writes results.json and prints a markdown summary table

Usage:
  AUTOGLUON_TIME_LIMIT=60 python scripts/evaluate_benchmarks.py --only ai4i_2020
  python scripts/evaluate_benchmarks.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from autogluon.tabular import TabularPredictor  # noqa: E402

from core.agents.agents import run_answer_agent  # noqa: E402
from core.pipeline.graph import app  # noqa: E402
from core.logger import setup_logger  # noqa: E402
from plugins.modeller import build_model_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data" / "benchmark_data"
DEFAULT_YAML = REPO_ROOT / "data" / "benchmark_metadata.yaml"
OUTPUT_ROOT = REPO_ROOT / "output" / "benchmarks"

# AUTOGLUON_TIME_LIMIT we set on the graph's throwaway training. The user's
# original value is restored before the benchmark trains its real model.
GRAPH_THROWAWAY_TIME_LIMIT = "30"

# ---------------------------------------------------------------------------
# Metric / task-type maps
# ---------------------------------------------------------------------------
_YAML_TO_AUTOGLUON_METRIC = {
    "roc_auc": "roc_auc",
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "rmse": "root_mean_squared_error",
}

_TASK_TYPE_TO_PROBLEM_TYPE = {
    "binary_classification": "binary",
    "multiclass_classification": "multiclass",
    "regression": "regression",
}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def three_way_split(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    seed: int = 42,
):
    """
    Stratified 80 / 10 / 10 train / val / test split.

    Stratifies on the target for classification; no stratification for
    regression. Two ``train_test_split`` calls — first 80/20, then split the
    20% holdout in half.

    Returns: (train_df, val_df, test_df) — all DataFrames with target intact.
    """
    is_classification = task_type in (
        "binary_classification",
        "multiclass_classification",
    )
    stratify = df[target_col] if is_classification else None

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=stratify,
    )

    stratify_holdout = holdout_df[target_col] if is_classification else None
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=seed,
        stratify=stratify_holdout,
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Recipe replay
# ---------------------------------------------------------------------------

class RecipeReplayError(Exception):
    """Raised when a cleaning recipe step fails to execute on a held-out fold."""


def replay_steps(steps, raw_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Re-execute a flat list of cleaning steps on a raw DataFrame.

    The steps come from state["applied_steps"], which accumulates every
    successfully executed step across ALL cleaning passes. Replaying this on
    the raw held-out fold reproduces the graph's clean_df exactly.

    Do NOT use state["current_plan"] here — that field is reset between passes
    and only contains the final pass's recipe, so replaying it on raw data
    silently skips earlier passes' column drops.

    AutoDS cleaning has no fitted state (no scalers, no imputers, only target
    dropna), so straight replay is correct — no train-test leakage to worry
    about. Used for both val and test folds.

    Returns: cleaned DataFrame with the target column intact. The caller is
    responsible for separating X / y as needed.

    Raises: RecipeReplayError on any step exception. Caller MUST NOT fall back.
    """
    df = raw_df.copy()
    namespace = {"df": df, "pd": pd, "np": np}
    for step in steps:
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

    return df


# ---------------------------------------------------------------------------
# Final AutoGluon training (held-out val fold)
# ---------------------------------------------------------------------------

def train_final_model(
    train_clean: pd.DataFrame,
    val_clean: pd.DataFrame,
    ds: Dict[str, Any],
    output_path: Path,
) -> tuple[TabularPredictor, float]:
    """
    Train a fresh AutoGluon TabularPredictor with the explicit val fold as
    ``tuning_data``. This bypasses the graph's ``plugins/modeller.py`` path
    so we can pass ``tuning_data`` and the YAML-specified eval_metric.

    Reads AUTOGLUON_TIME_LIMIT from env (default 300). The benchmark script
    is responsible for restoring this env var to the user's original value
    before calling.

    Returns: (predictor, training_time_seconds).
    """
    time_limit = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "300"))
    presets = os.environ.get("AUTOGLUON_PRESETS", "medium_quality")
    problem_type = _TASK_TYPE_TO_PROBLEM_TYPE[ds["task_type"]]
    eval_metric = _YAML_TO_AUTOGLUON_METRIC[ds["metric"]]

    start = time.time()
    predictor = TabularPredictor(
        label=ds["target_column"],
        path=str(output_path),
        problem_type=problem_type,
        eval_metric=eval_metric,
    ).fit(
        train_data=train_clean,
        tuning_data=val_clean,
        time_limit=time_limit,
        presets=presets,
    )
    elapsed = round(time.time() - start, 2)
    return predictor, elapsed


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
    final_model_dir = out_dir / "autogluon_model_final"
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
        "n_val": None,
        "n_test_input": None,
        "n_test_after_replay": None,
        "best_model": None,
        "training_time_seconds": None,
        "model_path": str(final_model_dir),
        "final_answer": None,
        "error": None,
    }

    # Snapshot AUTOGLUON_TIME_LIMIT so we can both override it for the graph's
    # throwaway training and restore it before training the real model. The
    # finally block guarantees we never leak the override into a later dataset
    # or the parent shell.
    original_time_limit = os.environ.get("AUTOGLUON_TIME_LIMIT")

    def _restore_time_limit() -> None:
        if original_time_limit is None:
            os.environ.pop("AUTOGLUON_TIME_LIMIT", None)
        else:
            os.environ["AUTOGLUON_TIME_LIMIT"] = original_time_limit

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

        # 3. Three-way stratified 80 / 10 / 10 split
        train_df, val_df, test_df = three_way_split(
            full_df, target_col, task_type, seed=42
        )
        result_row["n_train"] = len(train_df)
        result_row["n_val"] = len(val_df)
        result_row["n_test_input"] = len(test_df)
        print(
            f"[{name}] Split: {len(train_df)} train / "
            f"{len(val_df)} val / {len(test_df)} test"
        )

        # 4. Write the train fold CSV (the graph entry point loads via
        #    pd.read_csv). Val and test folds stay in memory.
        train_df.to_csv(train_csv, index=False)

        # 5. Set AUTO_TARGET_COLUMN to bypass the HITL prompt
        os.environ["AUTO_TARGET_COLUMN"] = target_col

        # 6. Override AUTOGLUON_TIME_LIMIT for the graph's throwaway training.
        #    plugins/modeller.py reads this env var inside train_model() each
        #    call, so the override applies to the very next graph invocation.
        os.environ["AUTOGLUON_TIME_LIMIT"] = GRAPH_THROWAWAY_TIME_LIMIT

        # 7. Build initial state identical to scripts/run_graph.py
        initial_state = {
            "user_query": user_query,
            "working_df": pd.read_csv(train_csv),
            "retry_count": 0,
            "tool_call_count": 0,
            "pass_count": 0,
            "target_column": None,
        }

        # 8. Invoke the graph (cleans + trains a throwaway AutoGluon model).
        #    We keep clean_df + applied_steps; everything else is discarded.
        print(f"[{name}] Invoking AutoDS graph (throwaway training, "
              f"AUTOGLUON_TIME_LIMIT={GRAPH_THROWAWAY_TIME_LIMIT}s)...")
        try:
            graph_result = app.invoke(initial_state)
        except Exception as e:
            traceback.print_exc()
            result_row["status"] = "graph_failed"
            result_row["error"] = f"{type(e).__name__}: {e}"
            return result_row

        # The graph's answer_agent ran at the end of app.invoke, but its answer
        # is based on the 30-second throwaway model's metadata. We deliberately
        # discard it and regenerate the answer below after train_final_model,
        # so the stored final_answer describes the real benchmark model.

        # Capture investigation_findings so we can feed them back into the
        # answer agent for the second invocation (they drive the "caveats"
        # section of the final answer).
        investigation_findings = graph_result.get("investigation_findings")

        clean_train = graph_result.get("clean_df")
        if clean_train is None:
            raise RuntimeError("Graph did not produce a cleaned train DataFrame")

        # Pull the full list of successfully-applied cleaning steps from state.
        # This accumulates across ALL passes (via the operator.add reducer),
        # unlike current_plan which only holds the latest pass's recipe — see
        # replay_steps() docstring for why that distinction matters.
        applied_steps = graph_result.get("applied_steps") or []
        if not applied_steps:
            raise RuntimeError(
                "Graph did not produce any applied cleaning steps "
                "(state['applied_steps'] is empty)"
            )

        # 9. Restore the user's AUTOGLUON_TIME_LIMIT before the *real* training
        _restore_time_limit()

        # 10. Replay the full cleaning sequence on the val and test folds.
        #     Cleaning has no fitted state so straight replay is
        #     methodologically sound.
        try:
            val_clean = replay_steps(applied_steps, val_df, target_col)
            test_clean = replay_steps(applied_steps, test_df, target_col)
        except RecipeReplayError as e:
            print(f"[{name}] Recipe replay FAILED: {e}")
            result_row["status"] = "replay_failed"
            result_row["error"] = str(e)
            return result_row

        result_row["n_test_after_replay"] = len(test_clean)
        print(
            f"[{name}] After replay: {len(val_clean)} val / "
            f"{len(test_clean)} test"
        )

        # Sanity: cleaned val/test must have the same columns as clean_train.
        # If this ever fires, the recipe replay has diverged from the graph's
        # own cleaning — fix the graph or the applied_steps accumulator, not
        # this script. Catching it here turns a confusing AutoGluon traceback
        # into a one-line bug report.
        train_cols = set(clean_train.columns)
        for split_name, split_df in (("val", val_clean), ("test", test_clean)):
            extra = set(split_df.columns) - train_cols
            missing = train_cols - set(split_df.columns)
            if extra or missing:
                raise RuntimeError(
                    f"[{name}] Replayed {split_name} columns differ from "
                    f"clean_train (extra={sorted(extra)}, "
                    f"missing={sorted(missing)}). This means the recipe "
                    f"replay diverged from the graph's cleaning."
                )

        # 11. Train the *real* AutoGluon model. Explicit tuning_data means
        #     model selection / ensemble weighting never see the test fold.
        time_budget = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "300"))
        print(
            f"[{name}] Training final AutoGluon model "
            f"(AUTOGLUON_TIME_LIMIT={time_budget}s, explicit tuning_data)..."
        )
        try:
            predictor, training_time = train_final_model(
                clean_train, val_clean, ds, final_model_dir
            )
        except Exception as e:
            traceback.print_exc()
            result_row["status"] = "training_failed"
            result_row["error"] = f"{type(e).__name__}: {e}"
            return result_row

        result_row["training_time_seconds"] = training_time

        leaderboard = predictor.leaderboard(silent=True)
        if not leaderboard.empty:
            top_row = leaderboard.iloc[0]
            result_row["autogluon_val_score"] = float(top_row["score_val"])
            result_row["best_model"] = str(top_row["model"])

        # 11b. Re-run the answer agent against the real model's metadata so
        #      result_row["final_answer"] narrates the benchmark model, not the
        #      30-second throwaway that the graph's own answer_agent saw.
        real_metadata = build_model_metadata(
            predictor=predictor,
            df=clean_train,
            elapsed=training_time,
            save_path=str(final_model_dir),
        )
        synthetic_state = {
            "user_query": user_query,
            "model_metadata": real_metadata,
            "investigation_findings": investigation_findings,
        }
        try:
            final_answer = run_answer_agent(synthetic_state)
        except Exception as e:
            # Don't fail the benchmark over an answer-agent hiccup — the real
            # test score is still the headline number. Log and continue.
            print(f"[{name}] WARNING: answer agent failed on real-model metadata: {e}")
            final_answer = None

        result_row["final_answer"] = final_answer
        if final_answer:
            (out_dir / "final_answer.txt").write_text(final_answer)
            print(f"\n[{name}] --- Final Answer ---")
            print(final_answer)
            print(f"[{name}] --- End Final Answer ---\n")
        else:
            print(f"[{name}] WARNING: no final_answer produced for real model")

        # 12. Score on the held-out test set with the YAML-specified metric.
        y_test = test_clean[target_col]
        X_test = test_clean.drop(columns=[target_col])
        score = score_with_metric(predictor, X_test, y_test, metric, task_type)
        result_row["autods_test_score"] = score
        if result_row["literature_baseline"] is not None:
            result_row["delta_vs_literature"] = (
                score - float(result_row["literature_baseline"])
            )
        print(
            f"[{name}] Test {metric} = {score:.4f} "
            f"(val score_val = {result_row['autogluon_val_score']})"
        )

        return result_row

    except Exception as e:
        traceback.print_exc()
        result_row["status"] = (
            "graph_failed" if result_row["status"] == "ok" else result_row["status"]
        )
        result_row["error"] = f"{type(e).__name__}: {e}"
        return result_row
    finally:
        # Hygiene: never let AUTO_TARGET_COLUMN or our AUTOGLUON_TIME_LIMIT
        # override bleed across datasets or into the parent shell. This runs
        # even if we crashed between the override and the inline restore above.
        os.environ.pop("AUTO_TARGET_COLUMN", None)
        _restore_time_limit()


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
    headers = [
        "Dataset",
        "Metric",
        "AutoDS (test)",
        "AutoGluon (val)",
        "Literature",
        "Delta vs lit",
        "Status",
    ]
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
                _fmt_score(r.get("autogluon_val_score")),
                _fmt_score(r.get("literature_baseline")),
                _fmt_delta(r.get("delta_vs_literature")),
                str(r["status"]),
            ])
            + " |"
        )
    print(
        "\nAutoDS (test) is scored on a held-out 10% fold never seen by "
        "training, model selection, or ensemble weighting. AutoGluon (val) is "
        "the same model's score on the validation fold used for selection — "
        "it's optimistic by construction. Literature baselines use different "
        "splits and sometimes different problem definitions; treat deltas as "
        "ballpark, not pass/fail."
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

    # Echo each dataset's final answer at the end so all responses are visible
    # in one place after the summary table.
    for r in results:
        answer = r.get("final_answer")
        if not answer:
            continue
        print("\n" + "=" * 70)
        print(f"FINAL ANSWER - {r['name']}")
        print("=" * 70)
        print(answer)


if __name__ == "__main__":
    main()
