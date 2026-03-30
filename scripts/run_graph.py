"""
Run the full AutoDS LangGraph pipeline end-to-end.

Loads a dataset, invokes the compiled graph, and prints results.
Saves the cleaned DataFrame to CSV for manual inspection.

Configuration (environment variables):
  OPENAI_API_KEY     OpenAI API key (required)
  INVESTIGATOR_MODEL Model for the investigator agent (default: gpt-5-2025-08-07)
  CODEGEN_MODEL      Model for the code generator agent (default: gpt-4.1-2025-04-14)

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/run_graph.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so imports work when run from any CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.pipeline.graph import app  # noqa: E402
from core.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = (
    REPO_ROOT / "data" / "sample_data" / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "cleaned_data.csv"
OUTPUT_VERBOSE_LOG = OUTPUT_DIR / "verbose.log"

# AutoDS is designed for supervised ML questions — regression, binary/multiclass
# classification — where the user wants to understand, explain, or predict a
# target variable. Examples:
#   "What drives high hospital bills?"              (regression)
#   "Which patients will be readmitted?"             (binary classification)
#   "What determines a patient's risk category?"     (multiclass classification)
# Descriptive/aggregation queries (e.g., "What's the average bill?") are out of
# scope and better served by SQL or BI tools.
USER_QUERY = "What patterns in patient visits predict high-cost outcomes?"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(OUTPUT_VERBOSE_LOG))

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows x {len(df.columns)} columns.\n")

    # ------------------------------------------------------------------
    # 2. Run the graph
    # ------------------------------------------------------------------
    initial_state = {
        "user_query": USER_QUERY,
        "working_df": df,
        "retry_count": 0,
        "tool_call_count": 0,
        "pass_count": 0,
        "target_column": None,
        "assessor_tool_call_count": 0,
        "residual_issues": None,
    }

    print("=" * 70)
    print("RUNNING LANGGRAPH PIPELINE")
    print("=" * 70)
    result = app.invoke(initial_state)

    # ------------------------------------------------------------------
    # 3. Display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    # Investigation findings
    findings = result.get("investigation_findings")
    if findings:
        findings_data = (
            findings.model_dump() if hasattr(findings, "model_dump") else findings
        )
        print(f"\n  Target column       : {findings_data.get('target_column')}")
        print(f"  Target rationale    : {findings_data.get('target_column_rationale')}")
        print(f"  Task type           : {findings_data.get('task_type')}")
        print(f"  Columns to drop     : {findings_data.get('columns_to_drop', [])}")
        for entry in findings_data.get("columns_to_drop_rationale", []):
            col = entry["column"] if isinstance(entry, dict) else entry.column
            reason = entry["reason"] if isinstance(entry, dict) else entry.reason
            print(f"    {col}: {reason}")

        violations = findings_data.get("violations", [])
        print(f"\n  Violations ({len(violations)}):")
        for v in violations:
            print(f"\n    [{v['violation_id']}] {v['severity']} — {v['category']}")
            print(f"      Columns  : {', '.join(v['affected_columns'])}")
            print(f"      Issue    : {v['description']}")
            print(f"      Evidence : {v['evidence']}")
            print(f"      Fix      : {v['suggested_action']}")

        caveats = findings_data.get("key_caveats", [])
        if caveats:
            print(f"\n  Key Caveats:")
            for c in caveats:
                print(f"    - {c}")

    # Cleaning recipe
    plan = result.get("current_plan")
    if plan:
        plan_data = plan.model_dump() if hasattr(plan, "model_dump") else plan
        steps = plan_data.get("steps", [])
        print(f"\n  Cleaning steps      : {len(steps)}")
        for step in steps:
            print(f"    Step {step['step_id']}: [{step['operation']}] {step.get('target_column', '')}")
            print(f"      Justification : {step['justification']}")
            print(f"      Code          : {step['python_code']}")

    # Cleaning history
    history = result.get("cleaning_history", [])
    if history:
        succeeded = sum(1 for h in history if (h.status if hasattr(h, "status") else h.get("status")) == "SUCCESS")
        failed = sum(1 for h in history if (h.status if hasattr(h, "status") else h.get("status")) == "FAILED")
        print(f"\n  Sandbox execution   : {succeeded} steps succeeded, {failed} failed")

    # Multi-pass summary
    pass_count = result.get("pass_count", 0)
    print(f"\n  Cleaning passes     : {pass_count}")
    pass_history = result.get("pass_history", [])
    if pass_history:
        for ph in pass_history:
            print(f"    Pass {ph['pass_number']}: "
                  f"violations={ph['violations_found']}, "
                  f"steps={ph['steps_executed']}, "
                  f"rows_after={ph['rows_after']}")

    # Quality assessment summary
    qa = result.get("quality_assessment") or {}
    structural = qa.get("structural_score")
    llm = qa.get("llm_assessment") or {}
    llm_score = llm.get("score") if isinstance(llm, dict) else None
    recommendation = qa.get("recommendation", "N/A")
    struct_str = f"{structural:.2f}" if structural is not None else "N/A"
    llm_str = f"{llm_score:.2f}" if llm_score is not None else "N/A"
    print(f"  Structural score    : {struct_str}")
    print(f"  LLM quality score   : {llm_str}")
    print(f"  Recommendation      : {recommendation}")

    # Clean DataFrame
    clean_df = result.get("clean_df")
    if clean_df is not None:
        print(f"\n  Clean DataFrame     : {len(clean_df)} rows x {len(clean_df.columns)} columns")

        # Save to CSV
        clean_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved to            : {OUTPUT_CSV}")
    else:
        print("\n  Clean DataFrame     : None (pipeline may have failed)")

    # Model results
    model_meta = result.get("model_metadata", {})
    if model_meta and not model_meta.get("error"):
        print(f"\n  --- Model Results ---")
        print(f"  Problem type        : {model_meta.get('problem_type')}")
        print(f"  Best model          : {model_meta.get('best_model')}")
        print(f"  Training time       : {model_meta.get('training_time_seconds')}s")
        print(f"  Model saved to      : {model_meta.get('model_path')}")

        eval_metrics = model_meta.get("eval_metrics", {})
        if eval_metrics:
            print(f"  Evaluation metrics  :")
            for metric, value in eval_metrics.items():
                print(f"    {metric}: {value}")

        feat_imp = model_meta.get("feature_importance", {})
        if feat_imp:
            sorted_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"  Top features        :")
            for name, imp in sorted_feats[:10]:
                print(f"    {name}: {imp:.4f}")

        leaderboard = model_meta.get("leaderboard", [])
        if leaderboard:
            print(f"  Leaderboard (top {len(leaderboard)}):")
            for entry in leaderboard:
                print(f"    {entry.get('model', '?')}: score={entry.get('score_val', '?')}")
    elif model_meta and model_meta.get("error"):
        print(f"\n  Model training FAILED: {model_meta['error']}")

    # Final answer
    print(f"\n  Final answer        : {result.get('final_answer', 'N/A')}")

    # Error state
    if result.get("latest_error"):
        print(f"\n  LATEST ERROR: {result['latest_error']}")
        print(f"  Retry count: {result.get('retry_count', 0)}")

    print()


if __name__ == "__main__":
    main()
