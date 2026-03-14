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

from core.graph import app  # noqa: E402
from core.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = (
    REPO_ROOT / "data" / "sample_data" / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)
OUTPUT_DIR = REPO_ROOT / "data" / "output"
OUTPUT_CSV = OUTPUT_DIR / "cleaned_data.csv"

USER_QUERY = "What patterns in patient visits predict high-cost outcomes?"


def main() -> None:
    setup_logger()

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
        "is_data_clean": False,
        "target_column": None,
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
        print(f"  Data quality score  : {findings_data.get('data_quality_score')}")
        print(f"  Columns to drop     : {findings_data.get('columns_to_drop', [])}")
        for col, reason in findings_data.get("columns_to_drop_rationale", {}).items():
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
            print(f"    Pass {ph['pass_number']}: quality={ph['quality_score']}, "
                  f"violations={ph['violations_found']}, "
                  f"steps={ph['steps_executed']}, "
                  f"rows_after={ph['rows_after']}")
    is_clean = result.get("is_data_clean", False)
    print(f"  Data declared clean : {is_clean}")

    # Clean DataFrame
    clean_df = result.get("clean_df")
    if clean_df is not None:
        print(f"\n  Clean DataFrame     : {len(clean_df)} rows x {len(clean_df.columns)} columns")

        # Save to CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved to            : {OUTPUT_CSV}")
    else:
        print("\n  Clean DataFrame     : None (pipeline may have failed)")

    # Final answer
    print(f"\n  Final answer        : {result.get('final_answer', 'N/A')}")

    # Error state
    if result.get("latest_error"):
        print(f"\n  LATEST ERROR: {result['latest_error']}")
        print(f"  Retry count: {result.get('retry_count', 0)}")

    print()


if __name__ == "__main__":
    main()
