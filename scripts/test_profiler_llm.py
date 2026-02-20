"""
Test script: Profiler → LLM → CleaningRecipe

Runs the profiler on data/sample_data/dirty_healthcare_visits.csv, then passes
the resulting DatasetProfile to a locally-hosted LLM (via vLLM's OpenAI-compatible
API) and asks it to return a CleaningRecipe conforming to core/schemas.py.

Configuration (environment variables):
  VLLM_BASE_URL  Base URL of the vLLM server  (default: http://localhost:8000/v1)
  VLLM_MODEL     Model name served by vLLM    (default: Qwen/Qwen3-4B-Instruct-2507)

Usage:
  # Start your vLLM server first:
  #   vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
  python scripts/test_profiler_llm.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so imports work when run from any CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.schemas import CleaningRecipe, DatasetProfile  # noqa: E402
from plugins.profiler import generate_profile  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DATA_PATH = REPO_ROOT / "data" / "sample_data" / "healthcare" / "dirty_healthcare_visits_no_notes.csv"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert data cleaning agent. Your job is to analyse a DatasetProfile \
and return a CleaningRecipe that will transform the raw dataset into a clean, \
analysis-ready form.

## CleaningRecipe schema
A CleaningRecipe contains an ordered list of CleaningStep objects. Each step has:
  - step_id        (int, 1-indexed, determines execution order)
  - operation      (one of the allowed OperationType values listed below)
  - target_column  (str | null — the column being modified, if applicable)
  - parameters     (dict — operation-specific arguments, e.g. {"value": 0})
  - justification  (str — concise reason WHY this step is needed)
  - python_code    (str — executable Python using the variable `df`, e.g. \
`df['age'] = df['age'].clip(lower=0)`)

## Allowed OperationType values
  DROP_COLUMN    — remove an entire column (parameters: {})
  DROP_ROWS      — filter out rows matching a condition (parameters: {"condition": "<pandas query>"})
  IMPUTE_MEAN    — fill missing values with column mean (parameters: {})
  IMPUTE_MEDIAN  — fill missing values with column median (parameters: {})
  IMPUTE_MODE    — fill missing values with column mode (parameters: {})
  IMPUTE_CONSTANT — fill missing values with a fixed value (parameters: {"value": <val>})
  RENAME_COLUMN  — rename a column (parameters: {"new_name": "<name>"})
  CAST_TYPE      — cast column to a new dtype (parameters: {"dtype": "<dtype>"})
  StandardScaler — z-score normalise a numeric column (parameters: {})
  MinMaxScaler   — min-max scale a numeric column (parameters: {})
  OneHotEncode   — one-hot encode a categorical column (parameters: {})
  CUSTOM_CODE    — any other transformation (parameters: {})

## Guidelines
- Address every data quality issue you can detect in the profile.
- Keep python_code self-contained and safe (no imports; only use `df` and \
built-in pandas/numpy methods already available on `df`).
- Order steps so that destructive operations (DROP_*) come before imputation \
or scaling.
- Use CUSTOM_CODE sparingly — prefer the named operations when they apply.
"""


def build_human_message(profile: DatasetProfile) -> str:
    profile_json = json.dumps(profile.model_dump(), indent=2, default=str)
    return (
        f"Here is the DatasetProfile for the dataset:\n\n"
        f"```json\n{profile_json}\n```\n\n"
        "Return a CleaningRecipe that addresses all data quality issues you can "
        "identify in this profile."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Load data
    print(f"Loading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows x {len(df.columns)} columns.\n")

    # 2. Run profiler
    print("Running profiler...")
    profile_dict = generate_profile(df, detailed_profiler=False)
    profile = DatasetProfile.model_validate(profile_dict)
    print(f"  Profile complete: {profile.row_count} rows, {len(profile.columns)} columns profiled.\n")

    # 3. Build LLM client
    print(f"Connecting to vLLM at: {VLLM_BASE_URL}")
    print(f"  Model: {VLLM_MODEL}\n")
    llm = ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="no-key",  # vLLM local server does not require a real key
        model=VLLM_MODEL,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(CleaningRecipe)

    # 4. Call the LLM
    print("Calling LLM for CleaningRecipe...")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_human_message(profile)),
    ]
    recipe: CleaningRecipe = structured_llm.invoke(messages)
    print(f"  Received {len(recipe.steps)} cleaning step(s).\n")

    # 5. Display results
    print("=" * 70)
    print("CLEANING RECIPE")
    print("=" * 70)
    for step in recipe.steps:
        print(f"\nStep {step.step_id}: [{step.operation}]"
              + (f" → {step.target_column}" if step.target_column else ""))
        print(f"  Justification : {step.justification}")
        if step.parameters:
            print(f"  Parameters    : {step.parameters}")
        print(f"  Python code   : {step.python_code}")
    print("\n" + "=" * 70)

    # 6. Dump full recipe as JSON for downstream use
    print("\nFull CleaningRecipe JSON:")
    print(recipe.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
