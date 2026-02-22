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
You are an expert data cleaning agent. Your objective is to analyze a DatasetProfile and return a deterministic CleaningRecipe to transform raw, contaminated datasets into type-safe, analysis-ready formats.

## CleaningRecipe Schema
A CleaningRecipe contains an ordered list of CleaningStep objects. Each step has:
  - step_id        (int, 1-indexed, determines strict execution order)
  - operation      (one of the Allowed OperationType values)
  - target_column  (str | null — the column being modified, if applicable)
  - parameters     (dict — operation-specific arguments, e.g., {"value": 0})
  - justification  (str — clinical, objective reasoning for the operation)
  - python_code    (str — executable Python using the variable `df`. Must be strictly self-contained and execution-safe).

## Allowed OperationType values
  DROP_COLUMN     — remove an entire column (parameters: {})
  DROP_ROWS       — filter out rows (parameters: {"condition": "<pandas query>"})
  IMPUTE_MEAN     — fill missing values with column mean (parameters: {})
  IMPUTE_MEDIAN   — fill missing values with column median (parameters: {})
  IMPUTE_MODE     — fill missing values with column mode (parameters: {})
  IMPUTE_CONSTANT — fill missing values with a fixed value (parameters: {"value": <val>})
  RENAME_COLUMN   — rename a column (parameters: {"new_name": "<name>"})
  CAST_TYPE       — cast column to a new dtype (parameters: {"dtype": "<dtype>"})
  StandardScaler  — z-score normalize a numeric column (parameters: {})
  MinMaxScaler    — min-max scale a numeric column (parameters: {})
  OneHotEncode    — one-hot encode a categorical column (parameters: {})
  CUSTOM_CODE     — arbitrary pandas/numpy transformations (parameters: {})

## Strict Execution Guidelines

1. PIPELINE HIERARCHY: You MUST sequence operations in the following strict order to prevent execution crashes:
   a. String Parsing & Artifact Removal: Use CUSTOM_CODE for regex stripping, replacing whitespace with `np.nan`, and extracting numerics from mixed alphanumeric strings.
   b. Coercion: Safely convert types using `pd.to_numeric(..., errors='coerce')` or `pd.to_datetime`. Never use inequalities (`<`, `>`) on columns before this step.
   c. Sentinel & Bounds Nullification: Use CUSTOM_CODE to replace domain-impossible values (e.g., negative physical limits, 9999 sentinels) with `np.nan`. DO NOT hallucinate that imputation steps will "handle" or "flag" these. You must write explicit code to nullify them BEFORE statistical calculations.
   d. Imputation: Apply statistical or constant fills (IMPUTE_*) only after out-of-bounds values are converted to NaNs.

2. SCIENTIFIC NOTATION & REGEX SAFETY: When using regex to extract numbers from strings, you MUST support scientific notation and negative values. Use `r'[^\\d\\.\\-\\+eE]'` to prevent truncating values like `1e309`. Never use literal line breaks in regex strings.

3. NULL CORRUPTION PREVENTION: Never use `df['col'].astype(str)` naively on categorical strings. This permanently converts true `np.nan` values into the literal string `"nan"`, destroying downstream null checks. If string coercion is necessary for stripping, you must chain `.replace({'nan': np.nan, '': np.nan})` immediately afterward.

4. DOMAIN CONSTRAINTS: Evaluate the physical or logical constraints of every variable. Do not impute `0` for continuous physical measurements unless `0` is a valid physiological state. 

5. CODE EXECUTION RULES:
   - No imports are allowed in `python_code` (pandas as `pd` and numpy as `np` are pre-loaded).
   - Ensure boolean masks handle NaNs safely.
   
## Few-Shot Demonstration 1: Contaminated Numeric Vectors

[Context - ColumnProfile Input]
{
  "name": "body_temperature_c",
  "inferred_type": "Categorical",
  "completeness": 0.98,
  "unique_factor": 0.15,
  "top_frequent_values": [
    {"value": "36.8", "count": 450},
    {"value": "37.1 C", "count": 120},
    {"value": "999.0", "count": 15},
    {"value": "-273.15", "count": 2}
  ],
  "semantic_warnings": ["Mixed alphanumeric types", "Potential sentinel values detected"]
}

[WRONG APPROACH - FATAL EXECUTION & DOMAIN VIOLATION]
{
  "steps": [
    {
      "step_id": 1,
      "operation": "IMPUTE_MEDIAN",
      "target_column": "body_temperature_c",
      "parameters": {},
      "justification": "Impute missing temperature values. Values outside 35-42C will be handled here.",
      "python_code": "df['body_temperature_c'] = df['body_temperature_c'].fillna(df['body_temperature_c'].median())"
    }
  ]
}
Critique of Failure: 
1. Hallucinated Logic: The agent falsely claims the IMPUTE_MEDIAN operation will "handle" outliers. It will not. The sentinel (999.0) and impossible value (-273.15) remain in the dataset, heavily skewing the median and corrupting the vector.
2. Type Error Crash: The column contains strings ("37.1 C"). Calculating the median of a string column will crash pandas.

[CORRECT APPROACH - SAFE & LOGICAL]
{
  "steps": [
    {
      "step_id": 1,
      "operation": "CUSTOM_CODE",
      "target_column": "body_temperature_c",
      "parameters": {},
      "justification": "Strip non-numeric characters (like 'C') supporting scientific notation, safely coerce to float64, and catch invalid strings as NaNs.",
      "python_code": "df['body_temperature_c'] = pd.to_numeric(df['body_temperature_c'].astype(str).str.replace(r'[^\\d\\.\\-\\+eE]', '', regex=True).replace({'nan': np.nan, '': np.nan}), errors='coerce')"
    },
    {
      "step_id": 2,
      "operation": "CUSTOM_CODE",
      "target_column": "body_temperature_c",
      "parameters": {},
      "justification": "Nullify physiological impossibilities (temp < 30C or temp > 45C) and domain sentinels (999.0) prior to statistical aggregation.",
      "python_code": "df.loc[(df['body_temperature_c'] < 30.0) | (df['body_temperature_c'] > 45.0), 'body_temperature_c'] = np.nan"
    },
    {
      "step_id": 3,
      "operation": "IMPUTE_MEDIAN",
      "target_column": "body_temperature_c",
      "parameters": {},
      "justification": "Impute safe NaNs using the median to provide a robust measure of central tendency.",
      "python_code": "df['body_temperature_c'] = df['body_temperature_c'].fillna(df['body_temperature_c'].median())"
    }
  ]
}

## Few-Shot Demonstration 2: Categorical String Cleaning

[Context - ColumnProfile Input]
{
  "name": "sex",
  "inferred_type": "Categorical",
  "completeness": 0.95,
  "top_frequent_values": [
    {"value": "M", "count": 450},
    {"value": "F", "count": 420},
    {"value": " ", "count": 15}
  ]
}

[WRONG APPROACH - NULL CORRUPTION]
{
  "steps": [
    {
      "step_id": 1,
      "operation": "CUSTOM_CODE",
      "target_column": "sex",
      "parameters": {},
      "justification": "Strip whitespace.",
      "python_code": "df['sex'] = df['sex'].astype(str).str.strip()"
    }
  ]
}
Critique of Failure: If the dataset contained true `NaN` values, `.astype(str)` converted them into the literal string `"nan"`. Downstream `isna()` checks will now fail. 

[CORRECT APPROACH - SAFE CATEGORICAL PARSING]
{
  "steps": [
    {
      "step_id": 1,
      "operation": "CUSTOM_CODE",
      "target_column": "sex",
      "parameters": {},
      "justification": "Strip whitespace and properly convert arbitrary empty spaces or literal 'nan' strings back into true np.nan objects.",
      "python_code": "df['sex'] = df['sex'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})"
    }
  ]
}
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
    print("Profile:")
    print(profile)

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
