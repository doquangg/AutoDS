"""
Test script: Profiler → Investigator Agent → Code Generator Agent

Runs the full two-agent pipeline WITHOUT LangGraph, to test the core logic
in isolation:

  1. Profile the dataset (pure computation, no LLM)
  2. Investigator Agent examines the profile, calls tools to inspect the data,
     and produces InvestigationFindings (structured diagnosis — no code)
  3. Code Generator Agent reads the findings and writes a CleaningRecipe
     (executable Python cleaning steps)

This script manually implements the investigator's tool loop that LangGraph
would normally handle via ToolNode. This makes it possible to test and debug
the agents without spinning up the full state machine.

Configuration (environment variables):
  OPENAI_API_KEY     OpenAI API key (required)
  INVESTIGATOR_MODEL Model for the investigator agent (default: gpt-5-2025-08-07)
  CODEGEN_MODEL      Model for the code generator agent (default: gpt-4.1-2025-04-14)

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/test_profiler_llm.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path so imports work when run from any CWD
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.schemas import (  # noqa: E402
    CleaningRecipe,
    DatasetProfile,
    InvestigationFindings,
)
from plugins.profiler import generate_profile  # noqa: E402
from core.tools import (  # noqa: E402
    investigation_tools,
    set_working_df,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INVESTIGATOR_MODEL = os.environ.get("INVESTIGATOR_MODEL", "gpt-5-2025-08-07")
CODEGEN_MODEL = os.environ.get("CODEGEN_MODEL", "gpt-4.1-2025-04-14")
DATA_PATH = (
    REPO_ROOT / "data" / "sample_data" / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)

MAX_TOOL_CALLS = 30  # Safety cap on investigator tool loop


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: Investigator
# ═══════════════════════════════════════════════════════════════════════════

# FIXME (#19):
# Prompts shouldn't live here. Also, consider restructuring these prompts;
# they were vibecoded. Update script to read prompt from some directory.
INVESTIGATOR_SYSTEM_PROMPT = """\
You are a senior data quality analyst. Your job is to examine a dataset profile \
and identify every semantic data quality issue that could corrupt a machine \
learning model.

You will receive:
1. The user's question (what they want to predict/answer)
2. A statistical profile of every column in the dataset

Your task:
- Identify the TARGET COLUMN that best answers the user's question
- Find ALL semantic violations in the data
- Classify each violation by severity and category
- Provide specific evidence from the profile for each finding
- Suggest plain-English fixes (NOT code — just intent)
- Assess overall data quality (0.0 to 1.0)
- Note any caveats that should accompany the final answer

WHAT TO LOOK FOR:
- Sentinel values masquerading as real data (-1, 0, 999, 9999, "N/A", "Unknown")
  Check: top_frequent_values, min_value, max_value
- Temporal impossibilities (event B before event A)
  Check: use temporal_ordering_check tool
- Cross-column logic errors (impossible combinations)
  Check: use cross_column_frequency tool
- Type mismatches (numeric column storing categories, or vice versa)
  Check: inferred_type vs semantic meaning
- Suspicious distributions (spikes at sentinel values, extreme skew)
  Check: use value_distribution tool
- Systematic missingness (columns null together)
  Check: use null_co_occurrence tool
- High cardinality columns that are likely IDs (should be dropped)
  Check: unique_factor > 0.95
- PII that should not be used as features

USE YOUR TOOLS to verify suspicions. Don't guess — inspect the actual data.
But be efficient: don't call the same tool repeatedly with minor variations.

Do NOT write any Python code. Your job is diagnosis, not treatment.\
"""


def _build_tool_lookup() -> dict:
    """Build a name → callable mapping for manual tool execution."""
    return {t.name: t for t in investigation_tools}


def _execute_tool_calls(tool_calls: list, tool_lookup: dict) -> list[ToolMessage]:
    """
    Manually execute tool calls and return ToolMessages.
    This replicates what LangGraph's ToolNode does automatically.
    """
    results = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        print(f"      🔧 Calling tool: {tool_name}({tool_args})")

        if tool_name not in tool_lookup:
            output = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                output = tool_lookup[tool_name].invoke(tool_args)
            except Exception as e:
                output = f"Error executing {tool_name}: {type(e).__name__}: {e}"

        # Truncate very long tool outputs to stay within context limits
        output_str = str(output)
        if len(output_str) > 3000:
            output_str = output_str[:3000] + "\n... [truncated]"

        print(f"      ↳ Result: {output_str[:200]}{'...' if len(output_str) > 200 else ''}")
        results.append(ToolMessage(content=output_str, tool_call_id=tool_id))

    return results



def run_investigator(
    llm: ChatOpenAI,
    profile: DatasetProfile,
    user_query: str,
) -> InvestigationFindings:
    """
    Run the investigator agent with a manual tool loop.

    This replicates the LangGraph flow:
        investigator → [tool_calls?] → tools → investigator → ... → findings

    Returns the structured InvestigationFindings.
    """
    # Bind tools so the LLM can request them
    llm_with_tools = llm.bind_tools(investigation_tools)
    tool_lookup = _build_tool_lookup()

    profile_json = json.dumps(profile.model_dump(), indent=2, default=str)

    messages = [
        SystemMessage(content=INVESTIGATOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"USER QUERY: {user_query}\n\n"
            f"DATASET PROFILE ({profile.row_count} rows):\n"
            f"{profile_json}"
        )),
    ]

    tool_call_count = 0

    while True:
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            # Agent is done investigating — extract structured findings
            print("    ✅ Investigator finished (no more tool calls).")
            structured_llm = llm.with_structured_output(InvestigationFindings, method="function_calling")
            return structured_llm.invoke(messages)

        # Execute tool calls and feed results back
        tool_call_count += len(tool_calls)
        print(f"    Tool calls this turn: {len(tool_calls)} "
              f"(total: {tool_call_count}/{MAX_TOOL_CALLS})")

        tool_results = _execute_tool_calls(tool_calls, tool_lookup)
        messages.extend(tool_results)

        # Safety cap
        if tool_call_count >= MAX_TOOL_CALLS:
            print(f"    ⚠️  Tool call limit reached ({MAX_TOOL_CALLS}). "
                  f"Forcing investigator to finalize.")
            structured_llm = llm.with_structured_output(InvestigationFindings, method="function_calling")
            return structured_llm.invoke(messages)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: Code Generator
# ═══════════════════════════════════════════════════════════════════════════

# FIXME (#19):
# Prompts shouldn't live here. Also, consider restructuring these prompts;
# they were vibecoded. Update script to read prompt from some directory.
CODEGEN_SYSTEM_PROMPT = """\
You are an expert Python data engineer. Your job is to write a CleaningRecipe: \
an ordered list of executable pandas code steps that clean a DataFrame.

You will receive:
1. Investigation findings describing exactly what's wrong with the data
2. The dataset profile (column types, shapes, value distributions)

RULES:
- Every step must contain valid, executable Python using the variable `df`
- Each step should be atomic: one clear transformation per step
- Steps execute in order. Each step receives the `df` from the previous step.
- Reference the violation_id from findings in your addresses_violation field
- Use the operation categories from OperationType (DROP_COLUMN, DROP_ROWS, etc.)
- Use CUSTOM_CODE for anything that doesn't fit a predefined operation
- Do NOT use StandardScaler or MinMaxScaler — AutoGluon handles scaling internally
- Preserve the target column — never drop or corrupt it
- Be conservative: prefer imputation over dropping rows when possible

PIPELINE HIERARCHY — sequence operations in this strict order:
  a. String Parsing & Artifact Removal (regex stripping, whitespace → NaN)
  b. Type Coercion (pd.to_numeric, pd.to_datetime with errors='coerce')
  c. Sentinel & Bounds Nullification (replace impossible values with NaN)
  d. Imputation (statistical fills ONLY after sentinels are nullified)
  e. Column drops and renames

CODE STYLE:
- Always reassign: `df = df[df['age'] > 0]` not `df.drop(..., inplace=True)`
- Handle edge cases: check column exists before operating on it
- Use .copy() when creating derived columns from slices
- String operations: use .str accessor, handle NaN with na=False
- Never use `df['col'].astype(str)` without chaining `.replace({{'nan': np.nan, '': np.nan}})`
- No imports in python_code — pd and np are pre-loaded

You MUST respond with a valid CleaningRecipe JSON object.\
"""


def run_code_generator(
    llm: ChatOpenAI,
    findings: InvestigationFindings,
    profile: DatasetProfile,
    user_query: str,
) -> CleaningRecipe:
    """
    Run the code generator agent. Takes investigation findings as input,
    returns a CleaningRecipe with executable Python steps.
    """
    structured_llm = llm.with_structured_output(CleaningRecipe, method="function_calling")

    findings_json = json.dumps(findings.model_dump(), indent=2, default=str)
    profile_json = json.dumps(profile.model_dump(), indent=2, default=str)

    messages = [
        SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"USER QUERY: {user_query}\n\n"
            f"INVESTIGATION FINDINGS:\n{findings_json}\n\n"
            f"DATASET PROFILE:\n{profile_json}\n\n"
            f"Write a CleaningRecipe to fix all identified violations "
            f"and prepare this data for ML training. "
            f"The target column is: {findings.target_column}"
        )),
    ]

    recipe: CleaningRecipe = structured_llm.invoke(messages)
    return recipe


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def display_findings(findings: InvestigationFindings) -> None:
    """Pretty-print the investigation findings."""
    print("=" * 70)
    print("INVESTIGATION FINDINGS")
    print("=" * 70)

    print(f"\n  Target Column     : {findings.target_column}")
    print(f"  Target Rationale  : {findings.target_column_rationale}")
    print(f"  Task Type         : {findings.task_type}")
    print(f"  Data Quality Score: {findings.data_quality_score}")

    if findings.violations:
        print(f"\n  Violations ({len(findings.violations)}):")
        for v in findings.violations:
            print(f"\n    [{v.violation_id}] {v.severity} — {v.category}")
            print(f"        Columns : {', '.join(v.affected_columns)}")
            print(f"        Issue   : {v.description}")
            print(f"        Evidence: {v.evidence}")
            print(f"        Fix     : {v.suggested_action}")

    if findings.columns_to_drop:
        print(f"\n  Columns to Drop: {findings.columns_to_drop}")
        for col, reason in findings.columns_to_drop_rationale.items():
            print(f"    {col}: {reason}")

    if findings.key_caveats:
        print(f"\n  Key Caveats for Answer Agent:")
        for c in findings.key_caveats:
            print(f"    ⚠ {c}")

    print()


def display_recipe(recipe: CleaningRecipe) -> None:
    """Pretty-print the cleaning recipe."""
    print("=" * 70)
    print("CLEANING RECIPE")
    print("=" * 70)
    for step in recipe.steps:
        violation_ref = (
            f" (addresses violation #{step.addresses_violation})"
            if step.addresses_violation
            else ""
        )
        print(
            f"\n  Step {step.step_id}: [{step.operation}]"
            + (f" → {step.target_column}" if step.target_column else "")
            + violation_ref
        )
        print(f"    Justification : {step.justification}")
        if step.parameters:
            print(f"    Parameters    : {step.parameters}")
        print(f"    Python code   : {step.python_code}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns.\n")

    # ------------------------------------------------------------------
    # 2. Profile
    # ------------------------------------------------------------------
    print("Running profiler...")
    profile_dict = generate_profile(df, detailed_profiler=False)
    profile = DatasetProfile.model_validate(profile_dict)
    print(f"  Profile complete: {profile.row_count} rows, "
          f"{len(profile.columns)} columns profiled.\n")

    # ------------------------------------------------------------------
    # 3. Build LLM clients (one per agent role)
    # ------------------------------------------------------------------
    print(f"Using OpenAI API")
    print(f"  Investigator model : {INVESTIGATOR_MODEL}")
    print(f"  Code generator model: {CODEGEN_MODEL}\n")
    investigator_llm = ChatOpenAI(
        model=INVESTIGATOR_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
    )
    codegen_llm = ChatOpenAI(
        model=CODEGEN_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
    )

    # ------------------------------------------------------------------
    # 4. Agent 1: Investigator (diagnosis — no code)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STAGE 1: INVESTIGATOR AGENT")
    print("=" * 70)
    print("  Analyzing data quality and identifying violations...\n")

    # Give tools access to the DataFrame
    set_working_df(df)

    # NOTE: Input user query here
    user_query = "What patterns in patient visits predict high-cost outcomes?"

    findings = run_investigator(investigator_llm, profile, user_query)
    display_findings(findings)

    # ------------------------------------------------------------------
    # 5. Agent 2: Code Generator (writes cleaning code)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STAGE 2: CODE GENERATOR AGENT")
    print("=" * 70)
    print("  Generating cleaning code based on investigation findings...\n")

    recipe = run_code_generator(codegen_llm, findings, profile, user_query)
    display_recipe(recipe)

    # ------------------------------------------------------------------
    # 6. Dump full JSON outputs
    # ------------------------------------------------------------------
    print("=" * 70)
    print("RAW JSON OUTPUTS")
    print("=" * 70)

    print("\n--- InvestigationFindings ---")
    print(findings.model_dump_json(indent=2))

    print("\n--- CleaningRecipe ---")
    print(recipe.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Violations found       : {len(findings.violations)}")
    print(f"  Data quality score     : {findings.data_quality_score}")
    print(f"  Target column          : {findings.target_column}")
    print(f"  Task type              : {findings.task_type}")
    print(f"  Cleaning steps planned : {len(recipe.steps)}")
    print(f"  Columns to drop        : {len(findings.columns_to_drop)}")

    ops = {}
    for step in recipe.steps:
        ops[step.operation] = ops.get(step.operation, 0) + 1
    print(f"  Operations breakdown   : {ops}")
    print()


if __name__ == "__main__":
    main()