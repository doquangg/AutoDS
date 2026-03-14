################################################################################
# Human-in-the-loop target column selection.
#
# After profiling, this module:
#   1. Sends column names + types + user query to the LLM to rank candidates
#   2. Presents the top candidates to the user via CLI prompt
#   3. Returns the confirmed target column name
#
# On pass > 0 (multi-pass re-examination), this is a no-op — the target
# column persists in state from the first pass.
#
# Non-interactive mode:
#   Set AUTO_TARGET_COLUMN=column_name to bypass the prompt.
################################################################################

from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from core.state import AgentState
from core.agents import get_investigator_llm
from core.logger import log_node


################################################################################
# Schemas for LLM ranking output
################################################################################

class TargetCandidate(BaseModel):
    name: str = Field(..., description="Exact column name from the dataset")
    rationale: str = Field(..., description="Why this column is a good prediction target for the user's query")


class TargetCandidateList(BaseModel):
    candidates: List[TargetCandidate] = Field(
        ...,
        description="Top target column candidates, ordered by relevance (most likely first)",
    )


################################################################################
# LLM ranking
################################################################################

_RANKING_SYSTEM_PROMPT = """\
You are a data science assistant. Given a user's question and a list of dataset \
columns, identify the top 5 columns most likely to be the prediction target \
(the column the user wants to predict or explain).

Return them ordered by relevance. For each candidate, explain in one sentence \
why it matches the user's intent.\
"""


def rank_target_candidates(
    user_query: str,
    columns_info: List[Dict[str, str]],
) -> List[TargetCandidate]:
    """
    Ask the LLM to rank the top 5 target column candidates.

    Args:
        user_query: The user's natural-language question.
        columns_info: List of {"name": ..., "type": ...} dicts from the profile.

    Returns:
        Ranked list of TargetCandidate objects.
    """
    llm = get_investigator_llm()
    structured_llm = llm.with_structured_output(
        TargetCandidateList, method="function_calling"
    )

    columns_text = "\n".join(
        f"  - {col['name']} ({col['type']})" for col in columns_info
    )

    messages = [
        SystemMessage(content=_RANKING_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"USER QUERY: {user_query}\n\n"
            f"DATASET COLUMNS:\n{columns_text}\n\n"
            f"Return the top 5 most likely target columns."
        )),
    ]

    result = structured_llm.invoke(messages)
    return result.candidates


################################################################################
# CLI prompt
################################################################################

def prompt_user_selection(
    candidates: List[TargetCandidate],
    all_column_names: List[str],
) -> str:
    """
    Display ranked candidates and prompt the user to select one.

    The user can enter a number (1-N) to pick a candidate, or type a column
    name directly to specify one not in the list.

    Args:
        candidates: Ranked candidates from the LLM.
        all_column_names: All column names in the dataset (for validation).

    Returns:
        The selected column name (validated to exist in the dataset).
    """
    print("\n" + "=" * 60)
    print("TARGET COLUMN SELECTION")
    print("=" * 60)
    print(f"\nBased on your query, these columns are the most likely targets:\n")

    for i, candidate in enumerate(candidates, 1):
        print(f"  [{i}] {candidate.name}")
        print(f"      {candidate.rationale}\n")

    print(f"  [C] Enter a custom column name\n")

    while True:
        choice = input("Select a target column (number or 'C'): ").strip()

        # Numeric selection
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                selected = candidates[idx].name
                if selected in all_column_names:
                    print(f"\n  >>> Selected: {selected}\n")
                    return selected
                else:
                    print(f"  Warning: '{selected}' not found in dataset columns. Try again.")
            else:
                print(f"  Invalid number. Enter 1-{len(candidates)} or 'C'.")

        # Custom column name
        elif choice.upper() == "C":
            print(f"\n  Available columns:")
            for name in all_column_names:
                print(f"    - {name}")
            while True:
                custom = input("\n  Enter column name: ").strip()
                if custom in all_column_names:
                    print(f"\n  >>> Selected: {custom}\n")
                    return custom
                else:
                    print(f"  '{custom}' not found in dataset. Try again.")

        else:
            print(f"  Invalid input. Enter a number (1-{len(candidates)}) or 'C'.")


################################################################################
# Graph node function
################################################################################

def select_target_column(state: AgentState) -> Dict[str, Any]:
    """
    Graph node: select the target column via human-in-the-loop.

    On pass 0: ranks candidates via LLM, then prompts the user (or reads
    AUTO_TARGET_COLUMN env var for non-interactive mode).

    On pass > 0: no-op (target_column already set in state).
    """
    # Already selected (multi-pass re-examination)
    if state.get("target_column"):
        log_node("target_selector", "skipping (already set)",
                 target_column=state["target_column"])
        return {}

    profile = state["profile"]
    columns = profile.get("columns", [])
    all_column_names = [col["name"] for col in columns]

    # Non-interactive mode
    auto_target = os.environ.get("AUTO_TARGET_COLUMN", "").strip()
    if auto_target:
        if auto_target in all_column_names:
            log_node("target_selector", "auto-selected via env var",
                     target_column=auto_target)
            print(f"\n  >>> Auto-selected target column: {auto_target} "
                  f"(from AUTO_TARGET_COLUMN env var)\n")
            return {"target_column": auto_target}
        else:
            print(f"  Warning: AUTO_TARGET_COLUMN='{auto_target}' not found "
                  f"in dataset. Falling back to interactive selection.")

    # Build compact column info for ranking
    columns_info = [
        {"name": col["name"], "type": col.get("inferred_type", "unknown")}
        for col in columns
    ]

    # Rank via LLM
    try:
        candidates = rank_target_candidates(state["user_query"], columns_info)
    except Exception as e:
        print(f"  Warning: LLM ranking failed ({e}). Showing all columns.")
        candidates = [
            TargetCandidate(name=name, rationale="(ranking unavailable)")
            for name in sorted(all_column_names)[:5]
        ]

    # Prompt user
    selected = prompt_user_selection(candidates, all_column_names)
    log_node("target_selector", "user selected target column",
             target_column=selected)

    return {"target_column": selected}
