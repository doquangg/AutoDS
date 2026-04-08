################################################################################
# This file documents the exact layout of the agent state, dictating the exact
# information passed between nodes in the LangGraph pipeline.
#
# KEY DESIGN DECISIONS:
#   1. Investigator and Code Generator use SEPARATE message lists. This prevents
#      investigation context from polluting code-generation retries, and keeps
#      each agent's context window focused.
#   2. InvestigationFindings is a structured artifact that persists across
#      code-gen retries. The investigator runs ONCE per pass; only the code
#      generator retries on sandbox failure.
#   3. tool_call_count caps runaway investigation tool loops.
#   4. Multi-pass: the profiler → investigator → codegen → sandbox cycle repeats
#      until the investigator declares the data clean or MAX_PASSES is reached.
#      Message lists use a resettable reducer so they can be cleared between
#      passes while still supporting append within a single pass.
################################################################################

import operator
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.messages import BaseMessage

from core.schemas import (
    DatasetProfile,
    InvestigationFindings,
    CleaningRecipe,
    CleaningStep,
    CleaningLogEntry,
)


def _resettable_list_reducer(existing: list, new: list) -> list:
    """
    List reducer that supports reset between passes.

    Normally appends (like operator.add). But if the new list starts with the
    string "__RESET__", the existing list is discarded and replaced with
    everything after the sentinel.
    """
    if new and isinstance(new[0], str) and new[0] == "__RESET__":
        return list(new[1:])
    return existing + new


class AgentState(TypedDict):
    # --- Inputs ---
    user_query: str

    # --- Data Assets ---
    working_df: pd.DataFrame
    clean_df: Optional[pd.DataFrame]
    model_metadata: Optional[Dict[str, Any]]  # Fixed: was Optional[str] (Issue #21)

    # --- Investigator Agent Context ---
    # These messages are ONLY used by the investigator agent and its tool loop.
    # They are not touched by the code generator or answer agent.
    # Uses resettable reducer: cleared between passes, appended within a pass.
    investigator_messages: Annotated[List[BaseMessage], _resettable_list_reducer]

    # --- Code Generator Agent Context ---
    # Separate message list for the code generator. On retry, we can trim this
    # without losing investigation context.
    # Uses resettable reducer: cleared between passes, appended within a pass.
    codegen_messages: Annotated[List[BaseMessage], _resettable_list_reducer]

    # --- Cleaning Artifacts ---
    profile: Optional[DatasetProfile]
    investigation_findings: Optional[InvestigationFindings]
    current_plan: Optional[CleaningRecipe]

    # Accumulating flat list of every CleaningStep that was successfully
    # executed by the sandbox, across all passes. Unlike current_plan (which
    # holds only the current pass's recipe and is reset between passes), this
    # field persists across passes via the operator.add reducer. Consumers
    # that need to replay the graph's full cleaning on a held-out fold (e.g.
    # scripts/evaluate_benchmarks.py) MUST use this field, not current_plan.
    applied_steps: Annotated[List[CleaningStep], operator.add]

    # --- Audit Trail ---
    cleaning_history: Annotated[List[CleaningLogEntry], operator.add]

    # --- Control Flow ---
    retry_count: int
    latest_error: Optional[str]
    tool_call_count: int            # Caps investigation tool loops
    final_answer: Optional[str]

    # --- Target Column (human-in-the-loop) ---
    target_column: Optional[str]    # Human-confirmed target; set once on pass 0, persists across passes

    # --- Multi-Pass Control ---
    pass_count: int                 # Current pass number (0-indexed, incremented after sandbox success)
    pass_history: Annotated[List[Dict[str, Any]], operator.add]  # Compact per-pass summaries
    previous_findings: Optional[InvestigationFindings]  # Findings from prior pass, used to prevent re-flagging

