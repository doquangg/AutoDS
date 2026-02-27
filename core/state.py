################################################################################
# This file documents the exact layout of the agent state, dictating the exact
# information passed between nodes in the LangGraph pipeline.
#
# KEY DESIGN DECISIONS:
#   1. Investigator and Code Generator use SEPARATE message lists. This prevents
#      investigation context from polluting code-generation retries, and keeps
#      each agent's context window focused.
#   2. InvestigationFindings is a structured artifact that persists across 
#      code-gen retries. The investigator runs ONCE; only the code generator
#      retries on sandbox failure.
#   3. tool_call_count caps runaway investigation tool loops.
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
    CleaningLogEntry,
)


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
    investigator_messages: Annotated[List[BaseMessage], operator.add]

    # --- Code Generator Agent Context ---
    # Separate message list for the code generator. On retry, we can trim this
    # without losing investigation context.
    codegen_messages: Annotated[List[BaseMessage], operator.add]

    # --- Cleaning Artifacts ---
    profile: Optional[DatasetProfile]
    investigation_findings: Optional[InvestigationFindings]  # NEW: structured handoff
    current_plan: Optional[CleaningRecipe]

    # --- Audit Trail ---
    cleaning_history: Annotated[List[CleaningLogEntry], operator.add]

    # --- Control Flow ---
    retry_count: int
    latest_error: Optional[str]
    tool_call_count: int            # NEW: caps investigation tool loops
    final_answer: Optional[str]