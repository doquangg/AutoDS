################################################################################
# Quality Assessor Agent (Tier 3)
#
# An LLM agent with investigation tools that verifies data quality after
# cleaning. Unlike the investigator (which discovers issues pre-cleaning),
# the assessor focuses on verification: "Is the data clean enough to stop?"
#
# INPUT:  profile + anomaly summary + cleaning history + structural score
# OUTPUT: LLMQualityAssessment (score, recommendation, reasoning, residual issues)
#
# The assessor has access to the same investigation tools as the investigator
# but uses them in a targeted verification mode (capped at fewer tool calls).
################################################################################

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.pipeline.state import AgentState
from core.schemas import LLMQualityAssessment
from core.runtime.tools import investigation_tools
from core.logger import log_llm_request, log_llm_response, log_node
from core.prompts import QUALITY_ASSESSOR_SYSTEM_PROMPT
from core.agents.agents import _get_llm, extract_structured_output, format_pass_history


################################################################################
# LLM Instance
################################################################################


def get_assessor_llm() -> ChatOpenAI:
    """Returns the LLM for the quality assessor (default: gpt-5.4-mini)."""
    return _get_llm(os.environ.get("ASSESSOR_MODEL", "gpt-5.4-mini"))


################################################################################
# Agent Entry Point
################################################################################

MAX_ASSESSOR_TOOL_CALLS = 8


def run_quality_assessor_agent(
    state: AgentState,
    structural_score_data: Dict[str, Any],
    anomaly_summary_data: Dict[str, Any],
    max_tool_calls: int = MAX_ASSESSOR_TOOL_CALLS,
) -> Dict[str, Any]:
    """
    Runs the quality assessor agent. Returns state updates including
    the LLMQualityAssessment and assessor_messages.

    The assessor uses tools via LangChain's bind_tools() for investigation.
    It is called repeatedly by the graph until it stops requesting tools.

    Args:
        state: Current pipeline state.
        structural_score_data: Tier 1 structural score dict.
        anomaly_summary_data: Tier 2 anomaly summary dict.
        max_tool_calls: Cap on tool calls per assessment.
    """
    llm = get_assessor_llm()

    llm_with_tools = llm.bind_tools(
        investigation_tools,
        response_format=LLMQualityAssessment,
        strict=True,
    )

    messages = list(state.get("assessor_messages", []))

    # On first call (no messages yet), inject system prompt and context
    if not messages:
        profile_json = json.dumps(state["profile"], indent=2, default=str)
        anomaly_json = json.dumps(anomaly_summary_data, indent=2, default=str)
        structural_json = json.dumps(structural_score_data, indent=2, default=str)

        # Build cleaning history context
        pass_history = state.get("pass_history", [])
        history_summary = format_pass_history(pass_history, empty_label="  (first pass)")

        user_content = (
            f"DATASET PROFILE ({state['profile']['row_count']} rows):\n"
            f"{profile_json}\n\n"
            f"STRUCTURAL SCORE (Tier 1):\n{structural_json}\n\n"
            f"ANOMALY SUMMARY (Tier 2):\n{anomaly_json}\n\n"
            f"CLEANING HISTORY:\n{history_summary}\n\n"
            f"TARGET COLUMN: {state.get('target_column') or 'not specified'}\n\n"
            f"Investigate this data and provide your quality assessment."
        )

        messages = [
            SystemMessage(content=QUALITY_ASSESSOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        log_llm_request(
            "quality_assessor",
            pass_count=state.get("pass_count", 0),
            message_count=len(messages),
            system_prompt_snippet=QUALITY_ASSESSOR_SYSTEM_PROMPT[:200],
            user_message_snippet=user_content[:300],
        )

    response = llm_with_tools.invoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    log_llm_response(
        "quality_assessor",
        content=response.content if isinstance(response.content, str) else str(response.content),
        tool_calls=getattr(response, "tool_calls", None),
    )

    new_tool_count = state.get("assessor_tool_call_count", 0) + len(
        getattr(response, "tool_calls", []) or []
    )
    at_limit = new_tool_count >= max_tool_calls

    updates: Dict[str, Any] = {
        "assessor_messages": [response],
        "assessor_tool_call_count": new_tool_count,
    }

    assessment = extract_structured_output(
        response, messages, llm, LLMQualityAssessment, at_limit, has_tool_calls
    )
    if assessment:
        log_node("quality_assessor", "assessment complete",
                 score=assessment.score,
                 recommendation=assessment.recommendation,
                 residual_issues=len(assessment.residual_issues))
        updates["llm_quality_assessment"] = assessment

    return updates
