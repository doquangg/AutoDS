################################################################################
# Agent definitions for the AutoDS pipeline.
#
# ARCHITECTURE:
#   This file defines THREE agents, each using a dedicated OpenAI model.
#   They are differentiated by their system prompts, output schemas, and models.
#
#   1. Investigator Agent   — Reads the profile, calls tools, produces
#                             InvestigationFindings (structured diagnosis).
#                             Default model: gpt-5-2025-08-07
#   2. Code Generator Agent — Reads findings, writes CleaningRecipe (Python code).
#                             This is the ONLY agent that retries on sandbox errors.
#                             Default model: gpt-4.1-2025-04-14
#   3. Answer Agent         — Reads model results + findings, writes the final
#                             natural language answer.
#                             Default model: gpt-4.1-2025-04-14
#
# LLM CONFIGURATION:
#   All agents use OpenAI's API. Set OPENAI_API_KEY in your environment.
#   Model defaults can be overridden per-role via environment variables:
#     INVESTIGATOR_MODEL  (default: gpt-5-2025-08-07)
#     CODEGEN_MODEL       (default: gpt-4.1-2025-04-14)
#     ANSWER_MODEL        (default: gpt-4.1-2025-04-14)
#
#   To use a local OpenAI-compatible server (Ollama, vLLM, etc.), also set:
#     OPENAI_API_BASE=http://localhost:8000/v1
################################################################################

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.pipeline.state import AgentState
from core.schemas import InvestigationFindings, CleaningRecipe, CleanlinessEvaluation
from core.runtime.tools import investigation_tools
from core.logger import (
    log_llm_request, log_llm_response, log_investigation_findings,
    log_cleaning_recipe,
)
from core.prompts import (
    INVESTIGATOR_SYSTEM_PROMPT,
    INVESTIGATOR_REEXAM_PROMPT,
    CODEGEN_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
)


################################################################################
# Per-Role LLM Instances
################################################################################

_llm_cache: dict[str, ChatOpenAI] = {}


def _get_llm(model: str) -> ChatOpenAI:
    """Returns a cached ChatOpenAI instance for the given model name."""
    if model not in _llm_cache:
        _llm_cache[model] = ChatOpenAI(
            model=model,
            temperature=0.0,  # Deterministic for reproducibility
            base_url=os.environ.get("OPENAI_API_BASE"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _llm_cache[model]


def get_investigator_llm() -> ChatOpenAI:
    """Returns the LLM for the investigator agent (default: gpt-5.1-2025-11-13)."""
    return _get_llm(os.environ.get("INVESTIGATOR_MODEL", "gpt-5.1-2025-11-13"))


def get_codegen_llm() -> ChatOpenAI:
    """Returns the LLM for the code generator agent (default: gpt-5.1-2025-11-13)."""
    return _get_llm(os.environ.get("CODEGEN_MODEL", "gpt-5.1-2025-11-13"))


def get_answer_llm() -> ChatOpenAI:
    """Returns the LLM for the answer agent (default: gpt-5.1-2025-11-13)."""
    return _get_llm(os.environ.get("ANSWER_MODEL", "gpt-5.1-2025-11-13"))


def get_evaluator_llm() -> ChatOpenAI:
    """Returns the LLM for the evaluator agent (default: gpt-4.1-2025-04-14)."""
    return _get_llm(os.environ.get("EVALUATOR_MODEL", "gpt-4.1-2025-04-14"))


################################################################################
# Agent 1: Investigator
#
# INPUT:  profile (DatasetProfile) + user_query + tool access
# OUTPUT: investigation_findings (InvestigationFindings)
#
# This agent ONLY diagnoses problems. It does NOT write Python code.
# It can call tools to inspect the data more deeply.
################################################################################


def run_investigator_agent(state: AgentState, max_tool_calls: int = 30) -> Dict[str, Any]:
    """
    Runs the investigator agent. Returns state updates including
    investigation_findings and investigator_messages.

    The investigator uses tools via LangChain's bind_tools(), which means
    LangGraph's ToolNode handles execution. This function is called
    repeatedly until the agent stops requesting tools.

    On pass > 0 (multi-pass re-examination), the system prompt is augmented
    with context about previous passes so the investigator can judge whether
    the data is now clean enough.

    Args:
        max_tool_calls: When the cumulative tool call count reaches this value,
            findings are extracted immediately even if the LLM wanted more tools.
            Pass MAX_TOOL_CALLS from graph.py to enforce the graph-level cap.
    """
    llm = get_investigator_llm()

    # Bind investigation tools so the LLM can call them, and fix the response
    # format
    llm_with_tools = llm.bind_tools(
        investigation_tools,
        response_format=InvestigationFindings,
        strict=True,
    )

    # Build messages for this agent's context
    messages = list(state.get("investigator_messages", []))

    # On first call (no messages yet), inject the system prompt and profile
    if not messages:
        profile_json = json.dumps(state["profile"], indent=2, default=str)
        pass_count = state.get("pass_count", 0)

        system_prompt = INVESTIGATOR_SYSTEM_PROMPT

        # On subsequent passes, append re-examination context
        if pass_count > 0:
            pass_history = state.get("pass_history", [])
            history_lines = []
            for ph in pass_history:
                history_lines.append(
                    f"  Pass {ph['pass_number']}: {ph['violations_found']} violations found, "
                    f"quality={ph['quality_score']}, {ph['steps_executed']} steps executed, "
                    f"{ph['rows_after']} rows remaining"
                )
            pass_history_summary = "\n".join(history_lines) if history_lines else "  (none)"

            system_prompt += INVESTIGATOR_REEXAM_PROMPT.format(
                pass_number=pass_count + 1,
                pass_history_summary=pass_history_summary,
                target_column=state.get("target_column") or "not yet confirmed",
            )

        # Include confirmed target column if set by human-in-the-loop selection
        target_section = ""
        if state.get("target_column"):
            target_section = (
                f"CONFIRMED TARGET COLUMN: {state['target_column']}\n"
                f"(Selected by the user. Use this as a given fact — "
                f"do NOT re-determine it.)\n\n"
            )

        user_content = (
            f"USER QUERY: {state['user_query']}\n\n"
            f"{target_section}"
            f"{'RE-EXAMINATION ' if pass_count > 0 else ''}"
            f"DATASET PROFILE ({state['profile']['row_count']} rows):\n"
            f"{profile_json}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        log_llm_request(
            "investigator", pass_count=pass_count,
            message_count=len(messages),
            system_prompt_snippet=system_prompt,
            user_message_snippet=user_content,
        )

    response = llm_with_tools.invoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    log_llm_response(
        "investigator",
        content=response.content if isinstance(response.content, str) else str(response.content),
        tool_calls=getattr(response, "tool_calls", None),
    )
    new_tool_count = state.get("tool_call_count", 0) + len(
        getattr(response, "tool_calls", []) or []
    )
    at_limit = new_tool_count >= max_tool_calls

    updates: Dict[str, Any] = {
        "investigator_messages": [response],
        "tool_call_count": new_tool_count,
    }

    if not has_tool_calls or at_limit:
        if at_limit and has_tool_calls:
            # Model wanted more tools but hit the cap.
            # Force a structured-only call (no tools) to extract findings.
            structured_only = llm.with_structured_output(
                InvestigationFindings, method="function_calling"
            )
            finalization_messages = messages + [
                HumanMessage(content=(
                    "Tool call limit reached. Based on your investigation so far, "
                    "provide your final InvestigationFindings."
                ))
            ]
            findings = structured_only.invoke(finalization_messages)
        else:
            # Normal case: response.content is already InvestigationFindings JSON.
            # Parse it directly — no second LLM call needed.
            parsed = response.additional_kwargs.get("parsed")
            if parsed and isinstance(parsed, InvestigationFindings):
                findings = parsed
            else:
                content = response.content
                data = json.loads(content)
                findings = InvestigationFindings(**data)
    
        log_investigation_findings(findings)
        updates["investigation_findings"] = findings
    return updates


################################################################################
# Agent 2: Code Generator
#
# INPUT:  investigation_findings + profile + (on retry: error message)
# OUTPUT: current_plan (CleaningRecipe with executable Python code)
#
# This agent ONLY writes code. It does NOT re-investigate the data.
# On retry, it receives the specific error and the failed code, and must
# fix only the broken step.
################################################################################


def run_codegen_agent(state: AgentState) -> Dict[str, Any]:
    """
    Runs the code generator agent. Returns state updates including
    current_plan and codegen_messages.
    """
    llm = get_codegen_llm()
    structured_llm = llm.with_structured_output(CleaningRecipe, method="function_calling")
    
    messages = []
    
    # Build the prompt based on whether this is a first run or retry
    is_retry = state.get("latest_error") is not None and state.get("retry_count", 0) > 0
    
    if is_retry:
        # On retry: include the error context but NOT the full investigation.
        # The findings haven't changed — only the code needs fixing.
        previous_plan = state.get("current_plan")
        previous_plan_json = json.dumps(
            previous_plan.model_dump() if hasattr(previous_plan, "model_dump") 
            else previous_plan, 
            indent=2, default=str
        )
        
        messages = [
            SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"RETRY #{state['retry_count']} — Your previous code FAILED.\n\n"
                f"ERROR MESSAGE:\n{state['latest_error']}\n\n"
                f"PREVIOUS PLAN THAT FAILED:\n{previous_plan_json}\n\n"
                f"Fix the failing step. Do not change steps that succeeded.\n"
                f"Return the complete corrected CleaningRecipe."
            )),
        ]
    else:
        # First run: provide findings + profile
        findings = state["investigation_findings"]
        findings_json = json.dumps(
            findings.model_dump() if hasattr(findings, "model_dump") 
            else findings,
            indent=2, default=str
        )
        profile_json = json.dumps(state["profile"], indent=2, default=str)
        
        messages = [
            SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"USER QUERY: {state['user_query']}\n\n"
                f"INVESTIGATION FINDINGS:\n{findings_json}\n\n"
                f"DATASET PROFILE:\n{profile_json}\n\n"
                f"Write a CleaningRecipe to fix all identified violations "
                f"and prepare this data for ML training."
            )),
        ]

    log_llm_request(
        "codegen", pass_count=state.get("pass_count", 0),
        is_retry=is_retry, retry_count=state.get("retry_count", 0),
        message_count=len(messages),
        system_prompt_snippet=messages[0].content if messages else "",
        user_message_snippet=messages[1].content if len(messages) > 1 else "",
    )

    recipe = structured_llm.invoke(messages)
    log_cleaning_recipe(recipe)

    return {
        "current_plan": recipe,
        "codegen_messages": messages + [
            AIMessage(content=json.dumps(recipe.model_dump(), default=str))
        ],
    }


################################################################################
# Agent 3: Answer Agent
#
# INPUT:  model_metadata + investigation_findings + user_query
# OUTPUT: final_answer (natural language business answer)
################################################################################


def run_answer_agent(state: AgentState) -> str:
    """
    Runs the answer agent. Returns the final_answer string.
    """
    llm = get_answer_llm()

    # Serialize model metadata
    model_meta = state.get("model_metadata", {})
    model_json = json.dumps(model_meta, indent=2, default=str)

    # Serialize investigation findings for caveats
    findings = state.get("investigation_findings")
    caveats_section = ""
    if findings:
        findings_data = (
            findings.model_dump() if hasattr(findings, "model_dump") else findings
        )
        caveats = findings_data.get("key_caveats", [])
        quality = findings_data.get("data_quality_score", "unknown")
        violations = findings_data.get("violations", [])

        caveats_section = (
            f"\nDATA QUALITY SCORE: {quality}\n"
            f"VIOLATIONS FOUND: {len(violations)}\n"
            f"KEY CAVEATS:\n" + "\n".join(f"- {c}" for c in caveats)
        )

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"USER QUESTION: {state['user_query']}\n\n"
            f"MODEL RESULTS:\n{model_json}\n"
            f"{caveats_section}"
        )),
    ]

    response = llm.invoke(messages)
    return response.content


################################################################################
# Agent 4: Evaluator
#
# INPUT:  post-cleaning profile + pass history
# OUTPUT: CleanlinessEvaluation (is_data_clean, quality_score, rationale)
#
# This agent runs AFTER the sandbox executes cleaning code. It examines the
# post-cleaning profile to decide whether data is clean enough for modeling.
# This replaces the previous approach where the investigator set is_data_clean
# before any cleaning code ran.
################################################################################


def run_evaluator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Runs the evaluator agent on post-cleaning data.
    Returns state updates including is_data_clean.
    """
    llm = get_evaluator_llm()
    structured_llm = llm.with_structured_output(
        CleanlinessEvaluation, method="function_calling"
    )

    profile_json = json.dumps(state["profile"], indent=2, default=str)

    pass_history = state.get("pass_history", [])
    history_lines = []
    for ph in pass_history:
        history_lines.append(
            f"  Pass {ph['pass_number']}: {ph['violations_found']} violations found, "
            f"quality={ph['quality_score']}, {ph['steps_executed']} steps executed, "
            f"{ph['rows_after']} rows remaining"
        )
    pass_history_summary = "\n".join(history_lines) if history_lines else "  (none)"

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"POST-CLEANING DATASET PROFILE ({state['profile']['row_count']} rows):\n"
            f"{profile_json}\n\n"
            f"PASS HISTORY:\n{pass_history_summary}\n\n"
            f"Evaluate whether this data is clean enough for modeling."
        )),
    ]

    log_llm_request(
        "evaluator", pass_count=state.get("pass_count", 0),
        message_count=len(messages),
        system_prompt_snippet=EVALUATOR_SYSTEM_PROMPT[:200],
        user_message_snippet=messages[1].content[:300],
    )

    evaluation = structured_llm.invoke(messages)

    log_llm_response(
        "evaluator",
        content=f"is_data_clean={evaluation.is_data_clean}, "
                f"quality_score={evaluation.quality_score}, "
                f"rationale={evaluation.rationale}",
    )

    return {"is_data_clean": evaluation.is_data_clean}