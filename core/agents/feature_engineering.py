"""
Feature Engineering stage: planner + codegen agents and the LangGraph node
that drives up to MAX_FE_ROUNDS rounds of planner -> codegen -> sandbox
-> re-profile. Mirrors the cleaning investigator/codegen split but runs
tools in-process because a single graph node can't invoke LangGraph's
ToolNode. Target-leakage is prevented by the system prompts plus a
defensive literal-match lint over generated python_code.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)

from core.agents.agents import _get_llm, extract_structured_output
from core.pipeline.state import AgentState
from core.schemas import (
    FeatureProposal,
    FeatureRecipe,
    FeatureStep,
    CleaningLogEntry,
)
from core.runtime.tools import investigation_tools, set_working_df
from core.runtime.sandbox import execute_plan_in_sandbox
from core.prompts import FE_PLANNER_SYSTEM_PROMPT, FE_CODEGEN_SYSTEM_PROMPT
from core.logger import log_node, log_llm_request, log_llm_response
from plugins.profiler import generate_profile


MAX_FE_ROUNDS = 3
MAX_FE_RETRIES = 3
MAX_FE_PLANNER_TOOL_CALLS = 15


def get_fe_planner_llm():
    return _get_llm(os.environ.get("FE_PLANNER_MODEL", "gpt-5.4"))


def get_fe_codegen_llm():
    return _get_llm(os.environ.get("FE_CODEGEN_MODEL", "gpt-5.4-mini"))


def _profile_to_json(profile) -> str:
    """Serialize a DatasetProfile (dict or pydantic) to a JSON string."""
    if profile is None:
        return "{}"
    if hasattr(profile, "model_dump"):
        return json.dumps(profile.model_dump(), indent=2, default=str)
    return json.dumps(profile, indent=2, default=str)


def _summarize_applied(applied_so_far: List[FeatureStep]) -> str:
    """Render a compact bullet list of features added in prior rounds."""
    if not applied_so_far:
        return "  (none yet)"
    lines = []
    for s in applied_so_far:
        lines.append(
            f"  [round {s.round_number}] {s.new_column} ({s.operation}): "
            f"{s.justification[:80]}"
        )
    return "\n".join(lines)


def _get_task_type(state: AgentState) -> Optional[str]:
    """Extract task_type from investigation_findings if available."""
    findings = state.get("investigation_findings")
    if findings is None:
        return None
    if hasattr(findings, "task_type"):
        return findings.task_type
    if isinstance(findings, dict):
        return findings.get("task_type")
    return None


def run_fe_planner_agent(
    state: AgentState,
    df,
    profile,
    round_num: int,
    applied_so_far: List[FeatureStep],
) -> FeatureProposal:
    """
    One planner invocation. Drives its own internal tool loop because the FE
    node is a single LangGraph node and has no access to a graph-level ToolNode.

    Returns a FeatureProposal. May raise on LLM / parsing errors — the caller
    is expected to catch and log.
    """
    # CRITICAL: refresh the module-level dataframe reference used by all tools.
    # Without this call, tools invoked during round 2/3 would inspect the
    # round-0 dataframe and miss columns added by prior rounds.
    set_working_df(df)

    llm = get_fe_planner_llm()
    llm_with_tools = llm.bind_tools(
        investigation_tools,
        response_format=FeatureProposal,
        strict=True,
    )

    target_col = state.get("target_column")
    task_type = _get_task_type(state)
    profile_json = _profile_to_json(profile)
    applied_summary = _summarize_applied(applied_so_far)

    user_content = (
        f"USER QUERY: {state['user_query']}\n"
        f"TARGET COLUMN: {target_col}\n"
        f"TASK TYPE: {task_type}\n"
        f"ROUND: {round_num} of {MAX_FE_ROUNDS}\n\n"
        f"FEATURES ALREADY ADDED IN PRIOR ROUNDS:\n{applied_summary}\n\n"
        f"FORBIDDEN ON RHS OF ANY FEATURE EXPRESSION: {target_col}\n\n"
        f"DATASET PROFILE:\n{profile_json}"
    )

    messages: List[BaseMessage] = [
        SystemMessage(content=FE_PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    log_llm_request(
        "fe_planner",
        pass_count=round_num,
        message_count=len(messages),
        system_prompt_snippet=FE_PLANNER_SYSTEM_PROMPT,
        user_message_snippet=user_content,
    )

    # Drive tools in-process: LangGraph's ToolNode can't be invoked from
    # inside a node body.
    tool_calls_made = 0
    tools_by_name = {t.name: t for t in investigation_tools}
    response = None
    has_tool_calls = False

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        has_tool_calls = bool(getattr(response, "tool_calls", None))

        log_llm_response(
            "fe_planner",
            content=response.content if isinstance(response.content, str) else str(response.content),
            tool_calls=getattr(response, "tool_calls", None),
        )

        if not has_tool_calls:
            break
        if tool_calls_made >= MAX_FE_PLANNER_TOOL_CALLS:
            break

        for tc in response.tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {}) or {}
            tool_id = tc.get("id", "")
            tool_fn = tools_by_name.get(tool_name)
            if tool_fn is None:
                result = f"tool '{tool_name}' not found"
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    result = f"tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            tool_calls_made += 1

    at_limit = tool_calls_made >= MAX_FE_PLANNER_TOOL_CALLS
    proposal = extract_structured_output(
        response,
        messages,
        llm,
        FeatureProposal,
        at_limit=at_limit,
        has_tool_calls=has_tool_calls,
    )

    # The model sometimes echoes a stale round_number; rewrite it.
    if proposal.round_number != round_num:
        proposal = FeatureProposal(
            round_number=round_num,
            ideas=proposal.ideas,
            no_more_features=proposal.no_more_features,
            reasoning=proposal.reasoning,
        )

    return proposal


def _leakage_lint(recipe: FeatureRecipe, target_col: Optional[str]) -> FeatureRecipe:
    """
    Drop any step whose python_code literally references the target column.

    This is a defensive syntactic guard against the most common LLM mistake
    (using the target column as a feature input). The primary defense is the
    system prompt; this is a hygiene backstop, not a statistical check.
    """
    if not target_col:
        return recipe

    forbidden_literals = (
        f"'{target_col}'",
        f'"{target_col}"',
    )

    safe_steps: List[FeatureStep] = []
    for step in recipe.steps:
        code = step.python_code or ""
        if any(lit in code for lit in forbidden_literals):
            log_node(
                "fe_codegen",
                f"DROPPED leakage-risk step: {step.new_column}",
                code_snippet=code[:160],
            )
            continue
        # Also forbid steps that try to create a column with the target name
        if step.new_column == target_col:
            log_node(
                "fe_codegen",
                f"DROPPED step targeting the target column: {step.new_column}",
            )
            continue
        safe_steps.append(step)

    return FeatureRecipe(round_number=recipe.round_number, steps=safe_steps)


def run_fe_codegen_agent(
    state: AgentState,
    proposal: FeatureProposal,
    profile,
    error: Optional[str] = None,
    previous_recipe: Optional[FeatureRecipe] = None,
) -> FeatureRecipe:
    """
    One codegen invocation. Mirrors run_codegen_agent from core.agents.agents.

    Returns a FeatureRecipe with executable python_code per step. On retry,
    the error message and the previous failing recipe are included so the
    model can fix only the broken step.
    """
    llm = get_fe_codegen_llm()
    structured_llm = llm.with_structured_output(FeatureRecipe, method="function_calling")

    target_col = state.get("target_column")
    proposal_json = json.dumps(proposal.model_dump(), indent=2, default=str)
    profile_json = _profile_to_json(profile)

    is_retry = error is not None and previous_recipe is not None
    if is_retry:
        prev_json = json.dumps(previous_recipe.model_dump(), indent=2, default=str)
        user_content = (
            f"RETRY — your previous FeatureRecipe FAILED in the sandbox.\n\n"
            f"ERROR MESSAGE:\n{error}\n\n"
            f"PREVIOUS RECIPE THAT FAILED:\n{prev_json}\n\n"
            f"TARGET COLUMN (forbidden on RHS): {target_col}\n\n"
            f"Fix only the failing step. Keep the steps that succeeded unchanged. "
            f"Return the complete corrected FeatureRecipe."
        )
    else:
        user_content = (
            f"USER QUERY: {state['user_query']}\n"
            f"TARGET COLUMN (forbidden on RHS): {target_col}\n"
            f"ROUND: {proposal.round_number} of {MAX_FE_ROUNDS}\n\n"
            f"FEATURE PROPOSAL:\n{proposal_json}\n\n"
            f"DATASET PROFILE:\n{profile_json}\n\n"
            f"Translate EVERY idea into exactly one FeatureStep whose python_code "
            f"adds a single new column named idea.name. Use only `df`, `pd`, `np`."
        )

    messages: List[BaseMessage] = [
        SystemMessage(content=FE_CODEGEN_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    log_llm_request(
        "fe_codegen",
        pass_count=proposal.round_number,
        is_retry=is_retry,
        message_count=len(messages),
        system_prompt_snippet=FE_CODEGEN_SYSTEM_PROMPT,
        user_message_snippet=user_content,
    )

    logger = logging.getLogger("autods")
    try:
        recipe = structured_llm.invoke(messages)
    except ValidationError as exc:
        logger.warning("fe_codegen validation error (will retry once): %s", exc)
        messages.append(AIMessage(content="[invalid response]"))
        messages.append(HumanMessage(content=(
            f"Your previous response failed schema validation:\n{exc}\n\n"
            f"Return a corrected FeatureRecipe. Every step must have step_id, "
            f"idea_id, round_number, new_column, operation, source_columns, "
            f"justification, and python_code."
        )))
        recipe = structured_llm.invoke(messages)

    # Force round_number consistency
    if recipe.round_number != proposal.round_number:
        recipe = FeatureRecipe(round_number=proposal.round_number, steps=recipe.steps)

    log_llm_response(
        "fe_codegen",
        content=f"FeatureRecipe with {len(recipe.steps)} steps",
    )

    # Defensive target-leakage lint
    recipe = _leakage_lint(recipe, target_col)
    return recipe


def node_feature_engineering(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node that runs up to MAX_FE_ROUNDS rounds of feature engineering.

    Each round: planner -> codegen -> sandbox (with retries) -> re-profile.
    Early stops on:
        - empty proposal
        - proposal.no_more_features == True
        - codegen retries exhausted (aborts FE entirely, keeps prior rounds)
        - a round where all steps were dropped by the leakage lint
    """
    print("--- Feature Engineering ---")

    df = state.get("working_df")
    # Need at least one row and one non-target column for FE to be meaningful.
    if df is None or df.shape[0] == 0 or df.shape[1] <= 1:
        log_node("feature_engineering", "skipped — df empty or target-only")
        return {
            "applied_fe_steps": [],
            "fe_history": [],
            "fe_log": [],
        }

    profile = state.get("profile")
    target_col = state.get("target_column")

    applied_so_far: List[FeatureStep] = []
    fe_logs: List[CleaningLogEntry] = []
    fe_history: List[Dict[str, Any]] = []

    for round_num in range(1, MAX_FE_ROUNDS + 1):
        log_node("feature_engineering", f"round {round_num} planner starting")

        try:
            proposal = run_fe_planner_agent(state, df, profile, round_num, applied_so_far)
        except Exception as e:
            log_node(
                "feature_engineering",
                "planner failed; aborting FE",
                error=str(e)[:300],
            )
            print(f"  [FE] planner error in round {round_num}: {e}")
            break

        if not proposal.ideas or proposal.no_more_features:
            log_node(
                "feature_engineering",
                "planner signaled stop",
                round=round_num,
                ideas=len(proposal.ideas),
                no_more=proposal.no_more_features,
            )
            print(f"  [FE] round {round_num}: stopping (ideas={len(proposal.ideas)}, "
                  f"no_more_features={proposal.no_more_features})")
            break

        recipe: Optional[FeatureRecipe] = None
        previous_recipe: Optional[FeatureRecipe] = None
        last_error: Optional[str] = None
        round_logs: List[CleaningLogEntry] = []
        new_df = None
        success = False

        for attempt in range(MAX_FE_RETRIES + 1):
            try:
                recipe = run_fe_codegen_agent(
                    state, proposal, profile,
                    error=last_error, previous_recipe=previous_recipe,
                )
            except Exception as e:
                last_error = f"codegen exception: {e}"
                previous_recipe = None
                continue

            # Lint stripped everything (or the model returned an empty recipe).
            # Retrying won't help — the planner's ideas referenced the target
            # column, so codegen would reproduce the same forbidden expressions.
            if not recipe.steps:
                last_error = "Recipe had no executable steps after leakage lint."
                break

            new_df, round_logs, sb_error = execute_plan_in_sandbox(df, recipe)
            if sb_error is None:
                success = True
                break
            last_error = sb_error
            previous_recipe = recipe

        if not success:
            log_node(
                "feature_engineering",
                f"round {round_num} aborted after retries",
                error=(last_error or "")[:300],
            )
            print(f"  [FE] round {round_num}: aborted — {last_error}")
            break

        df = new_df
        applied_so_far.extend(recipe.steps)
        fe_logs.extend(round_logs)
        fe_history.append({
            "round_number": round_num,
            "ideas_proposed": len(proposal.ideas),
            "steps_executed": len(recipe.steps),
            "columns_after": len(df.columns),
        })
        print(
            f"  [FE] round {round_num}: added {len(recipe.steps)} features, "
            f"total cols={len(df.columns)}"
        )
        log_node(
            "feature_engineering",
            f"round {round_num} complete",
            steps=len(recipe.steps),
            columns_after=len(df.columns),
        )

        # Re-profile so the next round's planner sees the new columns.
        # Skip on the final round since no further planner call will consume it.
        if round_num < MAX_FE_ROUNDS:
            try:
                profile = generate_profile(df, detailed_profiler=False, target_column=target_col)
            except Exception as e:
                log_node(
                    "feature_engineering",
                    "re-profile failed; continuing with stale profile",
                    error=str(e)[:300],
                )

    return {
        "working_df": df,
        "clean_df": df,
        "profile": profile,
        "applied_fe_steps": applied_so_far,
        "fe_log": fe_logs,
        "fe_history": fe_history,
    }
