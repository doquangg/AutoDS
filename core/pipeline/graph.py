################################################################################
# LangGraph state machine for the AutoDS pipeline.
#
# FLOW (multi-pass with three-tier quality assessment):
#   START → profiler → target_selector → investigator ⟲ tools →
#           code_generator → sandbox → [error?] → retry/fail
#                                     → [success] → quality_gate (Tiers 1&2)
#                                                    → [hard blocker?] → pass_reset
#                                                    → quality_assessor ⟲ tools (Tier 3)
#                                                       → [stop?] → autogluon → answer → END
#                                                       → [continue?] → pass_reset → profiler …
#
# KEY DESIGN DECISIONS:
#   1. SPLIT: investigator (diagnosis) and code_generator (Python code) are
#      separate agents. Investigation happens ONCE per pass; only code_generator
#      retries on sandbox failure.
#   2. Tool loop cap: investigator can call at most MAX_TOOL_CALLS tools.
#   3. Retry isolation: on sandbox failure, only code_generator re-runs.
#      Investigation findings are preserved within a pass.
#   4. THREE-TIER quality assessment after each successful sandbox execution:
#        Tier 1: Structural checks (completeness, inf/nan) — deterministic, free
#        Tier 2: Statistical anomaly summary (z-score, IQR) — deterministic, cheap
#        Tier 3: LLM quality assessor agent with investigation tools — semantic
#      Termination is driven by the LLM assessor's recommendation or MAX_PASSES.
#   5. Quality gate runs AFTER sandbox execution, not before code generation.
#   6. Residual issues from the assessor are fed to the next investigator pass.
#
# GRAPH VISUALIZATION:
#
#   START
#     │
#     ▼
#   profiler ◄──────────────────────────────┐
#     │                                     │
#     ▼                                     │
#   investigator ◄──┐                       │
#     │              │                      │
#     ├─[tools?]─► tools                    │
#     │                                     │
#     ▼                                     │
#   code_generator                          │
#     │                                     │
#     ▼                                     │
#   sandbox                                 │
#     │                                     │
#     ├─[error?]─► code_generator (retry)   │
#     │                                     │
#     ▼                                     │
#   quality_gate (Tiers 1&2)                │
#     │                                     │
#     ├─[hard blocker]──────► pass_reset ───┘
#     │                                     │
#     ▼                                     │
#   quality_assessor ◄──┐   (Tier 3)        │
#     │                  │                  │
#     ├─[tools?]─► assessor_tools           │
#     │                                     │
#     ├─[stop]──► autogluon                 │
#     │              │                      │
#     │           answer_agent              │
#     │              │                      │
#     │            END                      │
#     │                                     │
#     └─[continue]──► pass_reset ───────────┘
#
################################################################################

# Library Imports
import io
import os
from contextlib import redirect_stderr
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# Local Imports
from core.pipeline.state import AgentState
from core.schemas import QualityAssessment, AnomalySummary
from core.agents.agents import run_investigator_agent, run_codegen_agent, run_answer_agent
from core.agents.quality_assessor import run_quality_assessor_agent
from core.agents.target_selector import select_target_column
from core.runtime.sandbox import execute_cleaning_plan
from core.runtime.tools import investigation_tools, set_working_df
from core.logger import log_node, log_routing, log_profile_summary
from plugins.profiler import generate_profile, compute_structural_score, compute_anomaly_summary
from plugins.modeller import train_model


################################################################################
# Configuration
################################################################################

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_TOOL_CALLS = 15   # Cap on investigation tool calls to prevent runaway loops
MAX_RETRIES = 3       # Max code generation retries on sandbox failure
MAX_PASSES = 5        # Max cleaning passes before forcing move to modeling


################################################################################
# Node Definitions
################################################################################

def node_profiler(state: AgentState):
    """
    Profiles the input data, generating a JSON description of every column.
    Pure computation — no LLM involved.
    """
    print("--- [1] Profiling Data ---")
    target_col = state.get("target_column")
    verbose = os.environ.get("AUTODS_VERBOSE", "").strip().lower()
    is_verbose_enabled = bool(verbose and verbose != "0")
    if is_verbose_enabled:
        # ydata and related libraries may still emit progress to stderr in some paths.
        # Keep terminal clean in verbose mode because detailed logs are persisted to file.
        with redirect_stderr(io.StringIO()):
            profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)
    else:
        profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)
    log_profile_summary(profile)
    return {"profile": profile}


def node_target_selector(state: AgentState):
    """
    Human-in-the-loop target column selection.
    On pass 0: ranks candidates via LLM, prompts user to confirm.
    On pass > 0: no-op (target persists in state).
    """
    print("--- [1.5] Target Column Selection ---")
    return select_target_column(state)


def node_investigator(state: AgentState):
    """
    LLM agent that analyzes the profile and uses tools to diagnose data issues.
    Produces InvestigationFindings — a structured report of what's wrong.
    Does NOT write any Python code.
    """
    tool_count = state.get("tool_call_count", 0)
    pass_num = state.get("pass_count", 0)
    print(f"--- [2] Investigator (pass: {pass_num}, tool calls so far: {tool_count}) ---")

    # Give tools access to the current DataFrame
    set_working_df(state["working_df"])

    result = run_investigator_agent(state, max_tool_calls=MAX_TOOL_CALLS)

    findings = result.get("investigation_findings")
    if findings:
        log_node("investigator", "findings extracted",
                 violation_count=len(getattr(findings, "violations", [])))

    return result


def node_code_generator(state: AgentState):
    """
    LLM agent that reads InvestigationFindings and writes a CleaningRecipe
    containing executable Python code.
    
    On retry, receives the error message and fixes only the broken step.
    Does NOT re-investigate the data.
    """
    retry = state.get("retry_count", 0)
    print(f"--- [3] Code Generator (Retry: {retry}) ---")
    return run_codegen_agent(state)

def node_sandbox(state: AgentState):
    """
    Executes the cleaning plan in an isolated sandbox.
    Returns cleaned DataFrame on success, or error details on failure.

    On success, increments pass_count and records a compact pass summary
    for multi-pass context.
    """
    print("--- [4] Sandbox Execution ---")
    new_df, new_logs, error = execute_cleaning_plan(
        state["working_df"],
        state["current_plan"]
    )

    updates = {
        "cleaning_history": new_logs,
    }

    # Log per-step results
    for entry in new_logs:
        step_id = entry.step_id if hasattr(entry, "step_id") else entry.get("step_id")
        status = entry.status if hasattr(entry, "status") else entry.get("status")
        operation = entry.operation if hasattr(entry, "operation") else entry.get("operation")
        log_node("sandbox", f"step {step_id}: {status}", operation=operation)

    if error:
        print(f"!!! Execution Error: {error}")
        log_node("sandbox", "FAILED", error=error[:300])
        updates["latest_error"] = error
        updates["retry_count"] = state["retry_count"] + 1
    else:
        print(">>> Execution Successful")
        updates["working_df"] = new_df
        updates["clean_df"] = new_df
        updates["latest_error"] = None
        updates["retry_count"] = 0

        # Multi-pass tracking
        findings = state.get("investigation_findings")
        pass_summary = {
            "pass_number": state.get("pass_count", 0),
            "violations_found": len(findings.violations) if findings and hasattr(findings, "violations") else 0,
            "steps_executed": len(new_logs),
            "rows_after": len(new_df),
        }
        updates["pass_count"] = state.get("pass_count", 0) + 1
        updates["pass_history"] = [pass_summary]
        log_node("sandbox", "pass complete", **pass_summary)

    return updates


def node_quality_gate(state: AgentState):
    """
    Three-tier quality gate:
      Tier 1: Re-profile data and compute structural score (deterministic)
      Tier 2: Compute anomaly summary (deterministic)
      Tier 3: LLM quality assessor agent (launched as separate node)

    This node handles Tiers 1 & 2 and prepares context for Tier 3.
    The assessor agent runs in its own node (node_quality_assessor).
    """
    pass_num = state.get("pass_count", 0)
    print(f"--- [4.5] Quality Gate (pass: {pass_num}) ---")

    # Re-profile the cleaned data
    target_col = state.get("target_column")
    verbose = os.environ.get("AUTODS_VERBOSE", "").strip().lower()
    is_verbose_enabled = bool(verbose and verbose != "0")
    if is_verbose_enabled:
        with redirect_stderr(io.StringIO()):
            profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)
    else:
        profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)

    log_profile_summary(profile)

    # Tier 1: Structural score
    structural_data = compute_structural_score(profile, target_column=target_col)
    log_node("quality_gate", "structural score computed",
             structural_score=structural_data["structural_score"],
             pass_count=pass_num)

    # Tier 2: Anomaly summary
    anomaly_data = compute_anomaly_summary(state["working_df"])
    log_node("quality_gate", "anomaly summary computed",
             total_anomalous_rows=anomaly_data["total_anomalous_rows"],
             columns_with_anomalies=len(anomaly_data["column_summaries"]))

    # Check for structural hard blockers before invoking LLM
    has_hard_blocker = any("target_inf_nan" in f for f in structural_data.get("flags", []))

    return {
        "profile": profile,
        "quality_assessment": {
            "structural_score": structural_data["structural_score"],
            "structural_data": structural_data,
            "anomaly_summary": anomaly_data,
            "has_hard_blocker": has_hard_blocker,
        },
        # Reset assessor state for this pass
        "assessor_messages": ["__RESET__"],
        "assessor_tool_call_count": 0,
    }


def node_quality_assessor(state: AgentState):
    """
    Tier 3: LLM quality assessor agent with investigation tools.

    Receives profile + anomaly summary + cleaning history, investigates
    the data using tools, and produces a structured quality assessment.
    """
    pass_num = state.get("pass_count", 0)
    assessor_tool_count = state.get("assessor_tool_call_count", 0)
    print(f"--- [4.6] Quality Assessor (pass: {pass_num}, tool calls: {assessor_tool_count}) ---")

    # Give tools access to the current DataFrame
    set_working_df(state["working_df"])

    qa = state.get("quality_assessment", {})
    structural_data = qa.get("structural_data", {})
    anomaly_data = qa.get("anomaly_summary", {})

    result = run_quality_assessor_agent(
        state,
        structural_score_data=structural_data,
        anomaly_summary_data=anomaly_data,
    )

    # If assessment is complete, build the full QualityAssessment
    assessment = result.pop("llm_quality_assessment", None)
    if assessment:
        full_assessment = QualityAssessment(
            structural_score=qa.get("structural_score", 0.0),
            anomaly_summary=AnomalySummary(**anomaly_data),
            llm_assessment=assessment,
            recommendation=assessment.recommendation,
            flags=structural_data.get("flags", []),
        ).model_dump()

        result["quality_assessment"] = full_assessment
        result["residual_issues"] = assessment.residual_issues if assessment.residual_issues else None

    return result


def node_pass_reset(state: AgentState):
    """
    Resets per-pass state fields before starting a new cleaning iteration.

    Preserves: working_df, clean_df, cleaning_history, pass_count, pass_history,
               residual_issues (from assessor → next investigator).
    Resets: messages, retry_count, tool_call_count, error state.
    """
    print(f"\n--- [Loop] Starting Pass {state.get('pass_count', 0)} ---")
    log_node("pass_reset", "resetting per-pass state",
             pass_count=state.get("pass_count", 0),
             fields_reset="investigator_messages, codegen_messages, assessor_messages, "
                          "retry_count, tool_call_count, latest_error, "
                          "investigation_findings, current_plan")
    return {
        "investigator_messages": ["__RESET__"],
        "codegen_messages": ["__RESET__"],
        "assessor_messages": ["__RESET__"],
        "retry_count": 0,
        "tool_call_count": 0,
        "assessor_tool_call_count": 0,
        "latest_error": None,
        "previous_findings": state.get("investigation_findings"),
        "investigation_findings": None,
        "current_plan": None,
        # residual_issues is deliberately NOT reset — it flows from assessor to investigator
    }

def node_autogluon(state: AgentState):
    """Trains an ML model using AutoGluon on the cleaned data."""
    print("--- [5] AutoGluon Modeling ---")

    # Prefer human-confirmed target column; fall back to investigator's choice
    target_col = state.get("target_column")
    if not target_col:
        findings = state.get("investigation_findings")
        if findings:
            target_data = (
                findings.model_dump() if hasattr(findings, "model_dump") else findings
            )
            target_col = target_data.get("target_column")

    # Extract task_type hint from investigation findings
    findings = state.get("investigation_findings")
    task_type = None
    if findings:
        task_type = (
            findings.task_type if hasattr(findings, "task_type")
            else findings.get("task_type")
        )

    metadata = train_model(
        state["clean_df"],
        state["user_query"],
        target_column=target_col,
        task_type=task_type,
        output_dir=str(REPO_ROOT / "output"),
    )

    log_node("autogluon", "modeling complete",
             error=metadata.get("error"),
             best_model=metadata.get("best_model"))

    return {"model_metadata": metadata}


def node_answer(state: AgentState):
    """
    LLM agent that generates a business-friendly answer using model results
    and data quality caveats from the investigation.
    """
    print("--- [6] Final Answer ---")
    answer = run_answer_agent(state)
    return {"final_answer": answer}


################################################################################
# Routing Logic
################################################################################

def _route_tool_loop(state, messages_key, count_key, max_calls, tool_dest, log_name):
    """
    Shared routing logic for tool-calling agent loops.

    Checks the latest AI message for tool calls. Returns tool_dest if the
    agent wants more tools and is under the limit, None otherwise.
    """
    for msg in reversed(state.get(messages_key, [])):
        if isinstance(msg, AIMessage):
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            under_limit = state.get(count_key, 0) < max_calls

            if has_tool_calls and under_limit:
                log_routing(log_name, tool_dest,
                            has_tool_calls=True,
                            tool_call_count=state.get(count_key, 0))
                return tool_dest

            if has_tool_calls and not under_limit:
                print(f"!!! Tool call limit reached ({max_calls}). "
                      f"Forcing {log_name.replace('route_', '')} to finalize.")
            break
    return None


def route_investigator(state: AgentState):
    """
    After the investigator runs, decide whether to call tools or proceed to cleaning.
    """
    result = _route_tool_loop(
        state, "investigator_messages", "tool_call_count",
        MAX_TOOL_CALLS, "tools", "route_investigator"
    )
    if result:
        return result

    pass_count = state.get("pass_count", 0)
    findings = state.get("investigation_findings")
    violations = getattr(findings, "violations", []) if findings else []

    log_routing("route_investigator", "code_generator",
                pass_count=pass_count,
                violation_count=len(violations))
    return "code_generator"


def route_sandbox(state: AgentState):
    """
    After sandbox execution, decide next step.

    - Error + retries left → "retry" (back to code_generator)
    - Error + retries exhausted → "failed" (END)
    - Success → "evaluate" (quality_gate re-profiles and checks score)
    """
    latest_error = state.get("latest_error")
    retry_count = state.get("retry_count", 0)
    pass_count = state.get("pass_count", 0)

    if latest_error:
        if retry_count > MAX_RETRIES:
            print(f"!!! Max retries ({MAX_RETRIES}) reached. Stopping.")
            log_routing("route_sandbox", "failed",
                        retry_count=retry_count, pass_count=pass_count,
                        error=latest_error[:200])
            return "failed"
        log_routing("route_sandbox", "retry",
                    retry_count=retry_count, pass_count=pass_count,
                    error=latest_error[:200])
        return "retry"

    log_routing("route_sandbox", "evaluate", pass_count=pass_count)
    return "evaluate"


def route_quality_gate(state: AgentState):
    """
    After Tiers 1 & 2, decide whether to invoke the LLM assessor or skip.

    - Max passes reached → skip LLM, go straight to modeling
    - Structural hard blockers (with passes remaining) → skip LLM, loop back
    - Otherwise → invoke assessor for semantic evaluation
    """
    qa = state.get("quality_assessment", {})
    has_hard_blocker = qa.get("has_hard_blocker", False)
    pass_count = state.get("pass_count", 0)

    if pass_count >= MAX_PASSES:
        print(f">>> Max passes ({MAX_PASSES}) reached. Skipping LLM assessment, moving to modeling.")
        log_routing("route_quality_gate", "done",
                    reason="max_passes", pass_count=pass_count)
        return "done"

    if has_hard_blocker:
        print(f">>> Quality gate: structural hard blocker detected. Skipping LLM assessment.")
        log_routing("route_quality_gate", "next_pass",
                    reason="hard_blocker", pass_count=pass_count)
        return "next_pass"

    log_routing("route_quality_gate", "assess",
                reason="invoke_assessor", pass_count=pass_count)
    return "assess"


def route_quality_assessor(state: AgentState):
    """
    After the quality assessor runs, decide next step:
      - Wants more tools → "assessor_tools"
      - Assessment complete, recommends stop → "done"
      - Assessment complete, recommends continue → "next_pass"
      - Max passes reached → "done" (safety valve)
    """
    from core.agents.quality_assessor import MAX_ASSESSOR_TOOL_CALLS

    result = _route_tool_loop(
        state, "assessor_messages", "assessor_tool_call_count",
        MAX_ASSESSOR_TOOL_CALLS, "assessor_tools", "route_quality_assessor"
    )
    if result:
        return result

    # Assessment complete — route based on recommendation
    pass_count = state.get("pass_count", 0)
    qa = state.get("quality_assessment", {})
    llm_assessment = qa.get("llm_assessment")
    recommendation = qa.get("recommendation", "continue_cleaning")
    llm_score = llm_assessment.get("score") if isinstance(llm_assessment, dict) else None

    if pass_count >= MAX_PASSES:
        print(f">>> Max passes ({MAX_PASSES}) reached (LLM score={llm_score}). "
              f"Moving to modeling.")
        log_routing("route_quality_assessor", "done",
                    reason="max_passes", llm_score=llm_score,
                    pass_count=pass_count)
        return "done"

    if recommendation == "stop_cleaning":
        print(f">>> Quality assessor recommends STOP (score={llm_score}). "
              f"Moving to modeling.")
        log_routing("route_quality_assessor", "done",
                    reason="assessor_stop", llm_score=llm_score,
                    pass_count=pass_count)
        return "done"

    print(f">>> Quality assessor recommends CONTINUE (score={llm_score}). "
          f"Starting pass {pass_count + 1}.")
    log_routing("route_quality_assessor", "next_pass",
                llm_score=llm_score, pass_count=pass_count,
                residual_issues=len(qa.get("llm_assessment", {}).get("residual_issues", [])))
    return "next_pass"


################################################################################
# Graph Assembly
#
# FLOW (three-tier quality assessment):
#   START → profiler → target_selector → investigator ⟲ tools →
#           code_generator → sandbox → [error?] → retry/fail
#                                     → [success] → quality_gate (Tiers 1&2)
#                                                    → [hard blocker?] → pass_reset
#                                                    → quality_assessor ⟲ assessor_tools (Tier 3)
#                                                       → [stop?] → autogluon → answer → END
#                                                       → [continue?] → pass_reset → profiler …
################################################################################

workflow = StateGraph(AgentState)

# --- Nodes ---
workflow.add_node("profiler",            node_profiler)
workflow.add_node("target_selector",     node_target_selector)
workflow.add_node("investigator",        node_investigator)
workflow.add_node("tools",               ToolNode(investigation_tools, messages_key="investigator_messages"))
workflow.add_node("code_generator",      node_code_generator)
workflow.add_node("sandbox",             node_sandbox)
workflow.add_node("quality_gate",        node_quality_gate)
workflow.add_node("quality_assessor",    node_quality_assessor)
workflow.add_node("assessor_tools",      ToolNode(investigation_tools, messages_key="assessor_messages"))
workflow.add_node("pass_reset",          node_pass_reset)
workflow.add_node("autogluon",           node_autogluon)
workflow.add_node("answer_agent",        node_answer)

# --- Edges ---
# Linear flow: START → profiler → target_selector → investigator
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "target_selector")
workflow.add_edge("target_selector", "investigator")

# Investigator: tool loop or proceed to code_generator
workflow.add_conditional_edges(
    "investigator",
    route_investigator,
    {
        "tools": "tools",
        "code_generator": "code_generator",
    }
)
workflow.add_edge("tools", "investigator")  # Tool results go back to investigator

# Code generator → sandbox (always)
workflow.add_edge("code_generator", "sandbox")

# Sandbox: retry on error, or check quality on success
workflow.add_conditional_edges(
    "sandbox",
    route_sandbox,
    {
        "retry": "code_generator",       # Only re-generate code, not re-investigate
        "evaluate": "quality_gate",      # Success: re-profile and check score
        "failed": END,
    }
)

# Quality gate (Tiers 1&2): check for hard blockers or invoke assessor
workflow.add_conditional_edges(
    "quality_gate",
    route_quality_gate,
    {
        "assess": "quality_assessor",    # No hard blockers → invoke LLM assessor
        "next_pass": "pass_reset",       # Hard blocker → skip LLM, loop back
        "done": "autogluon",             # Max passes exhausted → skip LLM, model
    }
)

# Quality assessor (Tier 3): tool loop, or route based on assessment
workflow.add_conditional_edges(
    "quality_assessor",
    route_quality_assessor,
    {
        "assessor_tools": "assessor_tools",  # Assessor wants more data
        "done": "autogluon",                 # Assessor says stop or max passes
        "next_pass": "pass_reset",           # Assessor says continue
    }
)
workflow.add_edge("assessor_tools", "quality_assessor")  # Tool results go back to assessor

# Pass reset → profiler (loop back)
workflow.add_edge("pass_reset", "profiler")

# Linear flow: autogluon → answer → END
workflow.add_edge("autogluon", "answer_agent")
workflow.add_edge("answer_agent", END)

# --- Compile ---
app = workflow.compile()