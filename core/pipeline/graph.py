################################################################################
# LangGraph state machine for the AutoDS pipeline.
#
# FLOW (multi-pass):
#   START → profiler → investigator ⟲ tools → [clean check] →
#           code_generator → sandbox → [loop check] → pass_reset → profiler …
#         → autogluon → answer_agent → END
#
# KEY DESIGN DECISIONS:
#   1. SPLIT: investigator (diagnosis) and code_generator (Python code) are
#      separate agents. Investigation happens ONCE per pass; only code_generator
#      retries on sandbox failure.
#   2. Tool loop cap: investigator can call at most MAX_TOOL_CALLS tools.
#   3. Retry isolation: on sandbox failure, only code_generator re-runs.
#      Investigation findings are preserved within a pass.
#   4. Multi-pass loop: after a successful sandbox execution, the cleaned data
#      is fed back through profiler → investigator for re-examination. The loop
#      terminates when the investigator declares the data clean or MAX_PASSES
#      is reached.
#
# GRAPH VISUALIZATION:
#
#   START
#     │
#     ▼
#   profiler ◄──────────────────┐
#     │                         │
#     ▼                         │
#   investigator ◄──┐           │
#     │              │          │
#     ├─[tools?]─► tools        │
#     │                         │
#     ├─[clean?]──► autogluon   │  (pass > 0 only: data is clean)
#     │                │        │
#     ▼                ▼        │
#   code_generator  answer_agent│
#     │                │        │
#     ▼              END        │
#   sandbox                     │
#     │                         │
#     ├─[error?]─► code_generator (retry, max 3)
#     │                         │
#     ├─[max passes?]─► autogluon
#     │                         │
#     └─[next pass]─► pass_reset┘
#
################################################################################

# Library Imports
import io
import os
from contextlib import redirect_stderr

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# Local Imports
from core.pipeline.state import AgentState
from core.agents.agents import run_investigator_agent, run_codegen_agent, run_answer_agent
from core.agents.target_selector import select_target_column
from core.runtime.sandbox import execute_cleaning_plan
from core.runtime.tools import investigation_tools, set_working_df
from core.logger import log_node, log_routing, log_profile_summary
from plugins.profiler import generate_profile
from plugins.modeller import train_model


################################################################################
# Configuration
################################################################################

MAX_TOOL_CALLS = 30   # Cap on investigation tool calls to prevent runaway loops
MAX_RETRIES = 3       # Max code generation retries on sandbox failure
MAX_PASSES = 5        # Max cleaning passes before forcing move to modeling
CLEAN_SCORE_THRESHOLD = 0.85  # Minimum data_quality_score to consider data clean


################################################################################
# Node Definitions
################################################################################

def node_profiler(state: AgentState):
    """
    Profiles the input data, generating a JSON description of every column.
    Pure computation — no LLM involved.
    """
    print("--- [1] Profiling Data ---")
    verbose = os.environ.get("AUTODS_VERBOSE", "").strip().lower()
    is_verbose_enabled = bool(verbose and verbose != "0")
    if is_verbose_enabled:
        # ydata and related libraries may still emit progress to stderr in some paths.
        # Keep terminal clean in verbose mode because detailed logs are persisted to file.
        with redirect_stderr(io.StringIO()):
            profile = generate_profile(state["working_df"], detailed_profiler=True)
    else:
        profile = generate_profile(state["working_df"], detailed_profiler=True)
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

    On multi-pass re-examination, extracts is_data_clean from findings
    and writes it to state for the router to check.
    """
    tool_count = state.get("tool_call_count", 0)
    pass_num = state.get("pass_count", 0)
    print(f"--- [2] Investigator (pass: {pass_num}, tool calls so far: {tool_count}) ---")

    # Give tools access to the current DataFrame
    set_working_df(state["working_df"])

    result = run_investigator_agent(state, max_tool_calls=MAX_TOOL_CALLS)

    # Extract is_data_clean from findings and propagate to state
    findings = result.get("investigation_findings")
    if findings and hasattr(findings, "is_data_clean"):
        result["is_data_clean"] = findings.is_data_clean
        log_node("investigator", "findings extracted",
                 is_data_clean=findings.is_data_clean,
                 quality_score=getattr(findings, "data_quality_score", None),
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
            "quality_score": findings.data_quality_score if findings and hasattr(findings, "data_quality_score") else None,
            "steps_executed": len(new_logs),
            "rows_after": len(new_df),
        }
        updates["pass_count"] = state.get("pass_count", 0) + 1
        updates["pass_history"] = [pass_summary]
        log_node("sandbox", "pass complete", **pass_summary)

    return updates


def node_pass_reset(state: AgentState):
    """
    Resets per-pass state fields before starting a new cleaning iteration.

    Preserves: working_df, clean_df, cleaning_history, pass_count, pass_history.
    Resets: messages, retry_count, tool_call_count, error state.
    """
    print(f"\n--- [Loop] Starting Pass {state.get('pass_count', 0)} ---")
    log_node("pass_reset", "resetting per-pass state",
             pass_count=state.get("pass_count", 0),
             fields_reset="investigator_messages, codegen_messages, retry_count, "
                          "tool_call_count, latest_error, investigation_findings, current_plan")
    return {
        "investigator_messages": ["__RESET__"],
        "codegen_messages": ["__RESET__"],
        "retry_count": 0,
        "tool_call_count": 0,
        "latest_error": None,
        "investigation_findings": None,
        "current_plan": None,
    }

# FIXME (#3): Need to actually implement AutoGluon
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
    
    metadata = train_model(
        state["clean_df"], 
        state["user_query"],
        target_column=target_col  # Pass explicit target instead of guessing
    )
    return {"model_metadata": metadata}


def node_answer(state: AgentState):
    """
    LLM agent that generates a business-friendly answer using model results
    and data quality caveats from the investigation.

    TODO: Currently stubbed — returns placeholder until AutoGluon is implemented.
    Uncomment the real implementation when model_metadata contains real results.
    """
    print("--- [6] Final Answer [STUB] ---")
    # ans = run_answer_agent(state)
    return {"final_answer": "Answer agent stubbed — AutoGluon not yet implemented."}


################################################################################
# Routing Logic
################################################################################

def route_investigator(state: AgentState):
    """
    After the investigator runs, decide whether to call tools, clean, or skip.

    1. If the investigator requested tools and we're under the limit → "tools"
    2. If this is pass > 0 and data is clean → "autogluon" (skip cleaning)
    3. Otherwise → "code_generator" (proceed with cleaning)

    Clean-check uses a layered approach:
      - Investigator's is_data_clean boolean (primary signal)
      - data_quality_score >= threshold with no CRITICAL violations (secondary)
    """
    # --- Tool loop check ---
    for msg in reversed(state.get("investigator_messages", [])):
        if isinstance(msg, AIMessage):
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            under_limit = state.get("tool_call_count", 0) < MAX_TOOL_CALLS

            if has_tool_calls and under_limit:
                log_routing("route_investigator", "tools",
                            has_tool_calls=True,
                            tool_call_count=state.get("tool_call_count", 0),
                            max_tool_calls=MAX_TOOL_CALLS)
                return "tools"

            if has_tool_calls and not under_limit:
                print(f"!!! Tool call limit reached ({MAX_TOOL_CALLS}). "
                      f"Forcing investigator to finalize.")
            break

    # --- Multi-pass clean check (pass > 0 only) ---
    pass_count = state.get("pass_count", 0)
    is_clean = state.get("is_data_clean", False)
    findings = state.get("investigation_findings")
    score = getattr(findings, "data_quality_score", 0) if findings else 0
    violations = getattr(findings, "violations", []) if findings else []
    critical_count = len([v for v in violations if v.severity == "CRITICAL"])

    if pass_count > 0:
        # Primary signal: investigator explicitly says clean
        if is_clean:
            print(f">>> Investigator declares data clean on pass {pass_count}. "
                  f"Moving to modeling.")
            log_routing("route_investigator", "autogluon",
                        reason="is_data_clean=True", pass_count=pass_count,
                        quality_score=score, critical_count=critical_count)
            return "autogluon"

        # Secondary signal: quality score + no CRITICAL violations
        if findings and score >= CLEAN_SCORE_THRESHOLD and critical_count == 0:
            print(f">>> Quality score {score:.2f} >= {CLEAN_SCORE_THRESHOLD} "
                  f"with no CRITICAL violations. Moving to modeling.")
            log_routing("route_investigator", "autogluon",
                        reason="score_threshold", pass_count=pass_count,
                        quality_score=score, critical_count=critical_count)
            return "autogluon"

    log_routing("route_investigator", "code_generator",
                pass_count=pass_count, is_data_clean=is_clean,
                quality_score=score, violation_count=len(violations),
                critical_count=critical_count)
    return "code_generator"


def route_sandbox(state: AgentState):
    """
    After sandbox execution, decide next step.

    - Error + retries left → "retry" (back to code_generator)
    - Error + retries exhausted → "failed" (END)
    - Success + max passes reached → "done" (autogluon)
    - Success + under max passes → "next_pass" (pass_reset → profiler loop)
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

    # Success path: check if we should loop
    if pass_count >= MAX_PASSES:
        print(f">>> Max passes ({MAX_PASSES}) reached. Moving to modeling.")
        log_routing("route_sandbox", "done",
                    pass_count=pass_count, max_passes=MAX_PASSES)
        return "done"

    log_routing("route_sandbox", "next_pass",
                pass_count=pass_count, max_passes=MAX_PASSES)
    return "next_pass"


################################################################################
# Graph Assembly
################################################################################

workflow = StateGraph(AgentState)

# --- Nodes ---
workflow.add_node("profiler",          node_profiler)
workflow.add_node("target_selector",   node_target_selector)
workflow.add_node("investigator",      node_investigator)
workflow.add_node("tools",          ToolNode(investigation_tools, messages_key="investigator_messages"))
workflow.add_node("code_generator", node_code_generator)
workflow.add_node("sandbox",        node_sandbox)
workflow.add_node("pass_reset",     node_pass_reset)
workflow.add_node("autogluon",      node_autogluon)
workflow.add_node("answer_agent",   node_answer)

# --- Edges ---
# Linear flow: START → profiler → target_selector → investigator
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "target_selector")
workflow.add_edge("target_selector", "investigator")

# Investigator: tool loop, clean check, or proceed to code_generator
workflow.add_conditional_edges(
    "investigator",
    route_investigator,
    {
        "tools": "tools",
        "code_generator": "code_generator",
        "autogluon": "autogluon",       # Multi-pass: data is clean, skip cleaning
    }
)
workflow.add_edge("tools", "investigator")  # Tool results go back to investigator

# Code generator → sandbox (always)
workflow.add_edge("code_generator", "sandbox")

# Sandbox: retry, loop back for another pass, or proceed to modeling
workflow.add_conditional_edges(
    "sandbox",
    route_sandbox,
    {
        "retry": "code_generator",      # Only re-generate code, not re-investigate
        "next_pass": "pass_reset",      # Multi-pass: loop back for re-examination
        "done": "autogluon",            # Multi-pass: max passes reached
        "failed": END,
    }
)

# Pass reset → profiler (loop back)
workflow.add_edge("pass_reset", "profiler")

# Linear flow: autogluon → answer → END
workflow.add_edge("autogluon", "answer_agent")
workflow.add_edge("answer_agent", END)

# --- Compile ---
app = workflow.compile()