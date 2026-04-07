################################################################################
# LangGraph state machine for the AutoDS pipeline.
#
# FLOW (multi-pass):
#   START → profiler → target_selector → investigator ⟲ tools →
#           code_generator → sandbox → [error?] → retry/fail
#                                     → [success] → re_profile → route_post_clean
#                                                    → [done] → autogluon → answer → END
#                                                    → [next_pass] → pass_reset → profiler …
#
# KEY DESIGN DECISIONS:
#   1. SPLIT: investigator (diagnosis) and code_generator (Python code) are
#      separate agents. Investigation happens ONCE per pass; only code_generator
#      retries on sandbox failure.
#   2. Tool loop cap: investigator can call at most MAX_TOOL_CALLS tools.
#   3. Retry isolation: on sandbox failure, only code_generator re-runs.
#      Investigation findings are preserved within a pass.
#   4. Termination: after each successful sandbox execution, the pipeline
#      re-profiles the data. If the investigator found no CRITICAL violations
#      (or MAX_PASSES is reached), we move to modeling. No LLM is involved
#      in the termination decision.
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
#   re_profile                              │
#     │                                     │
#     ├─[done]──► autogluon                 │
#     │              │                      │
#     │           answer_agent              │
#     │              │                      │
#     │            END                      │
#     │                                     │
#     └─[next_pass]──► pass_reset ──────────┘
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
from core.agents.agents import run_investigator_agent, run_codegen_agent, run_answer_agent
from core.agents.target_selector import select_target_column
from core.runtime.sandbox import execute_cleaning_plan
from core.runtime.tools import investigation_tools, set_working_df
from core.logger import log_node, log_routing, log_profile_summary
from plugins.profiler import generate_profile
from plugins.modeller import train_model
from plugins.feature_engineering import engineer_features


################################################################################
# Configuration
################################################################################

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_TOOL_CALLS = 20   # Cap on investigation tool calls to prevent runaway loops
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


def node_re_profile(state: AgentState):
    """
    Re-profiles cleaned data after a successful sandbox execution.

    Regenerates the profile so the next investigator pass (if one happens)
    sees the current state of the data.
    """
    pass_num = state.get("pass_count", 0)
    print(f"--- [4.5] Re-Profile (pass: {pass_num}) ---")

    target_col = state.get("target_column")
    verbose = os.environ.get("AUTODS_VERBOSE", "").strip().lower()
    is_verbose_enabled = bool(verbose and verbose != "0")
    if is_verbose_enabled:
        with redirect_stderr(io.StringIO()):
            profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)
    else:
        profile = generate_profile(state["working_df"], detailed_profiler=True, target_column=target_col)

    log_profile_summary(profile)

    return {"profile": profile}


def node_pass_reset(state: AgentState):
    """
    Resets per-pass state fields before starting a new cleaning iteration.

    Preserves: working_df, clean_df, cleaning_history, pass_count, pass_history.
    Resets: messages, retry_count, tool_call_count, error state.
    """
    print(f"\n--- [Loop] Starting Pass {state.get('pass_count', 0)} ---")
    log_node("pass_reset", "resetting per-pass state",
             pass_count=state.get("pass_count", 0),
             fields_reset="investigator_messages, codegen_messages, "
                          "retry_count, tool_call_count, latest_error, "
                          "investigation_findings, current_plan")
    return {
        "investigator_messages": ["__RESET__"],
        "codegen_messages": ["__RESET__"],
        "retry_count": 0,
        "tool_call_count": 0,
        "latest_error": None,
        "previous_findings": state.get("investigation_findings"),
        "investigation_findings": None,
        "current_plan": None,
    }


def node_feature_engineering(state: AgentState):
    """
    Feature engineering stage (post-cleaning, pre-modeling).

    Deterministic + bounded:
    - Uses plugins.feature_engineering.engineer_features
    - Capped by MAX_FE_ROUNDS (default 3)
    - Tracks fe_round + fe_history in state
    """
    print("--- [5] Feature Engineering ---")

    MAX_FE_ROUNDS = 3  # default cap; can be made configurable later
    fe_round = state.get("fe_round", 0)

    if fe_round >= MAX_FE_ROUNDS:
        print(f">>> FE: max rounds ({MAX_FE_ROUNDS}) reached. Skipping FE.")
        return {"engineered_df": (state.get("clean_df") or state.get("working_df")), "clean_df": (state.get("clean_df") or state.get("working_df")), "fe_round": fe_round}

    df_in = state.get("clean_df") or state.get("working_df")
    target_col = state.get("target_column")

    result = engineer_features(df_in, target_column=target_col, round_id=fe_round)

    history_entry = {
        "round": fe_round,
        "new_features": result.new_features,
        "dropped_features": result.dropped_features,
        "notes": result.notes,
    }

    print(f">>> FE round {fe_round}: added {len(result.new_features)} features")

    return {
        "engineered_df": result.df,
        "clean_df": result.df,   # downstream (AutoGluon) continues using clean_df
        "fe_round": fe_round + 1,
        "fe_history": [history_entry],
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
    - Success → "evaluate" (re_profile regenerates data profile)
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


def route_post_clean(state: AgentState):
    """
    After re-profiling, decide whether to continue cleaning or move to modeling.

    - Max passes reached → "done" (hard cap)
    - No CRITICAL violations found → "done" (investigator self-terminates)
    - CRITICAL violations remain → "next_pass" (loop back)
    """
    pass_count = state.get("pass_count", 0)

    if pass_count >= MAX_PASSES:
        print(f">>> Max passes ({MAX_PASSES}) reached. Moving to modeling.")
        log_routing("route_post_clean", "done",
                    reason="max_passes", pass_count=pass_count)
        return "done"

    findings = state.get("investigation_findings")
    critical = [v for v in findings.violations if v.severity == "CRITICAL"] if findings else []

    if len(critical) == 0:
        print(f">>> No CRITICAL violations found. Moving to modeling.")
        log_routing("route_post_clean", "done",
                    reason="no_critical_violations", pass_count=pass_count)
        return "done"

    print(f">>> {len(critical)} CRITICAL violation(s) remain. Starting pass {pass_count + 1}.")
    log_routing("route_post_clean", "next_pass",
                critical_count=len(critical), pass_count=pass_count)
    return "next_pass"


################################################################################
# Graph Assembly
#
# FLOW:
#   START → profiler → target_selector → investigator ⟲ tools →
#           code_generator → sandbox → [error?] → retry/fail
#                                     → [success] → re_profile
#                                                    → [done] → autogluon → answer → END
#                                                    → [next_pass] → pass_reset → profiler …
################################################################################

workflow = StateGraph(AgentState)

# --- Nodes ---
workflow.add_node("profiler",            node_profiler)
workflow.add_node("target_selector",     node_target_selector)
workflow.add_node("investigator",        node_investigator)
workflow.add_node("tools",               ToolNode(investigation_tools, messages_key="investigator_messages"))
workflow.add_node("code_generator",      node_code_generator)
workflow.add_node("sandbox",             node_sandbox)
workflow.add_node("re_profile",          node_re_profile)
workflow.add_node("pass_reset",          node_pass_reset)
workflow.add_node("feature_engineering", node_feature_engineering)
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

# Sandbox: retry on error, or re-profile on success
workflow.add_conditional_edges(
    "sandbox",
    route_sandbox,
    {
        "retry": "code_generator",       # Only re-generate code, not re-investigate
        "evaluate": "re_profile",        # Success: re-profile cleaned data
        "failed": END,
    }
)

# Re-profile → route based on investigation findings
workflow.add_conditional_edges(
    "re_profile",
    route_post_clean,
    {
        "done": "feature_engineering",   # No critical violations or max passes
        "next_pass": "pass_reset",       # Critical violations remain
    }
)

# Pass reset → profiler (loop back)
workflow.add_edge("pass_reset", "profiler")

workflow.add_edge("feature_engineering", "autogluon")

# Linear flow: autogluon → answer → END
workflow.add_edge("autogluon", "answer_agent")
workflow.add_edge("answer_agent", END)

# --- Compile ---
app = workflow.compile()