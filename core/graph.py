################################################################################
# LangGraph state machine for the AutoDS pipeline.
#
# FLOW:
#   START → profiler → investigator ⟲ tools → code_generator → sandbox 
#         → autogluon → answer_agent → END
#
# KEY CHANGES FROM ORIGINAL:
#   1. SPLIT: investigator (diagnosis) and code_generator (Python code) are 
#      separate agents. Investigation happens ONCE; only code_generator retries.
#   2. Tool loop cap: investigator can call at most MAX_TOOL_CALLS tools.
#   3. Retry isolation: on sandbox failure, only code_generator re-runs.
#      Investigation findings are preserved and not re-generated.
#   4. Routing fix: route_investigator checks last AIMessage, not last message.
#
# GRAPH VISUALIZATION:
#
#   START
#     │
#     ▼
#   profiler
#     │
#     ▼
#   investigator ◄──┐
#     │              │
#     ├─[tools?]─► tools
#     │
#     ▼
#   code_generator ◄──┐
#     │                │
#     ▼                │
#   sandbox            │
#     │                │
#     ├─[error?]───────┘  (retry: only code_generator, not investigator)
#     │
#     ▼
#   autogluon
#     │
#     ▼
#   answer_agent
#     │
#     ▼
#   END
#
################################################################################

# Library Imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# Local Imports
from core.state import AgentState
from core.agents import run_investigator_agent, run_codegen_agent, run_answer_agent
from core.sandbox import execute_cleaning_plan
from core.tools import investigation_tools, set_working_df
from plugins.profiler import generate_profile
from plugins.modeller import train_model


################################################################################
# Configuration
################################################################################

MAX_TOOL_CALLS = 10   # Cap on investigation tool calls to prevent runaway loops
MAX_RETRIES = 3       # Max code generation retries on sandbox failure


################################################################################
# Node Definitions
################################################################################

def node_profiler(state: AgentState):
    """
    Profiles the input data, generating a JSON description of every column.
    Pure computation — no LLM involved.
    """
    print("--- [1] Profiling Data ---")
    return {"profile": generate_profile(state["working_df"])}


def node_investigator(state: AgentState):
    """
    LLM agent that analyzes the profile and uses tools to diagnose data issues.
    Produces InvestigationFindings — a structured report of what's wrong.
    Does NOT write any Python code.
    """
    tool_count = state.get("tool_call_count", 0)
    print(f"--- [2] Investigator (tool calls so far: {tool_count}) ---")
    
    # Give tools access to the current DataFrame
    set_working_df(state["working_df"])
    
    return run_investigator_agent(state)


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

# FIXME (#4): Need to actually implement sandbox
def node_sandbox(state: AgentState):
    """
    Executes the cleaning plan in an isolated sandbox.
    Returns cleaned DataFrame on success, or error details on failure.
    """
    print("--- [4] Sandbox Execution ---")
    new_df, new_logs, error = execute_cleaning_plan(
        state["working_df"],
        state["current_plan"]
    )

    updates = {
        "cleaning_history": new_logs,
    }

    if error:
        print(f"!!! Execution Error: {error}")
        updates["latest_error"] = error
        updates["retry_count"] = state["retry_count"] + 1
    else:
        print(">>> Execution Successful")
        updates["working_df"] = new_df
        updates["clean_df"] = new_df
        updates["latest_error"] = None
        updates["retry_count"] = 0

    return updates

# FIXME (#3): Need to actually implement AutoGluon
def node_autogluon(state: AgentState):
    """Trains an ML model using AutoGluon on the cleaned data."""
    print("--- [5] AutoGluon Modeling ---")
    
    # Use the target column identified by the investigator, if available
    findings = state.get("investigation_findings")
    target_col = None
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
    """
    print("--- [6] Final Answer ---")
    ans = run_answer_agent(state)
    return {"final_answer": ans}


################################################################################
# Routing Logic
################################################################################

def route_investigator(state: AgentState):
    """
    After the investigator runs, decide whether to call tools or move on.
    
    FIX: Checks the last AIMessage (not just last message), because ToolNode
    appends ToolMessages which don't have tool_calls.
    
    Also enforces MAX_TOOL_CALLS to prevent infinite investigation loops.
    """
    # Find the last AI message
    for msg in reversed(state.get("investigator_messages", [])):
        if isinstance(msg, AIMessage):
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            under_limit = state.get("tool_call_count", 0) < MAX_TOOL_CALLS
            
            if has_tool_calls and under_limit:
                return "tools"
            
            if has_tool_calls and not under_limit:
                print(f"!!! Tool call limit reached ({MAX_TOOL_CALLS}). "
                      f"Forcing investigator to finalize.")
            
            return "code_generator"
    
    # Fallback: no AI message found (shouldn't happen)
    return "code_generator"


def route_sandbox(state: AgentState):
    """
    After sandbox execution, decide whether to retry code generation or proceed.
    
    KEY: On retry, routes to code_generator (NOT back to investigator).
    Investigation findings are stable; only the code needs fixing.
    """
    if state.get("latest_error"):
        if state["retry_count"] > MAX_RETRIES:
            print(f"!!! Max retries ({MAX_RETRIES}) reached. Stopping.")
            return "failed"
        return "retry"
    return "success"


################################################################################
# Graph Assembly
################################################################################

workflow = StateGraph(AgentState)

# --- Nodes ---
workflow.add_node("profiler",       node_profiler)
workflow.add_node("investigator",   node_investigator)
workflow.add_node("tools",          ToolNode(investigation_tools))
workflow.add_node("code_generator", node_code_generator)
workflow.add_node("sandbox",        node_sandbox)
workflow.add_node("autogluon",      node_autogluon)
workflow.add_node("answer_agent",   node_answer)

# --- Edges ---
# Linear flow: START → profiler → investigator
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "investigator")

# Investigator tool loop: investigator ⟲ tools, then → code_generator
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

# Sandbox retry loop: on error → code_generator (NOT investigator)
workflow.add_conditional_edges(
    "sandbox",
    route_sandbox,
    {
        "retry": "code_generator",   # Only re-generate code, not re-investigate
        "success": "autogluon",
        "failed": END,
    }
)

# Linear flow: autogluon → answer → END
workflow.add_edge("autogluon", "answer_agent")
workflow.add_edge("answer_agent", END)

# --- Compile ---
app = workflow.compile()