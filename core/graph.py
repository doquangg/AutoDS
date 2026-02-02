################################################################################
# This file documents the exact layout of the logic of the agent, including what
# nodes and edges define our agentic workflow.
################################################################################\

# Library Imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Local Imports
from core.state import AgentState
from core.agent import run_investigating_agent, run_answer_agent
from core.sandbox import execute_cleaning_plan
from plugins.profiler import generate_profile
from plugins.modeller import train_model
from core.tools import investigation_tools

################################################################################
# Node Definitions
################################################################################

# FIXME: Need to actually define and implement generate_profile (#1)
# This node profiles the input data, generating a .json describing every column
# in the dataset. See schemas.py for the exact format of returned .json.
def node_profiler(state: AgentState):
    print("--- [1] Profiling Data ---")
    return {"profile": generate_profile(state["working_df"])}

# FIXME: Need to actually define and implement agent for investigation (#9, #10)
# This node triggers the LLM to investgate the profiled input data.
def node_investigator(state: AgentState):
    print(f"--- [2] Investigator (Retry: {state.get('retry_count', 0)}) ---")
    return run_investigating_agent(state)

# FIXME: Need to actually build sandbox (#4)
# This node is the sandbox, and executes agent-generated code, appending the 
# results to the updates dict as it goes.
def node_sandbox(state: AgentState):
    print("--- [3] Sandbox Execution ---")
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

# FIXME: Need to actually investigate AutoGluon (#3)
def node_autogluon(state: AgentState):
    print("--- [4] AutoGluon Modeling ---")
    metadata = train_model(state["clean_df"], state["user_query"])
    return {"model_metadata": metadata}

# FIXME: Need to actually define and implement agent for investigation (#9, #10)
# This node triggers the LLM to generate the final answer, given our trained
# AutoGluon model, the provided question, and any necessary context.
def node_answer(state: AgentState):
    print("--- [5] Final Answer ---")
    ans = run_answer_agent(state)
    return {"final_answer": ans}

################################################################################
# Routing Logic
################################################################################

# Determines next step after investigator node; either goes to tools or sandbox
def route_investigator(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "sandbox"

# Handles error handling and retires after the sandbox node
def route_sandbox(state: AgentState):
    if state["latest_error"]:
        if state["retry_count"] > 3: # FIXME Should this be a hyperparameter?
            print("!!! Max retries reached. Stopping.")
            return "failed" 
        return "retry"
    return "success"

################################################################################
# Graph
################################################################################
workflow = StateGraph(AgentState)

# Nodes (Actions we can take, these return things)
workflow.add_node("profiler", node_profiler)
workflow.add_node("investigator", node_investigator)
workflow.add_node("tools", ToolNode(investigation_tools))
workflow.add_node("sandbox", node_sandbox)
workflow.add_node("autogluon", node_autogluon)
workflow.add_node("answer_agent", node_answer)

# Edges (how information flows between nodes)
workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "investigator")
workflow.add_conditional_edges(
    "investigator",
    route_investigator,
    {
        "tools": "tools",
        "sandbox": "sandbox"
    }
)
workflow.add_edge("tools", "investigator")
workflow.add_conditional_edges(
    "sandbox",
    route_sandbox,
    {
        "retry": "investigator",   
        "success": "autogluon",    
        "failed": END              
    }
)
workflow.add_edge("autogluon", "answer_agent")
workflow.add_edge("answer_agent", END)

app = workflow.compile()