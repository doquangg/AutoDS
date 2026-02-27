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

from core.state import AgentState
from core.schemas import InvestigationFindings, CleaningRecipe
from core.tools import investigation_tools


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
            base_url=os.environ.get("OPENAI_API_BASE"),  # None = use OpenAI
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _llm_cache[model]


def get_investigator_llm() -> ChatOpenAI:
    """Returns the LLM for the investigator agent (default: gpt-5-2025-08-07)."""
    return _get_llm(os.environ.get("INVESTIGATOR_MODEL", "gpt-5-2025-08-07"))


def get_codegen_llm() -> ChatOpenAI:
    """Returns the LLM for the code generator agent (default: gpt-4.1-2025-04-14)."""
    return _get_llm(os.environ.get("CODEGEN_MODEL", "gpt-4.1-2025-04-14"))


def get_answer_llm() -> ChatOpenAI:
    """Returns the LLM for the answer agent (default: gpt-4.1-2025-04-14)."""
    return _get_llm(os.environ.get("ANSWER_MODEL", "gpt-4.1-2025-04-14"))


################################################################################
# Agent 1: Investigator
#
# INPUT:  profile (DatasetProfile) + user_query + tool access
# OUTPUT: investigation_findings (InvestigationFindings)
#
# This agent ONLY diagnoses problems. It does NOT write Python code.
# It can call tools to inspect the data more deeply.
################################################################################

# FIXME (#19):
# Prompts shouldn't live here. Also, consider restructuring these prompts;
# they were vibecoded. Update script to read prompt from some directory.
INVESTIGATOR_SYSTEM_PROMPT = """\
You are a senior data quality analyst. Your job is to examine a dataset profile \
and identify every semantic data quality issue that could corrupt a machine \
learning model.

You will receive:
1. The user's question (what they want to predict/answer)
2. A statistical profile of every column in the dataset

Your task:
- Identify the TARGET COLUMN that best answers the user's question
- Find ALL semantic violations in the data
- Classify each violation by severity and category
- Provide specific evidence from the profile for each finding
- Suggest plain-English fixes (NOT code — just intent)
- Assess overall data quality (0.0 to 1.0)
- Note any caveats that should accompany the final answer

WHAT TO LOOK FOR:
- Sentinel values masquerading as real data (-1, 0, 999, 9999, "N/A", "Unknown")
  Check: top_frequent_values, min_value, max_value
- Temporal impossibilities (event B before event A)
  Check: use temporal_ordering_check tool
- Cross-column logic errors (impossible combinations)
  Check: use cross_column_frequency tool
- Type mismatches (numeric column storing categories, or vice versa)
  Check: inferred_type vs semantic meaning
- Suspicious distributions (spikes at sentinel values, extreme skew)
  Check: use value_distribution tool
- Systematic missingness (columns null together)
  Check: use null_co_occurrence tool
- High cardinality columns that are likely IDs (should be dropped)
  Check: unique_factor > 0.95
- PII that should not be used as features

USE YOUR TOOLS to verify suspicions. Don't guess — inspect the actual data.
But be efficient: don't call the same tool repeatedly with minor variations.

Do NOT write any Python code. Your job is diagnosis, not treatment.\
"""


def run_investigator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Runs the investigator agent. Returns state updates including 
    investigation_findings and investigator_messages.
    
    The investigator uses tools via LangChain's bind_tools(), which means
    LangGraph's ToolNode handles execution. This function is called 
    repeatedly until the agent stops requesting tools.
    """
    llm = get_investigator_llm()

    # Bind investigation tools so the LLM can call them
    llm_with_tools = llm.bind_tools(
        investigation_tools,
        # Also allow structured output for the final response
    )

    # Build messages for this agent's context
    messages = list(state.get("investigator_messages", []))
    
    # On first call (no messages yet), inject the system prompt and profile
    if not messages:
        profile_json = json.dumps(state["profile"], indent=2, default=str)
        
        messages = [
            SystemMessage(content=INVESTIGATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"USER QUERY: {state['user_query']}\n\n"
                f"DATASET PROFILE ({state['profile']['row_count']} rows):\n"
                f"{profile_json}"
            )),
        ]

    response = llm_with_tools.invoke(messages)
    
    # Return the new message(s) to append to investigator_messages
    updates: Dict[str, Any] = {
        "investigator_messages": [response],
        "tool_call_count": state.get("tool_call_count", 0) + len(
            getattr(response, "tool_calls", []) or []
        ),
    }
    
    # If the agent is done (no tool calls), extract structured findings
    if not getattr(response, "tool_calls", None):
        structured_llm = llm.with_structured_output(InvestigationFindings, method="function_calling")
        findings = structured_llm.invoke(messages + [response])
        updates["investigation_findings"] = findings
    
    return updates


def _parse_investigation_findings(
    response: AIMessage, llm: ChatOpenAI
) -> InvestigationFindings:
    """
    Extract structured InvestigationFindings from the agent's final response.
    Uses with_structured_output if available, otherwise parses JSON from text.
    """
    # Approach 1: Try parsing JSON directly from the response text
    content = response.content
    try:
        # The LLM might wrap JSON in markdown code fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        data = json.loads(content)
        return InvestigationFindings(**data)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    
    # Approach 2: Ask the LLM to reformat its response as structured output
    structured_llm = llm.with_structured_output(InvestigationFindings, method="function_calling")
    findings = structured_llm.invoke([
        SystemMessage(content=(
            "Convert the following data quality analysis into the exact "
            "InvestigationFindings JSON schema. Preserve all information."
        )),
        HumanMessage(content=response.content),
    ])
    return findings


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

# FIXME (#19):
# Prompts shouldn't live here. Also, consider restructuring these prompts;
# they were vibecoded. Update script to read prompt from some directory.
CODEGEN_SYSTEM_PROMPT = """\
You are an expert Python data engineer. Your job is to write a CleaningRecipe: \
an ordered list of executable pandas code steps that clean a DataFrame.

You will receive:
1. Investigation findings describing exactly what's wrong with the data
2. The dataset profile (column types, shapes, value distributions)
3. On retry: the specific error message and the code that failed

RULES:
- Every step must contain valid, executable Python using the variable `df`
- Each step should be atomic: one clear transformation per step
- Steps execute in order. Each step receives the `df` from the previous step.
- Reference the violation_id from findings in your addresses_violation field
- Use the operation categories from OperationType (DROP_COLUMN, DROP_ROWS, etc.)
- Use CUSTOM_CODE for anything that doesn't fit a predefined operation
- Do NOT use StandardScaler or MinMaxScaler — AutoGluon handles scaling internally
- Preserve the target column — never drop or corrupt it
- Be conservative: prefer imputation over dropping rows when possible

CODE STYLE:
- Always reassign: `df = df[df['age'] > 0]` not `df.drop(..., inplace=True)`
- Handle edge cases: check column exists before operating on it
- Use .copy() when creating derived columns from slices
- String operations: use .str accessor, handle NaN with na=False

ON RETRY:
- You will see the error message and the code that caused it
- Fix ONLY the failing step — do not rewrite steps that already succeeded
- Common errors: KeyError (column already dropped/renamed), TypeError (wrong dtype),
  ValueError (invalid values for operation)

You MUST respond with a valid CleaningRecipe JSON object.\
"""


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

    recipe = structured_llm.invoke(messages)
    
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

# FIXME (#19):
# Prompts shouldn't live here. Also, consider restructuring these prompts;
# they were vibecoded. Update script to read prompt from some directory.
ANSWER_SYSTEM_PROMPT = """\
You are a data science consultant presenting results to a business stakeholder.

You will receive:
1. The user's original question
2. Model performance metrics and feature importances
3. Data quality findings (what issues were found and how they were handled)

Your job:
- Answer the user's question directly and concisely
- Lead with the actionable insight
- Mention the model's confidence/accuracy so they know how much to trust it
- Include relevant caveats from data quality findings
  (e.g., "Note: 15% of income values were imputed, which may affect accuracy 
  of income-related predictions")
- If the model performance is poor, say so honestly
- Use business language, not technical jargon

Keep your answer to 2-4 paragraphs maximum.\
"""


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