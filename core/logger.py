"""
Centralized verbose logging for the AutoDS pipeline.

Toggle via environment variable:
    AUTODS_VERBOSE=1 python scripts/run_graph.py

Logs go to stderr so they don't interfere with stdout progress output.
When AUTODS_VERBOSE is not set (or falsy), only WARNING+ messages appear.
"""

import json
import logging
import os
import sys


logger = logging.getLogger("autods")

# Set to True when AUTODS_VERBOSE=full — disables all truncation.
_full_verbose: bool = False


def setup_logger(log_file: str | None = None) -> None:
    """
    Configure the autods logger based on AUTODS_VERBOSE env var.
    Call once at startup before any pipeline work.

    If log_file is provided, verbose logs are also written there whenever
    AUTODS_VERBOSE enables verbose mode.

    Values:
        AUTODS_VERBOSE=1     — verbose mode (truncation enabled)
        AUTODS_VERBOSE=full  — verbose mode, no truncation
        unset / 0            — warnings only
    """
    global _full_verbose
    verbose = os.environ.get("AUTODS_VERBOSE", "").strip().lower()
    _full_verbose = verbose == "full"
    level = logging.DEBUG if verbose and verbose != "0" else logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    if log_file and verbose and verbose != "0":
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(file_handler)
    logger.propagate = False


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis if it exceeds max_len (no-op in full mode)."""
    if _full_verbose or len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated, {len(text)} chars total]"


def _safe_json(obj, indent: int = 2) -> str:
    """JSON-serialize with fallback for non-serializable objects."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except (TypeError, ValueError):
        return repr(obj)


# ── LLM call logging ────────────────────────────────────────────────

def log_llm_request(agent_name: str, *, pass_count: int = 0,
                    is_retry: bool = False, retry_count: int = 0,
                    message_count: int = 0, system_prompt_snippet: str = "",
                    user_message_snippet: str = "") -> None:
    """Log an outgoing LLM invocation."""
    logger.debug(
        "LLM_REQUEST agent=%s pass=%d is_retry=%s retry_count=%d "
        "messages=%d\n  system_prompt: %s\n  user_message: %s",
        agent_name, pass_count, is_retry, retry_count, message_count,
        _truncate(system_prompt_snippet, 300),
        _truncate(user_message_snippet, 300),
    )


def log_llm_response(agent_name: str, *, content: str = "",
                     tool_calls: list | None = None) -> None:
    """Log an LLM response (content and/or tool calls)."""
    parts = [f"LLM_RESPONSE agent={agent_name}"]
    if tool_calls:
        calls_summary = [
            {"name": tc.get("name", tc.get("function", {}).get("name", "?")),
             "args": tc.get("args", tc.get("function", {}).get("arguments", {}))}
            for tc in (
                tc if isinstance(tc, dict) else
                {"name": tc.name, "args": tc.args} if hasattr(tc, "name") else
                {"name": "?", "args": "?"}
                for tc in tool_calls
            )
        ]
        parts.append(f"\n  tool_calls ({len(tool_calls)}): {_safe_json(calls_summary)}")
    if content:
        parts.append(f"\n  content: {_truncate(content, 500)}")
    logger.debug("".join(parts))


def log_investigation_findings(findings) -> None:
    """Log the structured InvestigationFindings object."""
    data = findings.model_dump() if hasattr(findings, "model_dump") else findings
    violations = data.get("violations", [])

    lines = [
        f"INVESTIGATION_FINDINGS",
        f"  target_column: {data.get('target_column')}",
        f"  task_type: {data.get('task_type')}",
        f"  data_quality_score: {data.get('data_quality_score')}",
        f"  is_data_clean: {data.get('is_data_clean')}",
        f"  columns_to_drop: {data.get('columns_to_drop', [])}",
        f"  violations ({len(violations)}):",
    ]
    for v in violations:
        lines.append(
            f"    [{v['violation_id']}] {v['severity']} {v['category']} "
            f"cols={v['affected_columns']}\n"
            f"      description: {v['description']}\n"
            f"      suggested_action: {v['suggested_action']}"
        )
    if not violations:
        lines.append("    (none)")

    logger.debug("\n".join(lines))


def log_cleaning_recipe(recipe) -> None:
    """Log the structured CleaningRecipe object."""
    data = recipe.model_dump() if hasattr(recipe, "model_dump") else recipe
    steps = data.get("steps", [])

    lines = [f"CLEANING_RECIPE ({len(steps)} steps):"]
    for s in steps:
        lines.append(
            f"  step {s['step_id']}: [{s['operation']}] "
            f"target={s.get('target_column', 'N/A')} "
            f"violation={s.get('addresses_violation', 'N/A')}\n"
            f"    justification: {s['justification']}\n"
            f"    code: {_truncate(s['python_code'], 200)}"
        )
    logger.debug("\n".join(lines))


# ── Tool call logging ────────────────────────────────────────────────

def log_tool_call(tool_name: str, params: dict, result: str) -> None:
    """Log an investigation tool invocation and its result."""
    logger.debug(
        "TOOL_CALL %s\n  params: %s\n  result: %s",
        tool_name, _safe_json(params), _truncate(result, 500),
    )


# ── Routing decision logging ────────────────────────────────────────

def log_routing(router_name: str, decision: str, **context) -> None:
    """Log a routing decision with the values that drove it."""
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.debug("ROUTE %s → %s (%s)", router_name, decision, ctx_str)


# ── Node/state logging ──────────────────────────────────────────────

def log_node(node_name: str, message: str, **fields) -> None:
    """Log a node-level event with optional key-value fields."""
    if fields:
        field_str = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.debug("NODE %s | %s | %s", node_name, message, field_str)
    else:
        logger.debug("NODE %s | %s", node_name, message)


def log_profile_summary(profile: dict) -> None:
    """Log a compact summary of the profiler output."""
    columns = profile.get("columns", [])
    warnings_by_col = {}
    for col in columns:
        warns = col.get("semantic_warnings", [])
        if warns:
            warnings_by_col[col["name"]] = warns

    lines = [
        f"PROFILE_SUMMARY rows={profile.get('row_count')} columns={len(columns)}",
    ]
    if warnings_by_col:
        lines.append("  columns with warnings:")
        for col_name, warns in warnings_by_col.items():
            lines.append(f"    {col_name}: {warns}")
    else:
        lines.append("  no semantic warnings")

    logger.debug("\n".join(lines))
