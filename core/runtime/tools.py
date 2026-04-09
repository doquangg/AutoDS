################################################################################
# Investigation tools for the Investigator Agent.
#
# DESIGN PRINCIPLES:
#   - Tools return SUMMARIES, never full data dumps. The agent reasons over
#     digestible evidence, not thousands of rows.
#   - Most tools operate on the working DataFrame stored in the module-level
#     _current_df reference. This is set by the graph node before the agent
#     runs its tool loop.
#   - The web_search tool is an exception — it queries external sources to
#     verify semantic plausibility of data values.
#   - Tools are intentionally read-only — they inspect data, never modify it.
################################################################################

from __future__ import annotations
from typing import Any, Dict, Optional, List
import json
import os
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from core.schemas import (
    InspectRowsInput,
    CrossColumnFrequencyInput,
    TemporalOrderingCheckInput,
    ValueDistributionInput,
    NullCoOccurrenceInput,
    CorrelationScanInput,
    GetColumnForensicsInput,
    WebSearchInput,
)
from core.logger import log_tool_call
from plugins.profile_views import FORENSIC_COLUMN_FIELDS, _project_column

# ---------------------------------------------------------------------------
# Module-level DataFrame reference. Set by the graph node before the
# investigator agent enters its tool loop.
#
# This avoids passing the entire DataFrame through LangChain's tool-call
# serialization. The tools read from this reference directly.
# ---------------------------------------------------------------------------
_current_df: Optional[pd.DataFrame] = None

# Module-level profile reference. The graph node sets this alongside
# set_working_df() so tools can answer profile questions without re-profiling.
_current_profile: Optional[Dict[str, Any]] = None

# Web search rate limiting. Reset each time set_working_df() is called
# (i.e., at the start of each agent pass).
_web_search_count: int = 0
MAX_WEB_SEARCHES: int = 3


def set_working_df(df: pd.DataFrame) -> None:
    """Called by the graph node to give tools access to the current data."""
    global _current_df, _web_search_count
    _current_df = df
    _web_search_count = 0


def set_working_profile(profile: Optional[Dict[str, Any]]) -> None:
    """Called by the graph node to give tools access to the current profile."""
    global _current_profile
    _current_profile = profile


def _get_df() -> pd.DataFrame:
    """Internal helper — raises early if tools are called without data."""
    if _current_df is None:
        raise RuntimeError(
            "No working DataFrame set. Call set_working_df() before running tools."
        )
    return _current_df


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

@tool("inspect_rows", args_schema=InspectRowsInput)
def inspect_rows(query: str, limit: int = 5) -> str:
    """
    Return a small sample of rows matching a pandas query expression.
    Use this to verify suspicions from the profile — e.g., 'age < 0' to see
    what negative ages look like, or 'salary == 0' to inspect zero-salary rows.
    Keep limit small (≤10) to save tokens.
    """
    df = _get_df()
    try:
        filtered = df.query(query).head(limit)
    except Exception as e:
        result = f"Query failed: {e}"
        log_tool_call("inspect_rows", {"query": query, "limit": limit}, result)
        return result

    if filtered.empty:
        result = f"No rows match the query: {query}"
        log_tool_call("inspect_rows", {"query": query, "limit": limit}, result)
        return result

    result = (
        f"Found {len(df.query(query))} total rows matching '{query}'. "
        f"Showing first {len(filtered)}:\n\n"
        f"{filtered.to_string(max_colwidth=40)}"
    )
    log_tool_call("inspect_rows", {"query": query, "limit": limit}, result)
    return result


@tool("cross_column_frequency", args_schema=CrossColumnFrequencyInput)
def cross_column_frequency(col_a: str, col_b: str, top_n: int = 10) -> str:
    """
    Return a crosstab of the top value combinations between two columns.
    Use this to detect impossible or suspicious pairings — e.g., 
    department='Pediatrics' + procedure='Hip Replacement', or 
    status='Active' + salary=0.
    """
    df = _get_df()
    params = {"col_a": col_a, "col_b": col_b, "top_n": top_n}
    if col_a not in df.columns or col_b not in df.columns:
        result = f"Column not found. Available: {list(df.columns)}"
        log_tool_call("cross_column_frequency", params, result)
        return result

    # Build crosstab, flatten to top pairs
    ct = pd.crosstab(df[col_a], df[col_b])
    pairs = (
        ct.stack()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    total = len(df)
    pairs["pct"] = (pairs["count"] / total * 100).round(2)
    result = pairs.to_string(index=False)
    log_tool_call("cross_column_frequency", params, result)
    return result


@tool("temporal_ordering_check", args_schema=TemporalOrderingCheckInput)
def temporal_ordering_check(
    date_col_a: str, date_col_b: str, sample_violations: int = 5
) -> str:
    """
    Check whether date_col_a consistently occurs before date_col_b.
    Use this to detect causal violations — e.g., 'ship_date' before 'order_date',
    or 'termination_date' before 'hire_date'.
    """
    df = _get_df()
    if date_col_a not in df.columns or date_col_b not in df.columns:
        return f"Column not found. Available: {list(df.columns)}"

    params = {"date_col_a": date_col_a, "date_col_b": date_col_b,
              "sample_violations": sample_violations}

    a = pd.to_datetime(df[date_col_a], errors="coerce")
    b = pd.to_datetime(df[date_col_b], errors="coerce")

    # Only compare rows where both are non-null
    valid_mask = a.notna() & b.notna()
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        result = "No rows have valid dates in both columns — cannot check ordering."
        log_tool_call("temporal_ordering_check", params, result)
        return result

    violations = a[valid_mask] > b[valid_mask]
    violation_count = int(violations.sum())

    result = (
        f"Checked {valid_count} rows where both dates are non-null.\n"
        f"Violations ('{date_col_a}' occurs AFTER '{date_col_b}'): "
        f"{violation_count} ({violation_count / valid_count * 100:.1f}%)\n"
    )

    if violation_count > 0:
        sample = df[valid_mask][violations].head(sample_violations)
        result += f"\nSample violating rows:\n{sample[[date_col_a, date_col_b]].to_string()}"

    log_tool_call("temporal_ordering_check", params, result)
    return result


@tool("value_distribution", args_schema=ValueDistributionInput)
def value_distribution(column: str, bins: int = 20) -> str:
    """
    Return the distribution summary of a column. For numeric columns, returns a
    histogram with bin edges and counts. For categorical columns, returns value
    counts. Use this to spot sentinel spikes, multimodal distributions, or 
    outlier clusters.
    """
    params = {"column": column, "bins": bins}
    df = _get_df()
    if column not in df.columns:
        result = f"Column '{column}' not found. Available: {list(df.columns)}"
        log_tool_call("value_distribution", params, result)
        return result

    series = df[column].dropna()
    if series.empty:
        result = f"Column '{column}' is entirely null."
        log_tool_call("value_distribution", params, result)
        return result

    # Numeric: histogram
    if pd.api.types.is_numeric_dtype(series):
        series = series[np.isfinite(series)]
        if series.empty:
            result = f"Column '{column}' has no finite values (all inf/NaN)."
            log_tool_call("value_distribution", params, result)
            return result
        counts, edges = np.histogram(series, bins=bins)
        lines = [f"Distribution of '{column}' ({len(series)} non-null values):"]
        for i, count in enumerate(counts):
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * min(int(count / max(counts) * 30), 30)
            lines.append(f"  [{lo:>10.2f}, {hi:>10.2f}) {count:>6d} {bar}")
        result = "\n".join(lines)
        log_tool_call("value_distribution", params, result)
        return result

    # Categorical: value counts
    vc = series.value_counts().head(25)
    lines = [f"Value counts for '{column}' ({len(series)} non-null, {series.nunique()} unique):"]
    for val, count in vc.items():
        pct = count / len(series) * 100
        lines.append(f"  {val!s:<30s} {count:>6d} ({pct:.1f}%)")
    if series.nunique() > 25:
        lines.append(f"  ... and {series.nunique() - 25} more unique values")
    result = "\n".join(lines)
    log_tool_call("value_distribution", params, result)
    return result


@tool("null_co_occurrence", args_schema=NullCoOccurrenceInput)
def null_co_occurrence(threshold: float = 0.5) -> str:
    """
    Find columns that tend to be null together. A high co-occurrence ratio
    indicates systematic missingness — e.g., all address fields null together
    means the entire address record is missing, not just individual fields.
    """
    params = {"threshold": threshold}
    df = _get_df()
    null_matrix = df.isnull()
    null_cols = [c for c in df.columns if null_matrix[c].any()]

    if len(null_cols) < 2:
        result = "Fewer than 2 columns have any nulls — nothing to compare."
        log_tool_call("null_co_occurrence", params, result)
        return result

    pairs = []
    for i, col_a in enumerate(null_cols):
        for col_b in null_cols[i + 1 :]:
            both_null = (null_matrix[col_a] & null_matrix[col_b]).sum()
            either_null = (null_matrix[col_a] | null_matrix[col_b]).sum()
            if either_null > 0:
                ratio = both_null / either_null  # Jaccard-like overlap
                if ratio >= threshold:
                    pairs.append((col_a, col_b, int(both_null), round(ratio, 3)))

    if not pairs:
        result = f"No column pairs have null co-occurrence above {threshold}."
        log_tool_call("null_co_occurrence", params, result)
        return result

    pairs.sort(key=lambda x: x[3], reverse=True)
    lines = [f"Null co-occurrence pairs (threshold={threshold}):"]
    for col_a, col_b, count, ratio in pairs[:20]:
        lines.append(f"  {col_a} <-> {col_b}: {count} rows both null (overlap={ratio})")
    result = "\n".join(lines)
    log_tool_call("null_co_occurrence", params, result)
    return result


@tool("correlation_scan", args_schema=CorrelationScanInput)
def correlation_scan(target_column: str, top_n: int = 10) -> str:
    """
    Return the top N columns most correlated with the target column.
    Use this to understand which features are most relevant to the user's
    query, and to prioritize cleaning efforts on high-signal columns.
    Only works for numeric columns.
    """
    params = {"target_column": target_column, "top_n": top_n}
    df = _get_df()
    if target_column not in df.columns:
        result = f"Column '{target_column}' not found. Available: {list(df.columns)}"
        log_tool_call("correlation_scan", params, result)
        return result

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        result = f"Column '{target_column}' is not numeric — cannot compute correlation."
        log_tool_call("correlation_scan", params, result)
        return result

    corr = numeric_df.corr()[target_column].drop(target_column, errors="ignore")
    top = corr.abs().sort_values(ascending=False).head(top_n)

    lines = [f"Top {top_n} correlations with '{target_column}':"]
    for col, abs_val in top.items():
        actual = corr[col]
        direction = "+" if actual > 0 else "-"
        lines.append(f"  {col:<30s} {direction}{abs_val:.4f}")
    result = "\n".join(lines)
    log_tool_call("correlation_scan", params, result)
    return result


@tool("get_column_forensics", args_schema=GetColumnForensicsInput)
def get_column_forensics(column: str) -> str:
    """
    Return the full forensic view (stats + samples + coercion info + patterns)
    for a single column. Use this when the default profile view is missing
    a field you need — for example, random_sample_values, coercion_failure_samples,
    or dominant_pattern. Costs only the tokens for one column, not the whole profile.
    """
    params = {"column": column}
    if _current_profile is None:
        result = "No profile available. Cannot drill down."
        log_tool_call("get_column_forensics", params, result)
        return result

    columns = _current_profile.get("columns", [])
    col = next((c for c in columns if c.get("name") == column), None)
    if col is None:
        available = ", ".join(c.get("name", "?") for c in columns)
        result = f"Column '{column}' not found. Available: {available}"
        log_tool_call("get_column_forensics", params, result)
        return result

    projection = _project_column(col, FORENSIC_COLUMN_FIELDS)
    result = json.dumps(projection, separators=(",", ":"), default=str)
    log_tool_call("get_column_forensics", params, result)
    return result


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, domains: Optional[List[str]] = None) -> str:
    """
    Search the web to verify whether a data value is semantically possible
    in the real world. Use this to fact-check domain-specific claims —
    e.g., "Is a body temperature of 42°C survivable?", "What is the
    maximum plausible hospital bill in the US?", "Can a pediatric patient
    receive a hip replacement?"

    EXPENSIVE — each call has real monetary cost. Budget: 3 per pass.
    Use ONLY for obvious semantic outliers or domain-specific facts that
    the data alone cannot answer. Prefer data inspection tools first.

    For more reliable results, specify domains — e.g.,
    ['who.int', 'mayoclinic.org'] for medical facts.
    """
    global _web_search_count
    params = {"query": query, "domains": domains}

    if _web_search_count >= MAX_WEB_SEARCHES:
        result = (
            f"Web search budget exhausted ({MAX_WEB_SEARCHES}/{MAX_WEB_SEARCHES} "
            f"used this pass). Proceed with data-only investigation."
        )
        log_tool_call("web_search", params, result)
        return result

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        result = (
            "Web search unavailable: TAVILY_API_KEY not set. "
            "Proceed with data-only investigation."
        )
        log_tool_call("web_search", params, result)
        return result

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        search_kwargs = {
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
        }
        if domains:
            search_kwargs["include_domains"] = domains

        response = client.search(**search_kwargs)
        results = response.get("results", [])
    except Exception as e:
        result = f"Web search failed: {e}"
        log_tool_call("web_search", params, result)
        return result

    _web_search_count += 1

    if not results:
        result = f"No results found for: '{query}'"
        log_tool_call("web_search", params, result)
        return result

    lines = [
        f"Found {len(results)} result(s) for: '{query}' "
        f"[{_web_search_count}/{MAX_WEB_SEARCHES} searches used]"
    ]
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")
        # Truncate content to keep token count manageable
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"\n  [{title}]({url})")
        lines.append(f"  {content}")

    result = "\n".join(lines)
    log_tool_call("web_search", params, result)
    return result


# ---------------------------------------------------------------------------
# Export list — this is what gets passed to ToolNode and bind_tools()
# ---------------------------------------------------------------------------
investigation_tools = [
    inspect_rows,
    cross_column_frequency,
    temporal_ordering_check,
    value_distribution,
    null_co_occurrence,
    correlation_scan,
    get_column_forensics,
    web_search,
]