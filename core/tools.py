################################################################################
# Investigation tools for the Investigator Agent.
#
# DESIGN PRINCIPLES:
#   - Tools return SUMMARIES, never full data dumps. The agent reasons over
#     digestible evidence, not thousands of rows.
#   - Every tool operates on the working DataFrame stored in the module-level
#     _current_df reference. This is set by the graph node before the agent
#     runs its tool loop. 
#   - Tools are intentionally read-only — they inspect data, never modify it.
################################################################################

from __future__ import annotations
from typing import Optional
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
)

# ---------------------------------------------------------------------------
# Module-level DataFrame reference. Set by the graph node before the
# investigator agent enters its tool loop.
#
# This avoids passing the entire DataFrame through LangChain's tool-call
# serialization. The tools read from this reference directly.
# ---------------------------------------------------------------------------
_current_df: Optional[pd.DataFrame] = None


def set_working_df(df: pd.DataFrame) -> None:
    """Called by the graph node to give tools access to the current data."""
    global _current_df
    _current_df = df


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
        return f"Query failed: {e}"

    if filtered.empty:
        return f"No rows match the query: {query}"

    return (
        f"Found {len(df.query(query))} total rows matching '{query}'. "
        f"Showing first {len(filtered)}:\n\n"
        f"{filtered.to_string(max_colwidth=40)}"
    )


@tool("cross_column_frequency", args_schema=CrossColumnFrequencyInput)
def cross_column_frequency(col_a: str, col_b: str, top_n: int = 10) -> str:
    """
    Return a crosstab of the top value combinations between two columns.
    Use this to detect impossible or suspicious pairings — e.g., 
    department='Pediatrics' + procedure='Hip Replacement', or 
    status='Active' + salary=0.
    """
    df = _get_df()
    if col_a not in df.columns or col_b not in df.columns:
        return f"Column not found. Available: {list(df.columns)}"

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
    return pairs.to_string(index=False)


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

    a = pd.to_datetime(df[date_col_a], errors="coerce")
    b = pd.to_datetime(df[date_col_b], errors="coerce")

    # Only compare rows where both are non-null
    valid_mask = a.notna() & b.notna()
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return "No rows have valid dates in both columns — cannot check ordering."

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

    return result


@tool("value_distribution", args_schema=ValueDistributionInput)
def value_distribution(column: str, bins: int = 20) -> str:
    """
    Return the distribution summary of a column. For numeric columns, returns a
    histogram with bin edges and counts. For categorical columns, returns value
    counts. Use this to spot sentinel spikes, multimodal distributions, or 
    outlier clusters.
    """
    df = _get_df()
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"

    series = df[column].dropna()
    if series.empty:
        return f"Column '{column}' is entirely null."

    # Numeric: histogram
    if pd.api.types.is_numeric_dtype(series):
        counts, edges = np.histogram(series, bins=bins)
        lines = [f"Distribution of '{column}' ({len(series)} non-null values):"]
        for i, count in enumerate(counts):
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * min(int(count / max(counts) * 30), 30)
            lines.append(f"  [{lo:>10.2f}, {hi:>10.2f}) {count:>6d} {bar}")
        return "\n".join(lines)

    # Categorical: value counts
    vc = series.value_counts().head(25)
    lines = [f"Value counts for '{column}' ({len(series)} non-null, {series.nunique()} unique):"]
    for val, count in vc.items():
        pct = count / len(series) * 100
        lines.append(f"  {val!s:<30s} {count:>6d} ({pct:.1f}%)")
    if series.nunique() > 25:
        lines.append(f"  ... and {series.nunique() - 25} more unique values")
    return "\n".join(lines)


@tool("null_co_occurrence", args_schema=NullCoOccurrenceInput)
def null_co_occurrence(threshold: float = 0.5) -> str:
    """
    Find columns that tend to be null together. A high co-occurrence ratio
    indicates systematic missingness — e.g., all address fields null together
    means the entire address record is missing, not just individual fields.
    """
    df = _get_df()
    null_matrix = df.isnull()
    null_cols = [c for c in df.columns if null_matrix[c].any()]

    if len(null_cols) < 2:
        return "Fewer than 2 columns have any nulls — nothing to compare."

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
        return f"No column pairs have null co-occurrence above {threshold}."

    pairs.sort(key=lambda x: x[3], reverse=True)
    lines = [f"Null co-occurrence pairs (threshold={threshold}):"]
    for col_a, col_b, count, ratio in pairs[:20]:
        lines.append(f"  {col_a} <-> {col_b}: {count} rows both null (overlap={ratio})")
    return "\n".join(lines)


@tool("correlation_scan", args_schema=CorrelationScanInput)
def correlation_scan(target_column: str, top_n: int = 10) -> str:
    """
    Return the top N columns most correlated with the target column.
    Use this to understand which features are most relevant to the user's
    query, and to prioritize cleaning efforts on high-signal columns.
    Only works for numeric columns.
    """
    df = _get_df()
    if target_column not in df.columns:
        return f"Column '{target_column}' not found. Available: {list(df.columns)}"

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        return f"Column '{target_column}' is not numeric — cannot compute correlation."

    corr = numeric_df.corr()[target_column].drop(target_column, errors="ignore")
    top = corr.abs().sort_values(ascending=False).head(top_n)

    lines = [f"Top {top_n} correlations with '{target_column}':"]
    for col, abs_val in top.items():
        actual = corr[col]
        direction = "+" if actual > 0 else "-"
        lines.append(f"  {col:<30s} {direction}{abs_val:.4f}")
    return "\n".join(lines)


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
]