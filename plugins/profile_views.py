"""
Role-specific projections of the DatasetProfile for LLM consumption.

Three nested views, from smallest to largest:
    core     — name + dtypes + cardinality + completeness (target_selector, fe_codegen)
    stats    — core + numeric/categorical stats + semantic warnings (fe_planner, cleaning codegen)
    forensic — stats + sample values, coercion samples, regex patterns, ydata_metrics (investigator)

Every LLM-bound profile serialization site calls `profile_to_llm_json` with
one of these view names. This is the only place where profile size for LLMs
is decided, and the only place that strips `indent` for LLM-bound output.

Views are pure projections — they never mutate the underlying profile dict.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

ProfileViewName = Literal["core", "stats", "forensic"]


CORE_COLUMN_FIELDS = (
    "name",
    "inferred_type",
    "actual_dtype",
    "type_mismatch",
    "completeness",
    "unique_factor",
    "top_frequent_values",
)

STATS_COLUMN_FIELDS = CORE_COLUMN_FIELDS + (
    "min_value",
    "max_value",
    "mean",
    "median",
    "skewness",
    "q1",
    "q3",
    "zero_count",
    "negative_count",
    "inf_nan_count",
    "earliest_date",
    "latest_date",
    "datetime_format_consistency",
    "future_date_count",
    "coercion_failure_count",
    "semantic_warnings",
)

FORENSIC_COLUMN_FIELDS = STATS_COLUMN_FIELDS + (
    "random_sample_values",
    "coercion_failure_samples",
    "datetime_format_samples",
    "regex_format_consistency",
    "dominant_pattern",
    "ydata_metrics",
)


def _project_column(col: Dict[str, Any], fields: tuple[str, ...]) -> Dict[str, Any]:
    """Return a new dict containing only the whitelisted fields that are present and non-None."""
    out: Dict[str, Any] = {}
    for f in fields:
        if f not in col:
            continue
        val = col[f]
        if val is None:
            continue
        # Drop empty lists/dicts to keep the projection compact.
        if isinstance(val, (list, dict)) and len(val) == 0:
            continue
        out[f] = val
    return out


def project_profile(profile: Dict[str, Any], view: ProfileViewName) -> Dict[str, Any]:
    """
    Return a new profile dict projected into the requested view.

    The result is a shallow copy: `row_count` is copied, `columns` is a list
    of projected column dicts. The input is not modified.
    """
    if view == "core":
        fields = CORE_COLUMN_FIELDS
    elif view == "stats":
        fields = STATS_COLUMN_FIELDS
    elif view == "forensic":
        fields = FORENSIC_COLUMN_FIELDS
    else:
        raise ValueError(f"Unknown profile view: {view!r}")

    columns: List[Dict[str, Any]] = profile.get("columns", [])
    return {
        "row_count": profile.get("row_count", 0),
        "columns": [_project_column(c, fields) for c in columns],
    }


def profile_to_llm_json(profile: Dict[str, Any], view: ProfileViewName) -> str:
    """
    Project the profile into the requested view and serialize it for LLM use.

    This is the single allowed entry point for profile → LLM serialization.
    It strips whitespace (no `indent`) and uses compact separators.
    """
    projected = project_profile(profile, view)
    return json.dumps(projected, separators=(",", ":"), default=str)
