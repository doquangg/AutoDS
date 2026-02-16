"""
Profiler plugin: generates a compact DatasetProfile JSON for the Investigating Agent.

Contract:
- Must return a dict compatible with core.schemas.DatasetProfile
- Entry point: generate_profile(df: pandas.DataFrame) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List
import math
import warnings

import numpy as np
import pandas as pd

from core.schemas import DatasetProfile, ColumnProfile


def _safe_float(x: Any) -> float | None:
    """Convert numpy/pandas scalars to python float; return None if not finite."""
    if x is None:
        return None
    try:
        val = float(x)
    except Exception:
        return None
    if math.isfinite(val):
        return val
    return None


def _infer_type(series: pd.Series) -> str:
    """Return one of: 'Numeric', 'Categorical', 'Datetime' (as required by schema description)."""
    s = series.dropna()
    if s.empty:
        return "Categorical"  # default when no signal

    # Datetime detection (works even if dtype is already datetime)
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    # Try parsing datetimes for object-like columns (cheap sample)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = s.astype(str).head(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:  # mostly parseable => treat as datetime
            return "Datetime"

    # Numeric detection (including bool treated as categorical-ish)
    if pd.api.types.is_bool_dtype(series):
        return "Categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"

    # Fallback
    return "Categorical"


def _top_frequent_values(series: pd.Series, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k frequent values (including NaN excluded)."""
    vc = series.dropna().value_counts(dropna=True).head(k)
    out: List[Dict[str, Any]] = []
    for v, c in vc.items():
        # Convert numpy types to python primitives for JSON friendliness
        if isinstance(v, (np.generic,)):
            v = v.item()
        out.append({"value": v, "count": int(c)})
    return out


def _semantic_warnings(series: pd.Series, inferred_type: str, unique_factor: float) -> List[str]:
    warnings: List[str] = []
    s_nonnull = series.dropna()
    n = len(series)

    # High cardinality / possible ID or PII proxy
    if n > 0 and unique_factor > 0.95 and n >= 50:
        warnings.append("High Cardinality")

    # Constant / near-constant
    if len(s_nonnull) > 0:
        top = s_nonnull.value_counts().iloc[0]
        if top / len(s_nonnull) >= 0.99 and len(s_nonnull) >= 50:
            warnings.append("Near Constant Value")

    # Missingness
    missing = int(series.isna().sum())
    if n > 0 and (missing / n) >= 0.3:
        warnings.append("High Missingness")

    # Numeric-specific sentinel hints
    if inferred_type == "Numeric" and len(s_nonnull) > 0:
        s_num = pd.to_numeric(s_nonnull, errors="coerce")
        # Common sentinel values
        for sentinel in (-1, 0, 999, 9999):
            if (s_num == sentinel).mean() >= 0.05:  # appears in >=5% rows
                warnings.append(f"Possible Sentinel: {sentinel}")

        if (s_num < 0).any():
            warnings.append("Contains Negative Values")

    # String cleanliness
    if inferred_type == "Categorical" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        sample = s_nonnull.astype(str).head(200)
        if (sample.str.strip() != sample).any():
            warnings.append("Leading/Trailing Whitespace")

    return warnings


def generate_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a DatasetProfile-compatible dict for the given dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("generate_profile expects a pandas.DataFrame")

    row_count = int(len(df))
    columns: List[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        completeness = (non_null / row_count) if row_count > 0 else 0.0

        unique_count = int(series.nunique(dropna=True))
        unique_factor = (unique_count / row_count) if row_count > 0 else 0.0

        inferred_type = _infer_type(series)

        min_value = max_value = mean = skewness = None

        if inferred_type == "Numeric":
            s_num = pd.to_numeric(series, errors="coerce")
            # Exclude non-finite
            s_num = s_num.replace([np.inf, -np.inf], np.nan)
            if s_num.notna().any():
                min_value = _safe_float(s_num.min())
                max_value = _safe_float(s_num.max())
                mean = _safe_float(s_num.mean())
                skewness = _safe_float(s_num.skew())

        top_vals = _top_frequent_values(series, k=5)
        warnings = _semantic_warnings(series, inferred_type, unique_factor)

        columns.append(
            ColumnProfile(
                name=str(col),
                inferred_type=inferred_type,
                completeness=float(completeness),
                unique_factor=float(unique_factor),
                min_value=min_value,
                max_value=max_value,
                mean=mean,
                skewness=skewness,
                top_frequent_values=top_vals,
                semantic_warnings=warnings,
            )
        )

    profile = DatasetProfile(row_count=row_count, columns=columns)
    # Return plain dict for JSON serialization
    return profile.model_dump()
