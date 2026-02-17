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
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            except TypeError:
                # Older pandas fallback (no format="mixed")
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


def _pattern_signature(s: str) -> str:
    """Compact pattern signature: A=letters, 9=digits, _=space, .=other"""
    out = []
    for ch in s:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        elif ch.isspace():
            out.append("_")
        else:
            out.append(".")
    return "".join(out)


def _regex_consistency(series: pd.Series, sample_size: int = 200) -> tuple[float | None, str | None]:
    """Return (consistency_ratio, dominant_pattern_signature)."""
    s = series.dropna().astype(str)
    if s.empty:
        return None, None
    s = s.head(sample_size)
    sigs = s.map(_pattern_signature)
    vc = sigs.value_counts()
    dominant = vc.index[0]
    ratio = float(vc.iloc[0] / len(sigs)) if len(sigs) else None
    return ratio, str(dominant)


def _datetime_consistency(series: pd.Series, sample_size: int = 200) -> tuple[float | None, str | None, str | None]:
    """Return (consistency_ratio, earliest_iso, latest_iso) using best-effort parsing."""
    s = series.dropna()
    if s.empty:
        return None, None, None

    # If already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(s, errors="coerce")
    else:
        # Parse object/string datetimes on a sample (cheap)
        sample = s.astype(str).head(sample_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dt = pd.to_datetime(sample, errors="coerce", format="mixed")
            except TypeError:
                # Older pandas fallback (no format="mixed")
                dt = pd.to_datetime(sample, errors="coerce")

    if dt.isna().all():
        return 0.0, None, None

    ratio = float(dt.notna().mean())

    dt_valid = dt.dropna()
    earliest = dt_valid.min()
    latest = dt_valid.max()
    earliest_iso = earliest.isoformat() if hasattr(earliest, "isoformat") else str(earliest)
    latest_iso = latest.isoformat() if hasattr(latest, "isoformat") else str(latest)

    return ratio, earliest_iso, latest_iso


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

        median = None
        zero_count = None
        negative_count = None
        inf_nan_count = None

        earliest_date = None
        latest_date = None
        datetime_format_consistency = None

        regex_format_consistency = None
        dominant_pattern = None

        if inferred_type == "Numeric":
            s_num = pd.to_numeric(series, errors="coerce")
            # Exclude non-finite
            s_num = s_num.replace([np.inf, -np.inf], np.nan)
            if s_num.notna().any():
                min_value = _safe_float(s_num.min())
                max_value = _safe_float(s_num.max())
                mean = _safe_float(s_num.mean())
                skewness = _safe_float(s_num.skew())

                median = _safe_float(s_num.median())

            # Counts (computed even if all NaN after coercion)
            zero_count = int((s_num == 0).sum()) if len(s_num) else 0
            negative_count = int((s_num < 0).sum()) if len(s_num) else 0
            # inf count already mapped to NaN above, but count original non-finite via coercion result:
            inf_nan_count = int(s_num.isna().sum())

        top_vals = _top_frequent_values(series, k=5)

        if inferred_type == "Datetime":
            datetime_format_consistency, earliest_date, latest_date = _datetime_consistency(series)

        # Regex/pattern consistency for string/categorical-like columns
        if inferred_type == "Categorical" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            regex_format_consistency, dominant_pattern = _regex_consistency(series)
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
                median=median,
                zero_count=zero_count,
                negative_count=negative_count,
                inf_nan_count=inf_nan_count,
                earliest_date=earliest_date,
                latest_date=latest_date,
                datetime_format_consistency=datetime_format_consistency,
                regex_format_consistency=regex_format_consistency,
                dominant_pattern=dominant_pattern,
                top_frequent_values=top_vals,
                semantic_warnings=warnings,
            )
        )

    profile = DatasetProfile(row_count=row_count, columns=columns)
    # Return plain dict for JSON serialization
    return profile.model_dump()
