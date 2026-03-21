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
import json
import time

from core.schemas import DatasetProfile, ColumnProfile, AlgorithmicQualityScore

# ydata metric cap (keep profile compact for LLM/tokens)
DEFAULT_MAX_METRICS_PER_COL = 20


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


def _to_json_safe(val: Any) -> Any:
    """Convert values into JSON-safe primitives."""
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        try:
            return val.isoformat()
        except Exception:
            return str(val)
    if isinstance(val, (np.generic,)):
        return _to_json_safe(val.item())
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    if isinstance(val, dict):
        return {str(_to_json_safe(k)): _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_json_safe(x) for x in list(val)]
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _top_frequent_values(series: pd.Series, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k frequent values (including NaN excluded) in JSON-safe forms."""

    try:
        vc = series.dropna().value_counts(dropna=True).head(k)
    except TypeError:
        vc = series.dropna().map(_to_json_safe).astype(str).value_counts().head(k)
    out: List[Dict[str, Any]] = []
    for v, c in vc.items():
        out.append({"value": _to_json_safe(v), "count": int(c)})
    return out


# issues = local name (avoid shadowing stdlib warnings); schema field stays semantic_warnings for API stability
def _semantic_warnings(series: pd.Series, inferred_type: str, unique_factor: float) -> List[str]:
    issues: List[str] = []
    s_nonnull = series.dropna()
    n = len(series)

    # High cardinality / possible ID or PII proxy
    if n > 0 and unique_factor > 0.95 and n >= 50:
        issues.append("High Cardinality")

    # Constant / near-constant
    if len(s_nonnull) > 0:
        top = s_nonnull.value_counts().iloc[0]
        if top / len(s_nonnull) >= 0.99 and len(s_nonnull) >= 50:
            issues.append("Near Constant Value")

    # Missingness
    missing = int(series.isna().sum())
    if n > 0 and (missing / n) >= 0.3:
        issues.append("High Missingness")

    # Numeric-specific sentinel hints
    if inferred_type == "Numeric" and len(s_nonnull) > 0:
        s_num = pd.to_numeric(s_nonnull, errors="coerce")
        # Common sentinel values
        for sentinel in (-1, 0, 999, 9999):
            if (s_num == sentinel).mean() >= 0.05:  # appears in >=5% rows
                issues.append(f"Possible Sentinel: {sentinel}")

        if (s_num < 0).any():
            issues.append("Contains Negative Values")

    # Heaping detection: single value dominates a column that should have variety
    if inferred_type == "Numeric" and len(s_nonnull) > 0:
        s_num = pd.to_numeric(s_nonnull, errors="coerce").dropna()
        if len(s_num) > 0:
            vc = s_num.value_counts()
            top_value = vc.index[0]
            top_count = int(vc.iloc[0])
            top_ratio = top_count / len(s_num)
            # Flag if top value accounts for >40% of non-null values
            # AND column has some variety (not a true binary/categorical)
            if top_ratio > 0.40 and unique_factor > 0.05:
                issues.append(
                    f"Value Heaping: {top_value} appears {top_count} times "
                    f"({top_ratio:.0%} of non-null values)"
                )

    # String cleanliness
    if inferred_type == "Categorical" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        sample = s_nonnull.astype(str).head(200)
        if (sample.str.strip() != sample).any():
            issues.append("Leading/Trailing Whitespace")

    return issues


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




def _ydata_profile_whitelist(df: pd.DataFrame, max_metrics_per_col: int = 20) -> tuple[Dict[str, Dict[str, Any]], str | None]:
    """
    Run ydata-profiling and return a compact per-column dict of whitelisted metrics.

    Fail-open behavior: on any error, returns ({}, error_string). Baseline still works.
    """
    try:
        # Import only when needed to keep baseline lightweight
        from ydata_profiling import ProfileReport

        # Keep it lightweight: minimal report, no progress bar
        report = ProfileReport(df, minimal=True, progress_bar=False)
        payload = json.loads(report.to_json())

        variables = payload.get("variables") or {}
        out: Dict[str, Dict[str, Any]] = {}

        # Small whitelist: prefer *non-overlapping* extras (only include keys if present)
        common = [
            "n_missing", "p_missing",
            "n_distinct", "p_distinct",
            "memory_size",
        ]
        numeric_extra = [
            "std", "variance", "kurtosis", "iqr", "mad",
        ]
        categorical_extra = [
            "n_unique", "p_unique",
            "max_length", "min_length", "avg_length",
        ]
        datetime_extra = [
            "min", "max",
            "n_unique", "p_unique",
        ]

        for col, v in variables.items():
            vtype = (v.get("type") or "").lower()
            keep = {}

            for k in common:
                if k in v:
                    keep[k] = v[k]

            # Type-sensitive extras (best-effort; keys may differ by ydata version)
            if "numeric" in vtype:
                for k in numeric_extra:
                    if k in v:
                        keep[k] = v[k]
            elif "date" in vtype or "time" in vtype or "datetime" in vtype:
                for k in datetime_extra:
                    if k in v:
                        keep[k] = v[k]
            else:
                for k in categorical_extra:
                    if k in v:
                        keep[k] = v[k]

            # Cap size hard to avoid token blowups
            if keep:
                out[str(col)] = dict(list(keep.items())[:max_metrics_per_col])

        return out, None

    except Exception as e:
        return {}, str(e)


def compute_quality_score(
    profile: Dict[str, Any],
    target_column: str | None = None,
) -> Dict[str, Any]:
    """
    Compute a deterministic quality score from profile statistics.

    Returns a dict compatible with AlgorithmicQualityScore:
        overall, completeness, target_integrity, value_plausibility,
        structural_integrity, flags.

    Weights (with target column):
        completeness       0.00  (informational only — AutoGluon handles missing features)
        target_integrity   0.45
        value_plausibility 0.30
        structural_integrity 0.25

    Weights (no target column):
        completeness       0.00
        value_plausibility 0.55
        structural_integrity 0.45
    """
    columns = profile.get("columns", [])
    if not columns:
        return AlgorithmicQualityScore(
            overall=0.0, completeness=0.0, target_integrity=None,
            value_plausibility=0.0, structural_integrity=0.0,
            flags=["no_columns_in_profile"],
        ).model_dump()

    row_count = profile.get("row_count", 0)
    flags: List[str] = []

    # --- Completeness sub-score ---
    completeness_values = [c.get("completeness", 0.0) for c in columns]
    completeness_score = sum(completeness_values) / len(completeness_values)

    # --- Target integrity sub-score ---
    target_integrity = None
    if target_column:
        target_col = next(
            (c for c in columns if c.get("name") == target_column), None
        )
        if target_col:
            ti = target_col.get("completeness", 0.0)
            # Penalize inf/nan in target
            inf_nan = target_col.get("inf_nan_count") or 0
            if inf_nan > 0 and row_count > 0:
                penalty = min(inf_nan / row_count, 1.0)
                ti = ti * (1.0 - penalty)
                flags.append(
                    f"target_inf_nan: {target_column} has {inf_nan} inf/nan values"
                )
            # Check for heaping in target
            warnings_list = target_col.get("semantic_warnings", [])
            for w in warnings_list:
                if "Value Heaping" in w:
                    ti *= 0.7
                    flags.append(f"target_heaping: {target_column} — {w}")
                    break
            target_integrity = ti
        else:
            flags.append(f"target_column_not_found: {target_column}")

    # --- Value plausibility sub-score ---
    # Penalizes heaping/template patterns in numeric columns
    numeric_cols = [c for c in columns if c.get("inferred_type") == "Numeric"]
    if numeric_cols:
        plausibility_scores = []
        for col in numeric_cols:
            col_score = 1.0
            warnings_list = col.get("semantic_warnings", [])
            for w in warnings_list:
                if "Value Heaping" in w:
                    # Extract ratio if possible, else use default penalty
                    col_score *= 0.4
                    col_name = col.get("name", "unknown")
                    flags.append(f"heaping: {col_name} — {w}")
                    break
                if "Possible Sentinel" in w:
                    col_score *= 0.7
            plausibility_scores.append(col_score)
        value_plausibility = sum(plausibility_scores) / len(plausibility_scores)
    else:
        value_plausibility = 1.0

    # --- Structural integrity sub-score ---
    # Fraction of numeric columns with no inf/nan
    if numeric_cols:
        clean_cols = sum(
            1 for c in numeric_cols
            if (c.get("inf_nan_count") or 0) == 0
        )
        structural_integrity = clean_cols / len(numeric_cols)
        bad_cols = [
            c.get("name") for c in numeric_cols
            if (c.get("inf_nan_count") or 0) > 0
        ]
        if bad_cols:
            flags.append(f"inf_nan_present: {', '.join(bad_cols)}")
    else:
        structural_integrity = 1.0

    # --- Weighted composite ---
    # Completeness is informational only (weight 0) because AutoGluon
    # handles missing feature data natively.
    if target_integrity is not None:
        overall = (
            0.45 * target_integrity
            + 0.30 * value_plausibility
            + 0.25 * structural_integrity
        )
    else:
        overall = (
            0.55 * value_plausibility
            + 0.45 * structural_integrity
        )

    overall = max(0.0, min(1.0, overall))

    return AlgorithmicQualityScore(
        overall=round(overall, 3),
        completeness=round(completeness_score, 3),
        target_integrity=round(target_integrity, 3) if target_integrity is not None else None,
        value_plausibility=round(value_plausibility, 3),
        structural_integrity=round(structural_integrity, 3),
        flags=flags,
    ).model_dump()


def generate_profile(df: pd.DataFrame, detailed_profiler: bool = False, target_column: str | None = None) -> Dict[str, Any]:
    """
    Generate a DatasetProfile-compatible dict for the given dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("generate_profile expects a pandas.DataFrame")

    row_count = int(len(df))

    # Optional detailed profiling (ydata-profiling) with guardrails
    ydata_by_col: Dict[str, Dict[str, Any]] = {}
    ydata_error: str | None = None
    ydata_skipped: str | None = None
    if detailed_profiler:
        # Guardrails: sample big datasets, cap extremely wide tables, fail-open on errors
        ROW_THRESHOLD = 50_000
        SAMPLE_N = 10_000
        COL_CAP = 200

        # FIXME: This needs to be better considered...
        TIME_BUDGET_SEC = 8.0  # soft budget; if exceeded, treat as fallback
        if df.shape[1] <= COL_CAP:
            df_y = df
            if row_count > ROW_THRESHOLD:
                df_y = df.sample(n=min(SAMPLE_N, row_count), random_state=42)
            try:
                _t0 = time.perf_counter()
                ydata_by_col, ydata_error = _ydata_profile_whitelist(df_y, max_metrics_per_col=DEFAULT_MAX_METRICS_PER_COL)
                _elapsed = time.perf_counter() - _t0
                if _elapsed > TIME_BUDGET_SEC:
                    # Soft budget: we can't interrupt ydata mid-run, but we can fail-open if it took too long
                    ydata_by_col = {}
                    ydata_error = f"ydata_time_budget_exceeded:{_elapsed:.2f}s"
            except Exception as e:
                ydata_by_col, ydata_error = {}, f"ydata_exception:{type(e).__name__}"
        else:
            # Too many columns -> skip ydata to avoid performance/memory blowups
            ydata_by_col = {}
            ydata_skipped = "wide_table"

    columns: List[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        completeness = (non_null / row_count) if row_count > 0 else 0.0
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            # Handle unhashable objects (e.g., dict/list) by counting uniques on JSON-safe strings
            unique_count = int(series.dropna().map(_to_json_safe).astype(str).nunique())

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
        issues = _semantic_warnings(series, inferred_type, unique_factor)
        if detailed_profiler:
            if ydata_error:
                issues.append("ydata_fallback_to_baseline")
            elif ydata_skipped == "wide_table":
                issues.append("ydata_skipped_wide_table")

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
                ydata_metrics=ydata_by_col.get(str(col)) if detailed_profiler else None,
                top_frequent_values=top_vals,
                semantic_warnings=issues,
            )
        )

    profile = DatasetProfile(row_count=row_count, columns=columns)
    # Return plain dict for JSON serialization
    result = profile.model_dump()

    # Compute and attach algorithmic quality score
    result["algorithmic_quality_score"] = compute_quality_score(result, target_column=target_column)

    return result
