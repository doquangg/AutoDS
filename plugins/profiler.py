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

from core.schemas import DatasetProfile, ColumnProfile

# ydata metric cap (keep profile compact for LLM/tokens)
DEFAULT_MAX_METRICS_PER_COL = 20


def _safe_float(x: Any) -> float | None:
    """Convert numpy/pandas scalars to python float rounded to 4 significant figures.

    Returns None if the value is not finite. Rounding happens at the source
    so every downstream consumer (LLMs, disk artifacts, quality checks) sees
    a compact representation without needing its own rounding pass.
    """
    if x is None:
        return None
    try:
        val = float(x)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    if val == 0.0:
        return 0.0
    # 4 significant figures — keeps skewness like 0.0003421 readable and
    # strips meaningless trailing precision on values like 123.456789.
    return float(f"{val:.4g}")


def _infer_type(series: pd.Series, s_nonnull: pd.Series | None = None) -> str:
    """Return one of: 'Numeric', 'Categorical', 'Datetime'."""
    s = s_nonnull if s_nonnull is not None else series.dropna()
    if s.empty:
        return "Categorical"

    # If already datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"

    # Object/string columns: try Datetime first (cheap sample)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = s.astype(str).head(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
            except TypeError:
                parsed_dt = pd.to_datetime(sample, errors="coerce")
        if parsed_dt.notna().mean() >= 0.6:
            return "Datetime"

        # If not datetime, try Numeric coercion on the same sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_num = pd.to_numeric(sample, errors="coerce")
        if parsed_num.notna().mean() >= 0.6:
            return "Numeric"

    # Bool treated as categorical
    if pd.api.types.is_bool_dtype(series):
        return "Categorical"

    # Native numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"

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


def _top_frequent_values(series: pd.Series, k: int = 5, s_nonnull: pd.Series | None = None) -> List[Dict[str, Any]]:
    """Return top-k frequent values (including NaN excluded) in JSON-safe forms."""
    s = s_nonnull if s_nonnull is not None else series.dropna()
    try:
        vc = s.value_counts(dropna=True).head(k)
    except TypeError:
        vc = s.map(_to_json_safe).astype(str).value_counts().head(k)
    out: List[Dict[str, Any]] = []
    for v, c in vc.items():
        out.append({"value": _to_json_safe(v), "count": int(c)})
    return out


# issues = local name (avoid shadowing stdlib warnings); schema field stays semantic_warnings for API stability
def _count_future_dates(s_nonnull: pd.Series) -> int:
    """Count datetime values after today. Returns 0 on any error."""
    if len(s_nonnull) == 0:
        return 0
    try:
        dt_vals = pd.to_datetime(s_nonnull, errors="coerce").dropna()
        today = pd.Timestamp.now().normalize()
        return int((dt_vals > today).sum())
    except Exception:
        return 0


def _semantic_warnings(
    series: pd.Series,
    inferred_type: str,
    unique_factor: float,
    s_nonnull: pd.Series | None = None,
    s_num_clean: pd.Series | None = None,
    future_date_count: int | None = None,
) -> List[str]:
    issues: List[str] = []
    if s_nonnull is None:
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

    # Missingness (informational only — AutoGluon handles missing data natively)
    missing = int(series.isna().sum())
    if n > 0 and (missing / n) >= 0.3:
        issues.append("High Missingness")

    # Numeric-specific sentinel hints
    if inferred_type == "Numeric" and len(s_nonnull) > 0:
        s_num = s_num_clean if s_num_clean is not None else pd.to_numeric(s_nonnull, errors="coerce")
        # Common sentinel values
        for sentinel in (-1, 0, 999, 9999):
            if (s_num == sentinel).mean() >= 0.05:  # appears in >=5% rows
                issues.append(f"Possible Sentinel: {sentinel}")

        if (s_num < 0).any():
            issues.append("Contains Negative Values")

    # String cleanliness
    if inferred_type == "Categorical" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        sample = s_nonnull.astype(str).head(200)
        if (sample.str.strip() != sample).any():
            issues.append("Leading/Trailing Whitespace")

    # Datetime: future date detection
    if inferred_type == "Datetime" and len(s_nonnull) > 0:
        if future_date_count is None:
            future_date_count = _count_future_dates(s_nonnull)
        if future_date_count > 0:
            issues.append(f"Future Dates Detected: {future_date_count} values after {pd.Timestamp.now().normalize().date().isoformat()}")

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


def compute_anomaly_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a per-column statistical anomaly summary.

    For numeric columns, flags outliers using z-score (|z| > 3) and IQR methods.
    For categorical columns, flags rare categories (< 1% frequency).

    Returns a dict with keys: column_summaries, total_rows, total_anomalous_rows.
    """
    import numpy as np

    column_summaries: List[Dict[str, Any]] = []
    # Track anomalous rows efficiently with a boolean array.
    # Use the DataFrame's own index so label-based indexing works correctly
    # even when the index is non-contiguous after row drops.
    anomaly_flags = pd.Series(False, index=df.index)

    for col in df.columns:
        series = df[col]
        s_nonnull = series.dropna()

        if pd.api.types.is_numeric_dtype(series) and len(s_nonnull) > 0:
            s_num = pd.to_numeric(s_nonnull, errors="coerce").dropna()
            if len(s_num) < 2:
                continue

            # Z-score outliers (|z| > 3)
            mean = s_num.mean()
            std = s_num.std()
            z_outlier_mask = pd.Series(False, index=s_num.index)
            if std > 0:
                z_scores = (s_num - mean).abs() / std
                z_outlier_mask = z_scores > 3

            # IQR outliers
            q1 = s_num.quantile(0.25)
            q3 = s_num.quantile(0.75)
            iqr = q3 - q1
            iqr_outlier_mask = pd.Series(False, index=s_num.index)
            if iqr > 0:
                iqr_outlier_mask = (s_num < q1 - 1.5 * iqr) | (s_num > q3 + 1.5 * iqr)

            # Combine: flagged by either method
            combined_mask = z_outlier_mask | iqr_outlier_mask
            outlier_count = int(combined_mask.sum())

            if outlier_count > 0:
                outlier_vals = s_num[combined_mask]
                high = int((outlier_vals > mean).sum())
                low = int((outlier_vals <= mean).sum())
                if high > 0 and low > 0:
                    direction = "both"
                elif high > 0:
                    direction = "high"
                else:
                    direction = "low"

                anomaly_flags[s_num.index[combined_mask]] = True
            else:
                direction = "none"

            column_summaries.append({
                "column": col,
                "outlier_count": outlier_count,
                "outlier_direction": direction,
                "rare_categories": None,
            })

        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if len(s_nonnull) == 0:
                continue
            vc = s_nonnull.value_counts(normalize=True)
            rare = [str(v) for v, pct in vc.items() if pct < 0.01]

            if rare:
                rare_mask = s_nonnull.isin(rare)
                anomaly_flags[s_nonnull.index[rare_mask]] = True

            column_summaries.append({
                "column": col,
                "outlier_count": 0,
                "outlier_direction": "none",
                "rare_categories": rare[:10] if rare else None,
            })

    return {
        "column_summaries": column_summaries,
        "total_rows": len(df),
        "total_anomalous_rows": int(anomaly_flags.sum()),
    }


def profile_one_column(
    df: pd.DataFrame,
    col_name: str,
    *,
    row_count: int,
    detailed_profiler: bool,
    ydata_col_metrics: Dict[str, Any] | None,
) -> ColumnProfile:
    """
    Compute a ColumnProfile for exactly one column of ``df``.

    This is the per-column body of generate_profile, factored out so callers
    that already have a previous profile can recompute only the columns that
    actually changed (see extend_profile / refresh_profile_for_recipe).

    Pure function: given the same inputs, must return the same ColumnProfile
    that generate_profile would produce for the same column on the same data.
    Global ydata bootstrap warnings (ydata_fallback_to_baseline,
    ydata_skipped_wide_table) live in generate_profile, NOT here, because
    they apply to the whole-dataframe ydata run rather than to any one column.
    """
    col = col_name
    series = df[col]
    s_nonnull = series.dropna()  # compute once; passed to all helpers below
    non_null = len(s_nonnull)
    completeness = (non_null / row_count) if row_count > 0 else 0.0
    try:
        unique_count = int(series.nunique(dropna=True))
    except TypeError:
        # Handle unhashable objects (e.g., dict/list) by counting uniques on JSON-safe strings
        unique_count = int(s_nonnull.map(_to_json_safe).astype(str).nunique())

    unique_factor = (unique_count / row_count) if row_count > 0 else 0.0

    inferred_type = _infer_type(series, s_nonnull=s_nonnull)

    # ------------------------------------------------------------------
    # Optional profiler signals (optional / non-breaking)
    # ------------------------------------------------------------------
    actual_dtype = str(series.dtype)

    # Strong TYPE_ERROR hint: inferred Numeric/Datetime but stored as object/string
    type_mismatch = bool(
        inferred_type in ("Numeric", "Datetime")
        and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
    )

    # Random sample values (more representative than top-5 frequent)
    random_sample_values = None
    if non_null > 0:
        try:
            random_sample_values = [
                _to_json_safe(v)
                for v in s_nonnull.sample(n=min(5, non_null), random_state=42).tolist()
            ]
        except Exception:
            random_sample_values = [_to_json_safe(v) for v in s_nonnull.head(5).tolist()]

    # Coercion failures (count + sample bad values)
    coercion_failure_count = None
    coercion_failure_samples = None

    # Percentiles for numeric columns (IQR reasoning)
    q1 = None
    q3 = None

    # Datetime competing format samples
    datetime_format_samples = None

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

    s_num_clean: pd.Series | None = None
    if inferred_type == "Numeric":
        s_num = pd.to_numeric(series, errors="coerce")
        original_nan = row_count - non_null
        inf_values = int(np.isinf(s_num).sum()) if len(s_num) else 0
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
        # Count only inf values and coercion failures, NOT natural missing data.
        # AutoGluon handles NaN natively, so natural missingness is not a defect.
        coercion_failures = int(s_num.isna().sum()) - original_nan - inf_values
        # Percentiles (Q1/Q3) for numeric columns
        if s_num.notna().any():
            q1 = _safe_float(s_num.quantile(0.25))
            q3 = _safe_float(s_num.quantile(0.75))

        # Coercion failures: count + sample raw values
        coercion_failure_count = int(coercion_failures)
        try:
            coerced_nonnull = pd.to_numeric(s_nonnull, errors="coerce")
            bad_mask = coerced_nonnull.isna()
            if bad_mask.any():
                coercion_failure_samples = (
                    s_nonnull[bad_mask].astype(str).drop_duplicates().head(5).tolist()
                )
            else:
                coercion_failure_samples = []
        except Exception:
            coercion_failure_samples = []
        assert coercion_failures >= 0, (
            f"Column '{col}': negative coercion_failures={coercion_failures} "
            f"(isna={int(s_num.isna().sum())}, original_nan={original_nan}, inf={inf_values})"
        )
        inf_nan_count = inf_values + coercion_failures
        s_num_clean = s_num.dropna()  # precomputed for _semantic_warnings

    top_vals = _top_frequent_values(series, k=5, s_nonnull=s_nonnull)

    future_date_count = None
    if inferred_type == "Datetime":
        datetime_format_consistency, earliest_date, latest_date = _datetime_consistency(series)
        future_date_count = _count_future_dates(s_nonnull)

        # Datetime: sample competing raw formats
        if non_null > 0:
            dt_strings = s_nonnull.astype(str)
            try:
                datetime_format_samples = (
                    dt_strings.sample(n=min(5, len(dt_strings)), random_state=42)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )
            except Exception:
                datetime_format_samples = dt_strings.drop_duplicates().head(5).tolist()

            # coercion failures for datetime parsing (count + samples)
            try:
                try:
                    parsed = pd.to_datetime(dt_strings.head(200), errors="coerce", format="mixed")
                except TypeError:
                    # Older pandas fallback (no format="mixed")
                    parsed = pd.to_datetime(dt_strings.head(200), errors="coerce")
                bad_mask = parsed.isna()
                coercion_failure_count = int(bad_mask.sum())
                if bad_mask.any():
                    coercion_failure_samples = (
                        dt_strings.head(200)[bad_mask].drop_duplicates().head(5).tolist()
                    )
                else:
                    coercion_failure_samples = []
            except Exception:
                coercion_failure_count = None
                coercion_failure_samples = None

    # Regex/pattern consistency for string/categorical-like columns
    if inferred_type == "Categorical" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        regex_format_consistency, dominant_pattern = _regex_consistency(series)
    issues = _semantic_warnings(series, inferred_type, unique_factor, s_nonnull=s_nonnull, s_num_clean=s_num_clean, future_date_count=future_date_count)

    return ColumnProfile(
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
        future_date_count=future_date_count,
        regex_format_consistency=regex_format_consistency,
        dominant_pattern=dominant_pattern,
        ydata_metrics=ydata_col_metrics if detailed_profiler else None,

        # Optional profiler additions (all optional)
        actual_dtype=actual_dtype,
        type_mismatch=type_mismatch,
        coercion_failure_count=coercion_failure_count,
        coercion_failure_samples=coercion_failure_samples,
        random_sample_values=random_sample_values,
        q1=q1,
        q3=q3,
        datetime_format_samples=datetime_format_samples,
        top_frequent_values=top_vals,
        semantic_warnings=issues,
    )


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
        col_profile = profile_one_column(
            df,
            col_name=str(col),
            row_count=row_count,
            detailed_profiler=detailed_profiler,
            ydata_col_metrics=ydata_by_col.get(str(col)) if detailed_profiler else None,
        )
        if detailed_profiler:
            # Global ydata-bootstrap warnings: applied per-column for backward
            # compatibility, but they describe the whole-dataframe ydata run,
            # not anything column-specific. Keeping them out of profile_one_column
            # so it can stay a pure per-column function.
            if ydata_error:
                col_profile.semantic_warnings.append("ydata_fallback_to_baseline")
            elif ydata_skipped == "wide_table":
                col_profile.semantic_warnings.append("ydata_skipped_wide_table")
        columns.append(col_profile)

    profile = DatasetProfile(row_count=row_count, columns=columns)
    # Return plain dict for JSON serialization
    result = profile.model_dump()

    return result


def extend_profile(
    existing_profile: Dict[str, Any],
    new_df: pd.DataFrame,
    added_cols: List[str],
    *,
    detailed_profiler: bool = False,
    target_column: str | None = None,
) -> Dict[str, Any]:
    """
    Produce a new profile dict for ``new_df`` by reusing all entries from
    ``existing_profile`` unchanged and computing fresh ColumnProfile entries
    only for ``added_cols``.

    Preconditions (checked):
      - every name in ``added_cols`` is present in new_df.columns
      - every column already in existing_profile is still present in new_df
      - len(new_df) equals existing_profile['row_count']

    This function is ONLY appropriate when the transformation between the
    previous df and new_df was strictly additive (new columns added, no
    existing column mutated, no rows dropped). The caller (currently the
    feature_engineering round-end reprofile) is responsible for enforcing
    that contract; if there's any doubt, fall back to generate_profile.
    """
    missing = [c for c in added_cols if c not in new_df.columns]
    if missing:
        raise KeyError(f"added_cols not in new_df: {missing}")

    existing_row_count = existing_profile.get("row_count", len(new_df))
    if len(new_df) != existing_row_count:
        raise ValueError(
            f"extend_profile expects same row count "
            f"(existing={existing_row_count}, new={len(new_df)}). "
            f"Use refresh_profile_for_recipe or generate_profile instead."
        )

    existing_names = {c["name"] for c in existing_profile["columns"]}
    for name in existing_names:
        if name not in new_df.columns:
            raise ValueError(
                f"extend_profile expects all existing columns to remain in new_df; "
                f"missing '{name}'. Use generate_profile instead."
            )

    # Reuse existing entries verbatim
    columns = list(existing_profile["columns"])

    # Compute and append new ones (FE rounds never pass ydata through, so
    # ydata_col_metrics is always None on this path)
    row_count = int(len(new_df))
    for name in added_cols:
        col_profile = profile_one_column(
            new_df,
            col_name=name,
            row_count=row_count,
            detailed_profiler=detailed_profiler,
            ydata_col_metrics=None,
        )
        columns.append(col_profile.model_dump())

    out = dict(existing_profile)
    out["row_count"] = row_count
    out["columns"] = columns
    return out
