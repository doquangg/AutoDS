"""
Feature Engineering (post-cleaning, pre-modeling).

Goal:
- Add safe, bounded, leakage-avoiding features to improve downstream modeling.
- Must be deterministic by default (no LLM required).
- Must be capped by max rounds + max new features per round to avoid feature explosion.

This module is called by the pipeline stage we will add in core/pipeline/graph.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# Hard caps (can be tuned later)
DEFAULT_MAX_FE_ROUNDS = 3
DEFAULT_MAX_NEW_FEATURES_PER_ROUND = 30
DEFAULT_SAMPLE_UNIQUE_CAP = 50  # cap for high-cardinality string buckets


@dataclass
class FEResult:
    df: pd.DataFrame
    new_features: List[str]
    dropped_features: List[str]
    notes: List[str]


def engineer_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    round_id: int = 0,
    max_new_features: int = DEFAULT_MAX_NEW_FEATURES_PER_ROUND,
) -> FEResult:
    """
    Deterministic, safe feature engineering.

    Rules:
    - Do NOT use target_column to create features (no leakage).
    - Only transform feature columns.
    - Keep feature count bounded by max_new_features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("engineer_features expects a pandas.DataFrame")

    df_out = df.copy()
    new_features: List[str] = []
    dropped_features: List[str] = []
    notes: List[str] = []

    # Helper: add a column only if we haven't exceeded the cap
    def _add(name: str, values: Any) -> None:
        nonlocal df_out, new_features
        if name in df_out.columns:
            return
        if len(new_features) >= max_new_features:
            return
        df_out[name] = values
        new_features.append(name)

    # Identify columns (exclude target if provided)
    feature_cols = [c for c in df_out.columns if c != target_column]

    # -----------------------------
    # 1) Missingness indicators
    # -----------------------------
    for c in feature_cols:
        if df_out[c].isna().any():
            _add(f"{c}__is_missing", df_out[c].isna().astype(int))

    # -----------------------------
    # 2) Datetime decompositions
    # -----------------------------
    for c in feature_cols:
        s = df_out[c]
        # Only attempt if dtype already datetime or looks like dates in string
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        else:
            # cheap parse attempt on a small sample
            sample = s.dropna().astype(str).head(50)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() < 0.6:
                continue
            dt = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")

        _add(f"{c}__year", dt.dt.year)
        _add(f"{c}__month", dt.dt.month)
        _add(f"{c}__day", dt.dt.day)
        _add(f"{c}__dayofweek", dt.dt.dayofweek)
        _add(f"{c}__is_weekend", dt.dt.dayofweek.isin([5, 6]).astype(int))

    # -----------------------------
    # 3) Numeric transforms (safe)
    # -----------------------------
    for c in feature_cols:
        s = df_out[c]
        if pd.api.types.is_numeric_dtype(s):
            # log1p for positive-skewed values (only if all >= 0)
            s_num = pd.to_numeric(s, errors="coerce")
            if (s_num.dropna() >= 0).all():
                # only add if skew seems meaningful
                try:
                    if abs(s_num.dropna().skew()) >= 1.0:
                        _add(f"{c}__log1p", np.log1p(s_num))
                except Exception:
                    pass

            # simple winsor-like clip bounds (Q1/Q3 based) as an extra stabilized version
            try:
                q1 = s_num.quantile(0.25)
                q3 = s_num.quantile(0.75)
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    lo = q1 - 1.5 * iqr
                    hi = q3 + 1.5 * iqr
                    _add(f"{c}__iqr_clipped", s_num.clip(lower=lo, upper=hi))
            except Exception:
                pass

    # -----------------------------
    # 4) Simple string features
    # -----------------------------
    for c in feature_cols:
        s = df_out[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s_str = s.astype("string")
            # length feature
            _add(f"{c}__len", s_str.fillna("").str.len())

            # very cheap "bucket" for low-cardinality strings
            try:
                vc = s_str.dropna().value_counts()
                if 2 <= len(vc) <= DEFAULT_SAMPLE_UNIQUE_CAP:
                    # bucketize by top values; rare -> "__OTHER__"
                    top = set(vc.head(10).index.tolist())
                    bucket = s_str.fillna("__MISSING__").apply(lambda x: x if x in top else "__OTHER__")
                    _add(f"{c}__bucket_top10", bucket)
            except Exception:
                pass

    notes.append(f"FE round {round_id}: added {len(new_features)} features (cap={max_new_features}).")
    return FEResult(df=df_out, new_features=new_features, dropped_features=dropped_features, notes=notes)
