"""Standalone verification of the slim-profile overhaul.

Run: python scripts/verify_profile_views.py
Exit code 0 if all invariants hold.
"""

import json
import sys
from pathlib import Path

# Repo root on sys.path so this script can be invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from plugins.profiler import generate_profile
from plugins.profile_views import (
    project_profile,
    profile_to_llm_json,
    CORE_COLUMN_FIELDS,
    STATS_COLUMN_FIELDS,
    FORENSIC_COLUMN_FIELDS,
)


def main() -> int:
    df = pd.DataFrame(
        {
            "age": [1, 2, 3, 4, 5, 999, None],
            "salary": [10000.0, 20000.5, 30000.0, -1.0, 50000.0, None, 70000.0],
            "name": ["a", "b", "c", "d", "e", "f", "g"],
            "joined": ["2020-01-01", "2021-02-03", None, "2019-12-31", None, None, None],
        }
    )

    profile = generate_profile(df, detailed_profiler=False, target_column="age")

    # 1. AlgorithmicQualityScore must be gone.
    assert "algorithmic_quality_score" not in profile, "quality_score should be deleted"

    # 2. Projections are strictly nested.
    core = project_profile(profile, "core")
    stats = project_profile(profile, "stats")
    forensic = project_profile(profile, "forensic")

    core_col = core["columns"][0]
    stats_col = stats["columns"][0]
    forensic_col = forensic["columns"][0]

    assert set(core_col.keys()).issubset(set(stats_col.keys())), (
        f"core keys not a subset of stats: {set(core_col.keys()) - set(stats_col.keys())}"
    )
    assert set(stats_col.keys()).issubset(set(forensic_col.keys())), (
        f"stats keys not a subset of forensic: {set(stats_col.keys()) - set(forensic_col.keys())}"
    )

    # 3. Stats view has numeric stats, core view does not.
    assert "mean" not in core_col or core_col.get("mean") is None
    numeric_stats_col = next(
        (c for c in stats["columns"] if c["name"] in ("age", "salary")), None
    )
    assert numeric_stats_col is not None
    assert "mean" in numeric_stats_col

    # 4. Forensic view has ydata_metrics or random_sample_values when they exist.
    forensic_age = next((c for c in forensic["columns"] if c["name"] == "age"), None)
    assert forensic_age is not None
    assert "random_sample_values" in forensic_age

    # 5. LLM serialization strips indent.
    core_json = profile_to_llm_json(profile, "core")
    stats_json = profile_to_llm_json(profile, "stats")
    forensic_json = profile_to_llm_json(profile, "forensic")

    for label, js in (("core", core_json), ("stats", stats_json), ("forensic", forensic_json)):
        assert "\n" not in js, f"{label} view has newlines"
        assert ", " not in js, f"{label} view has loose separators"

    # 6. Size is strictly monotone (core < stats < forensic).
    assert len(core_json) < len(stats_json) < len(forensic_json), (
        f"sizes not monotone: core={len(core_json)} stats={len(stats_json)} forensic={len(forensic_json)}"
    )

    # 7. Floats are rounded.
    salary_stats = next((c for c in stats["columns"] if c["name"] == "salary"), None)
    assert salary_stats is not None
    mean = salary_stats.get("mean")
    assert mean is not None
    # 4 sig figs on a value like 35999.7083... should round to ~36000 or 3.6e+04.
    assert len(str(mean).replace(".", "").replace("-", "").rstrip("0").lstrip("0")) <= 4, (
        f"mean not rounded to 4 sig figs: {mean}"
    )

    print("All profile-view invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
