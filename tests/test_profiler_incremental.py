"""
Tests for the incremental profiling helpers extracted from generate_profile.

The contract: profile_one_column / extend_profile / refresh_profile_for_recipe
must each produce output that is byte-equal to what generate_profile would
produce on the same data. generate_profile remains the ground-truth path.
"""

import pandas as pd

from plugins.profiler import generate_profile, profile_one_column


def _sample_df():
    return pd.DataFrame({
        "age": [25, 30, -1, 45, 50, -1, 60, 70, 80, 90],
        "income": [30000, 40000, 50000, None, 70000, 80000, 90000, 1e6, 110000, 120000],
        "city": ["NY", "LA", "SF", "NY", "LA", "SF", "NY", "LA", "SF", "NY"],
        "signup_date": pd.to_datetime([
            "2020-01-01", "2020-02-15", "2020-03-01", "2020-04-01", "2020-05-01",
            "2020-06-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01",
        ]),
    })


def test_profile_one_column_matches_generate_profile_per_column():
    df = _sample_df()
    full = generate_profile(df, detailed_profiler=False, target_column="income")

    for col_profile in full["columns"]:
        name = col_profile["name"]
        one = profile_one_column(
            df,
            col_name=name,
            row_count=len(df),
            detailed_profiler=False,
            ydata_col_metrics=None,
        )
        assert one.model_dump() == col_profile, f"mismatch on column {name}"
