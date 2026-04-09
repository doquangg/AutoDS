"""
Tests for the incremental profiling helpers extracted from generate_profile.

The contract: profile_one_column / extend_profile / refresh_profile_for_recipe
must each produce output that is byte-equal to what generate_profile would
produce on the same data. generate_profile remains the ground-truth path.
"""

import pandas as pd

from plugins.profiler import (
    extend_profile,
    generate_profile,
    profile_one_column,
)


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


def test_extend_profile_appends_new_columns_without_recomputing_old_ones():
    df1 = _sample_df()
    base_profile = generate_profile(df1, detailed_profiler=False, target_column="income")

    # Simulate an FE round that added two new columns.
    df2 = df1.copy()
    df2["age_squared"] = df2["age"] ** 2
    df2["city_freq"] = df2["city"].map(df2["city"].value_counts())

    extended = extend_profile(
        base_profile,
        df2,
        added_cols=["age_squared", "city_freq"],
        detailed_profiler=False,
        target_column="income",
    )

    # 1) Every original column entry must be byte-equal to the base profile's
    #    entry — proves we reused it instead of recomputing.
    base_by_name = {c["name"]: c for c in base_profile["columns"]}
    for col in extended["columns"]:
        if col["name"] in base_by_name:
            assert col == base_by_name[col["name"]], (
                f"existing column {col['name']} was unexpectedly recomputed"
            )

    # 2) The new columns must be present and equal to what a full profile
    #    would produce.
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    full_by_name = {c["name"]: c for c in full["columns"]}
    extended_names = {c["name"] for c in extended["columns"]}
    for new_col in ("age_squared", "city_freq"):
        assert new_col in extended_names
        ext_entry = next(c for c in extended["columns"] if c["name"] == new_col)
        assert ext_entry == full_by_name[new_col]

    # 3) row_count must be correct on the extended profile.
    assert extended["row_count"] == len(df2)


def test_extend_profile_rejects_added_cols_not_in_df():
    df = _sample_df()
    base = generate_profile(df, detailed_profiler=False, target_column="income")
    try:
        extend_profile(
            base,
            df,
            added_cols=["not_a_column"],
            detailed_profiler=False,
            target_column="income",
        )
    except KeyError:
        return
    assert False, "expected KeyError for missing column"


def test_extend_profile_rejects_row_count_mismatch():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")
    df2 = df1.iloc[:5].copy()
    df2["age_squared"] = df2["age"] ** 2
    try:
        extend_profile(
            base,
            df2,
            added_cols=["age_squared"],
            detailed_profiler=False,
            target_column="income",
        )
    except ValueError:
        return
    assert False, "expected ValueError for row count mismatch"
