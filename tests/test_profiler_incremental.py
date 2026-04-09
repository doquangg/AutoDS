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
    refresh_profile_for_recipe,
)


class _FakeStep:
    """Duck-typed CleaningStep for tests (avoids importing the real schema)."""

    def __init__(self, operation, target_column):
        self.operation = operation
        self.target_column = target_column


class _FakeRecipe:
    def __init__(self, steps):
        self.steps = steps


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


def test_refresh_profile_with_cast_type_only_recomputes_target_column():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.copy()
    df2["age"] = df2["age"].astype(float) + 0.0  # dtype flip, values preserved
    recipe = _FakeRecipe([_FakeStep("CAST_TYPE", "age")])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )

    # Columns other than "age" must be byte-equal to base
    base_by_name = {c["name"]: c for c in base["columns"]}
    for col in refreshed["columns"]:
        if col["name"] != "age":
            assert col == base_by_name[col["name"]]

    # "age" must equal what a full profile would produce on df2.
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    full_age = next(c for c in full["columns"] if c["name"] == "age")
    ref_age = next(c for c in refreshed["columns"] if c["name"] == "age")
    assert ref_age == full_age


def test_refresh_profile_drops_rows_falls_back_to_full_profile():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.iloc[:5].copy()  # rows dropped
    recipe = _FakeRecipe([_FakeStep("DROP_ROWS", "income")])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )

    # Must match a full profile byte-for-byte
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    assert refreshed["row_count"] == full["row_count"]
    assert refreshed["columns"] == full["columns"]


def test_refresh_profile_drop_column_removes_entry_without_recompute():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.drop(columns=["city"])
    recipe = _FakeRecipe([_FakeStep("DROP_COLUMN", "city")])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )
    names = [c["name"] for c in refreshed["columns"]]
    assert "city" not in names
    # Other columns unchanged
    base_other = {c["name"]: c for c in base["columns"] if c["name"] != "city"}
    ref_by_name = {c["name"]: c for c in refreshed["columns"]}
    for name, col in base_other.items():
        assert ref_by_name[name] == col


def test_refresh_profile_custom_code_on_unknown_column_falls_back():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.copy()
    df2["income"] = df2["income"].fillna(0)  # custom code touched income
    # Intentionally pass target_column=None to simulate an ambiguous scope
    recipe = _FakeRecipe([_FakeStep("CUSTOM_CODE", None)])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )
    # Fallback should be byte-equal to a full profile
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    assert refreshed["columns"] == full["columns"]


def test_refresh_profile_custom_code_on_known_column_recomputes_only_that_column():
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.copy()
    df2["income"] = df2["income"].fillna(0)
    recipe = _FakeRecipe([_FakeStep("CUSTOM_CODE", "income")])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )

    # Other columns byte-equal to base
    base_by_name = {c["name"]: c for c in base["columns"]}
    for col in refreshed["columns"]:
        if col["name"] != "income":
            assert col == base_by_name[col["name"]]

    # "income" matches a full profile of df2
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    full_income = next(c for c in full["columns"] if c["name"] == "income")
    ref_income = next(c for c in refreshed["columns"] if c["name"] == "income")
    assert ref_income == full_income


def test_refresh_profile_picks_up_invented_column_via_custom_code():
    """If CUSTOM_CODE creates a brand-new column, refresh must profile it."""
    df1 = _sample_df()
    base = generate_profile(df1, detailed_profiler=False, target_column="income")

    df2 = df1.copy()
    df2["age_bucket"] = (df2["age"] // 10).astype(int)  # invented column
    # Step targets a known column that wasn't touched, so dirty_cols is empty
    # for the named target — but new_df has a column not in the existing
    # profile, which the helper should pick up via the "invented" branch.
    recipe = _FakeRecipe([_FakeStep("CUSTOM_CODE", "age")])

    refreshed = refresh_profile_for_recipe(
        base,
        df1,
        df2,
        recipe,
        detailed_profiler=False,
        target_column="income",
    )
    names = {c["name"] for c in refreshed["columns"]}
    assert "age_bucket" in names
    full = generate_profile(df2, detailed_profiler=False, target_column="income")
    full_by_name = {c["name"]: c for c in full["columns"]}
    ref_by_name = {c["name"]: c for c in refreshed["columns"]}
    assert ref_by_name["age_bucket"] == full_by_name["age_bucket"]
