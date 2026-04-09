from core.agents.feature_engineering import _summarize_findings
from core.schemas import (
    InvestigationFindings,
    SemanticViolation,
    ColumnDropRationale,
)


def test_summarize_findings_none_returns_placeholder():
    result = _summarize_findings(None)
    assert "(investigation findings unavailable)" in result


def test_summarize_findings_empty_object_renders_skeleton():
    findings = InvestigationFindings()
    result = _summarize_findings(findings)
    # Empty but well-formed — no crash, no stray None.
    assert "CRITICAL VIOLATIONS" in result
    assert "COLUMNS MARKED FOR DROP" in result
    assert "KEY CAVEATS" in result
    assert "None" not in result.replace("None if", "")  # no stringified Nones


def test_summarize_findings_populated_renders_all_sections():
    findings = InvestigationFindings(
        target_column="income",
        task_type="regression",
        violations=[
            SemanticViolation(
                violation_id=1,
                severity="CRITICAL",
                category="SENTINEL_VALUE",
                affected_columns=["age"],
                description="Column age contains 847 rows with value -1.",
                evidence="min_value=-1",
                suggested_action="Replace -1 with NaN",
            ),
            SemanticViolation(
                violation_id=2,
                severity="INFO",
                category="OUTLIER",
                affected_columns=["salary"],
                description="A handful of salary values are high but plausible.",
                evidence="p99=450000",
                suggested_action="Leave alone",
            ),
        ],
        columns_to_drop=["ssn", "customer_id"],
        columns_to_drop_rationale=[
            ColumnDropRationale(column="ssn", reason="PII"),
            ColumnDropRationale(column="customer_id", reason="high-card ID"),
        ],
        key_caveats=["Age column had sentinel -1 values replaced with NaN."],
    )

    result = _summarize_findings(findings)

    # CRITICAL surfaced, INFO suppressed
    assert "SENTINEL_VALUE" in result
    assert "age" in result
    assert "OUTLIER" not in result  # INFO violations are intentionally hidden

    # Columns-to-drop rendered as a comma-separated list
    assert "ssn" in result
    assert "customer_id" in result

    # Caveats rendered
    assert "Age column had sentinel -1" in result


def test_summarize_findings_accepts_dict_shape():
    # Tolerance for dict-shaped findings mirrors _get_task_type.
    result = _summarize_findings({
        "violations": [{"severity": "CRITICAL", "category": "TYPE_ERROR",
                        "affected_columns": ["x"], "description": "wrong dtype"}],
        "columns_to_drop": ["z"],
        "key_caveats": [],
    })
    assert "TYPE_ERROR" in result
    assert "z" in result
