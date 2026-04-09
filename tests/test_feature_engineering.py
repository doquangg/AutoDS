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
