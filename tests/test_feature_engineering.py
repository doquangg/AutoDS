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


def test_run_fe_planner_agent_includes_findings_in_user_content(monkeypatch):
    """
    Smoke test: the user message passed to the planner LLM must contain
    the findings block. We stub the LLM so this test doesn't hit the network.
    """
    from core.agents import feature_engineering as fe_mod
    from core.schemas import FeatureProposal, InvestigationFindings, SemanticViolation

    captured_messages = {}

    class _StubResponse:
        content = ""
        tool_calls = []

    class _StubLLM:
        def bind_tools(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            # Snapshot, not a reference — the caller mutates `messages` after
            # invoke returns (appending the response), which would otherwise
            # shift the HumanMessage off [-1].
            captured_messages["messages"] = list(messages)
            return _StubResponse()

    def _stub_extract(response, messages, llm, schema, at_limit, has_tool_calls):
        return FeatureProposal(
            round_number=1, ideas=[], no_more_features=True, reasoning="stub"
        )

    monkeypatch.setattr(fe_mod, "get_fe_planner_llm", lambda: _StubLLM())
    monkeypatch.setattr(fe_mod, "extract_structured_output", _stub_extract)
    monkeypatch.setattr(fe_mod, "set_working_df", lambda df: None)

    import pandas as pd
    df = pd.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30]})

    findings = InvestigationFindings(
        target_column="income",
        task_type="regression",
        violations=[
            SemanticViolation(
                violation_id=1,
                severity="CRITICAL",
                category="SENTINEL_VALUE",
                affected_columns=["age"],
                description="sentinel -1 in age",
                evidence="min=-1",
                suggested_action="replace with NaN",
            )
        ],
        columns_to_drop=["ssn"],
    )

    state = {
        "user_query": "predict income",
        "target_column": "income",
        "investigation_findings": findings,
    }

    fe_mod.run_fe_planner_agent(
        state=state, df=df, profile={}, round_num=1, applied_so_far=[]
    )

    human_msg = captured_messages["messages"][-1].content
    assert "CRITICAL VIOLATIONS" in human_msg
    assert "SENTINEL_VALUE" in human_msg
    assert "ssn" in human_msg
    # Findings block appears before the profile block, so the planner sees it
    # as context on the same level as task type, not as an afterthought.
    assert human_msg.index("CRITICAL VIOLATIONS") < human_msg.index("DATASET PROFILE")
