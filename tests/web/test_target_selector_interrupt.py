"""
Target selector branching: CLI mode unchanged, web mode calls interrupt().

We don't drive the full LangGraph here; we call select_target_column directly
with a hand-built state and inspect the side effects. The web branch should
call langgraph.types.interrupt(...) with the candidate/column payload. We
patch interrupt() to simulate the resume return value and assert the call
args.
"""
import os
from unittest.mock import patch

import pytest

from core.agents.target_selector import select_target_column


def _state(target=None) -> dict:
    return {
        "user_query": "what predicts churn?",
        "target_column": target,
        "profile": {
            "row_count": 100,
            "columns": [
                {"name": "id", "inferred_type": "int"},
                {"name": "churned", "inferred_type": "bool"},
                {"name": "tenure_months", "inferred_type": "int"},
            ],
        },
    }


def test_already_set_is_noop(monkeypatch):
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "web")
    assert select_target_column(_state(target="churned")) == {}


def test_auto_target_env_var_wins_in_web_mode(monkeypatch):
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "web")
    monkeypatch.setenv("AUTO_TARGET_COLUMN", "churned")
    assert select_target_column(_state())["target_column"] == "churned"


def test_web_mode_calls_interrupt_with_payload(monkeypatch):
    """In web mode without AUTO_TARGET_COLUMN, call interrupt() with the
    candidates/columns payload and use its return value as the selection."""
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "web")
    monkeypatch.delenv("AUTO_TARGET_COLUMN", raising=False)

    # Stub the LLM ranking so we don't hit the network
    from core.agents import target_selector as ts
    fake_candidates = [
        ts.TargetCandidate(name="churned", rationale="binary outcome"),
        ts.TargetCandidate(name="tenure_months", rationale="continuous"),
    ]
    with patch.object(ts, "rank_target_candidates", return_value=fake_candidates), \
         patch.object(ts, "interrupt", return_value="churned") as mock_interrupt:
        result = select_target_column(_state())

        assert result == {"target_column": "churned"}
        assert mock_interrupt.call_count == 1
        payload = mock_interrupt.call_args.args[0]
        assert "candidates" in payload
        assert payload["candidates"][0]["name"] == "churned"
        assert "all_columns" in payload
        assert "churned" in payload["all_columns"]
        assert payload["user_query"] == "what predicts churn?"


def test_web_mode_rejects_invalid_selection(monkeypatch):
    """If the resumed selection isn't a known column, raise ValueError."""
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "web")
    monkeypatch.delenv("AUTO_TARGET_COLUMN", raising=False)

    from core.agents import target_selector as ts
    fake_candidates = [ts.TargetCandidate(name="churned", rationale="binary")]
    with patch.object(ts, "rank_target_candidates", return_value=fake_candidates), \
         patch.object(ts, "interrupt", return_value="not_a_column"):
        with pytest.raises(ValueError, match="not found in dataset columns"):
            select_target_column(_state())


def test_cli_mode_calls_prompt(monkeypatch):
    """Default mode still uses the CLI prompt path."""
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "cli")
    monkeypatch.delenv("AUTO_TARGET_COLUMN", raising=False)

    from core.agents import target_selector as ts
    fake_candidates = [
        ts.TargetCandidate(name="churned", rationale="binary"),
    ]
    with patch.object(ts, "rank_target_candidates", return_value=fake_candidates), \
         patch.object(ts, "prompt_user_selection", return_value="churned") as p:
        result = select_target_column(_state())
        assert result == {"target_column": "churned"}
        p.assert_called_once()
