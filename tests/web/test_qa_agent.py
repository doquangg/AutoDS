"""
Tests for the QA agent's prompt assembly and the streaming wrapper.
We patch the ChatOpenAI factory so no real network call happens.
"""
from unittest.mock import MagicMock, patch

import pytest

from core.agents.qa_agent import build_qa_messages
from core.web.events import EventTypes
from core.web.qa import stream_qa_answer
from core.web.session import Session


def _artifacts() -> dict:
    return {
        "user_query": "what drives bills?",
        "target_column": "total_bill",
        "investigation_findings": {
            "violations": [
                {"violation_id": "v1", "description": "missing values in age"}
            ],
            "key_caveats": ["age column was 30% null and imputed"],
        },
        "applied_steps": [{
            "step_id": 1,
            "operation": "DROP_COLUMN",
            "target_column": "ssn",
            "justification": "PII",
        }],
        "applied_fe_steps": [],
        "model_metadata": {
            "best_model": "WeightedEnsemble",
            "eval_metrics": {"r2": 0.81},
        },
        "final_answer": "Age and prior-visit count drive bills.",
    }


def test_build_qa_messages_includes_all_artifacts():
    msgs = build_qa_messages(_artifacts(), [], "why did you drop ssn?")
    body = "\n".join(m.content for m in msgs)
    assert "ssn" in body
    assert "total_bill" in body
    assert "WeightedEnsemble" in body
    assert "why did you drop ssn?" in body


def test_build_qa_messages_appends_history():
    history = [("user", "first q"), ("assistant", "first a")]
    msgs = build_qa_messages(_artifacts(), history, "follow-up")
    body = "\n".join(m.content for m in msgs)
    assert "first q" in body
    assert "first a" in body
    assert "follow-up" in body


@pytest.mark.asyncio
async def test_stream_qa_answer_yields_token_events():
    s = Session(
        id="x",
        user_query="q",
        dataset_filename="x.csv",
        csv_bytes=b"",
        artifacts=_artifacts(),
    )

    async def fake_stream(_messages):
        for tok in ["hello", " ", "world"]:
            chunk = MagicMock()
            chunk.content = tok
            yield chunk

    mock_llm = MagicMock()
    mock_llm.astream = fake_stream

    with patch("core.agents.qa_agent.get_qa_llm", return_value=mock_llm):
        events = []
        async for ev in stream_qa_answer(s, "why?"):
            events.append(ev)

    types = [e["type"] for e in events]
    assert EventTypes.QA_TOKEN in types
    assert EventTypes.QA_COMPLETE in types
    tokens = "".join(
        e["text"] for e in events if e["type"] == EventTypes.QA_TOKEN
    )
    assert tokens == "hello world"
    # History was persisted on the session
    assert s.qa_messages[-1]["role"] == "assistant"
    assert s.qa_messages[-1]["content"] == "hello world"
