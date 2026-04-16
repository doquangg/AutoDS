"""
Tests for the QA agent's prompt assembly and the streaming wrapper.
We patch the ChatOpenAI factory so no real network call happens.
"""
from unittest.mock import MagicMock, patch

import pytest

from core.agents.qa_agent import build_qa_messages
from core.web.events import EventTypes
from core.web.qa import run_qa_task
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
async def test_run_qa_task_emits_token_and_complete_events():
    """run_qa_task pushes QA_TOKEN events per chunk and a final QA_COMPLETE
    onto the session's shared queue + ring buffer, using the session's seq
    counter. No generator return value — the caller awaits the task."""
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
        await run_qa_task(s, "why?")

    types = [e["type"] for e in s.event_buffer]
    assert EventTypes.QA_TOKEN in types
    assert EventTypes.QA_COMPLETE in types
    tokens = "".join(
        e["text"] for e in s.event_buffer if e["type"] == EventTypes.QA_TOKEN
    )
    assert tokens == "hello world"
    # Events drained from queue match what's in the buffer (same seqs)
    queued_seqs: list[int] = []
    while not s.queue.empty():
        queued_seqs.append(s.queue.get_nowait()["seq"])
    assert queued_seqs == [e["seq"] for e in s.event_buffer]
    # History was persisted on the session
    assert s.qa_messages[-1]["role"] == "assistant"
    assert s.qa_messages[-1]["content"] == "hello world"


@pytest.mark.asyncio
async def test_run_qa_task_emits_qa_error_on_llm_failure():
    """If the LLM raises, emit QA_ERROR instead of letting the exception
    propagate and silently end the session stream."""
    s = Session(
        id="x",
        user_query="q",
        dataset_filename="x.csv",
        csv_bytes=b"",
        artifacts=_artifacts(),
    )

    async def boom(_messages):
        raise RuntimeError("model not found")
        yield  # pragma: no cover — makes this an async generator

    mock_llm = MagicMock()
    mock_llm.astream = boom

    with patch("core.agents.qa_agent.get_qa_llm", return_value=mock_llm):
        await run_qa_task(s, "why?")

    types = [e["type"] for e in s.event_buffer]
    assert EventTypes.QA_ERROR in types
    assert EventTypes.QA_COMPLETE not in types
    err = next(e for e in s.event_buffer if e["type"] == EventTypes.QA_ERROR)
    assert "model not found" in err["error"]


@pytest.mark.asyncio
async def test_run_qa_task_emits_qa_error_on_empty_response():
    """An LLM that yields nothing is almost always a misconfiguration;
    emit QA_ERROR so the UI can explain instead of showing nothing."""
    s = Session(
        id="x",
        user_query="q",
        dataset_filename="x.csv",
        csv_bytes=b"",
        artifacts=_artifacts(),
    )

    async def empty(_messages):
        if False:
            yield  # pragma: no cover — keeps this an async generator
        return

    mock_llm = MagicMock()
    mock_llm.astream = empty

    with patch("core.agents.qa_agent.get_qa_llm", return_value=mock_llm):
        await run_qa_task(s, "why?")

    types = [e["type"] for e in s.event_buffer]
    assert EventTypes.QA_ERROR in types
    assert EventTypes.QA_COMPLETE not in types
