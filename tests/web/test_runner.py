"""
Runner integration tests using a stub LangGraph that is shaped exactly
like the real one for our purposes:
  - one node that emits a couple of log records
  - one node that calls interrupt() and then receives the resume value
"""
import asyncio
import logging

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from typing_extensions import TypedDict

from core.web.events import EventTypes
from core.web.runner import PipelineRunner
from core.web.session import SessionManager


class TinyState(TypedDict, total=False):
    user_query: str
    target_column: str | None
    answer: str | None


def _build_stub_graph():
    """A 2-node graph: noisy_node -> ask_target -> answer."""
    autods_log = logging.getLogger("autods")

    def noisy_node(state):
        autods_log.debug("ROUTE route_test \u2192 next (k=v)")
        autods_log.debug(
            "TOOL_CALL fake_tool\n  params: {\"x\": 1}\n  result: ok"
        )
        return {}

    def ask_target(state):
        if state.get("target_column"):
            return {}
        target = interrupt({
            "candidates": [{"name": "churned", "rationale": "binary"}],
            "all_columns": ["churned", "tenure"],
            "user_query": state["user_query"],
        })
        return {"target_column": target}

    def answer_node(state):
        return {"answer": f"target was {state['target_column']}"}

    g = StateGraph(TinyState)
    g.add_node("noisy", noisy_node)
    g.add_node("ask_target", ask_target)
    g.add_node("answer", answer_node)
    g.add_edge(START, "noisy")
    g.add_edge("noisy", "ask_target")
    g.add_edge("ask_target", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_runner_completes_with_interrupt_and_resume():
    mgr = SessionManager()
    session = mgr.create(b"a\n1\n", "x.csv", "predict churn?")
    runner = PipelineRunner(session=session, graph_app=_build_stub_graph())

    # Run the pipeline as a background task so we can resume mid-run
    task = asyncio.create_task(
        runner.run(initial_state={"user_query": "predict churn?"})
    )

    # Wait for the runner to pause
    for _ in range(50):
        await asyncio.sleep(0.05)
        if session.status == "paused":
            break
    assert session.status == "paused"
    assert session.paused_for == "target_selection"
    assert session.pause_payload["candidates"][0]["name"] == "churned"

    # Resume with a selection
    mgr.resume(session.id, target_column="churned")

    # Wait for completion
    await asyncio.wait_for(task, timeout=5)
    assert session.status == "complete"

    # Check we emitted the lifecycle events
    types = [e["type"] for e in session.event_buffer]
    assert EventTypes.SESSION_STARTED in types
    assert EventTypes.NODE_STARTED in types
    assert EventTypes.NODE_COMPLETED in types
    assert EventTypes.TARGET_SELECTION_REQUIRED in types
    assert EventTypes.TARGET_SELECTION_RESOLVED in types
    assert EventTypes.PIPELINE_COMPLETE in types

    # Logger sink events also flowed through
    assert EventTypes.ROUTE_DECISION in types
    assert EventTypes.TOOL_CALL in types


@pytest.mark.asyncio
async def test_runner_emits_pipeline_failed_on_exception():
    """If a node raises, we surface pipeline_failed."""

    def boom(state):
        raise RuntimeError("kaboom")

    g = StateGraph(TinyState)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    app = g.compile(checkpointer=MemorySaver())

    mgr = SessionManager()
    session = mgr.create(b"a\n1\n", "x.csv", "q")
    runner = PipelineRunner(session=session, graph_app=app)

    await runner.run(initial_state={"user_query": "q"})
    assert session.status == "failed"
    types = [e["type"] for e in session.event_buffer]
    assert EventTypes.PIPELINE_FAILED in types
    assert "kaboom" in session.error
