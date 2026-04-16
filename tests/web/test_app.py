"""HTTP-level tests using FastAPI's AsyncClient/ASGITransport."""
import asyncio

import pytest
from httpx import AsyncClient, ASGITransport
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from core.web.app import create_app


class TinyState(TypedDict, total=False):
    user_query: str


def _trivial_graph():
    def node(state):
        return {}
    g = StateGraph(TinyState)
    g.add_node("trivial", node)
    g.add_edge(START, "trivial")
    g.add_edge("trivial", END)
    return g.compile(checkpointer=MemorySaver())


@pytest.fixture
def app():
    return create_app(graph_app=_trivial_graph())


@pytest.mark.asyncio
async def test_post_sessions_creates_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
        data = {"user_query": "what predicts churn?"}
        r = await client.post("/sessions", files=files, data=data)
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body


@pytest.mark.asyncio
async def test_post_sessions_rejects_non_csv(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("data.txt", b"not csv", "text/plain")}
        data = {"user_query": "q"}
        r = await client.post("/sessions", files=files, data=data)
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_get_session_returns_metadata(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("data.csv", b"a\n1\n", "text/csv")}
        r = await client.post("/sessions", files=files, data={"user_query": "q"})
        sid = r.json()["session_id"]
        r2 = await client.get(f"/sessions/{sid}")
        assert r2.status_code == 200
        meta = r2.json()
        assert meta["session_id"] == sid
        assert meta["status"] in {"pending", "running", "complete", "failed"}


@pytest.mark.asyncio
async def test_resume_when_not_paused_returns_409(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("data.csv", b"a\n1\n", "text/csv")}
        r = await client.post("/sessions", files=files, data={"user_query": "q"})
        sid = r.json()["session_id"]
        # Wait briefly for the trivial graph to complete
        await asyncio.sleep(0.5)
        r2 = await client.post(
            f"/sessions/{sid}/resume", json={"target_column": "x"}
        )
        assert r2.status_code == 409


@pytest.mark.asyncio
async def test_unknown_session_returns_404(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/sessions/does-not-exist")
        assert r.status_code == 404
