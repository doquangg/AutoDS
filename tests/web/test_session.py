"""Tests for the in-memory session manager."""
import asyncio

import pytest

from core.web.session import Session, SessionManager


@pytest.mark.asyncio
async def test_create_returns_session_with_id_and_status():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a,b\n1,2\n", filename="x.csv", user_query="q?")
    assert s.id
    assert s.status == "pending"
    assert s.user_query == "q?"
    assert s.dataset_filename == "x.csv"


@pytest.mark.asyncio
async def test_get_returns_session_and_raises_on_missing():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q")
    assert mgr.get(s.id) is s
    with pytest.raises(KeyError):
        mgr.get("nonexistent")


@pytest.mark.asyncio
async def test_record_event_appends_to_ring_buffer():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q")
    s.record_event({"type": "node_started", "seq": 1})
    s.record_event({"type": "node_completed", "seq": 2})
    assert len(s.event_buffer) == 2
    assert s.event_buffer[-1]["seq"] == 2


@pytest.mark.asyncio
async def test_ring_buffer_caps_at_max_size():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q",
                   buffer_size=10)
    for i in range(15):
        s.record_event({"type": "log", "seq": i + 1})
    assert len(s.event_buffer) == 10
    assert s.event_buffer[0]["seq"] == 6   # oldest kept
    assert s.event_buffer[-1]["seq"] == 15


@pytest.mark.asyncio
async def test_replay_since_returns_events_after_seq():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q")
    for i in range(5):
        s.record_event({"type": "log", "seq": i + 1})
    replayed = s.replay_since(2)
    assert [e["seq"] for e in replayed] == [3, 4, 5]


@pytest.mark.asyncio
async def test_replay_since_returns_truncation_signal_when_too_old():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q",
                   buffer_size=3)
    for i in range(5):
        s.record_event({"type": "log", "seq": i + 1})
    # Buffer holds seq 3,4,5. Asking for >1 means we'd miss seq 2.
    out = s.replay_since(1)
    assert out == "truncated"


@pytest.mark.asyncio
async def test_resume_with_target_sets_future():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q")
    s.status = "paused"
    s.paused_for = "target_selection"
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    s.resume_future = fut
    mgr.resume(s.id, target_column="churned")
    assert fut.done()
    assert fut.result() == "churned"


@pytest.mark.asyncio
async def test_resume_when_not_paused_raises():
    mgr = SessionManager()
    s = mgr.create(csv_bytes=b"a\n1\n", filename="x.csv", user_query="q")
    with pytest.raises(RuntimeError):
        mgr.resume(s.id, target_column="x")
