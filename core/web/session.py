"""
In-memory session manager for the AutoDS web demo.

Each Session owns:
  - an asyncio.Queue for live event streaming
  - a deque (ring buffer) for SSE reconnect replay
  - a status / paused_for / resume_future for human-in-the-loop pauses
  - the run's artifacts after the pipeline completes (powering Q&A)

Single-process, no persistence. Sessions live until DELETE or process exit.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.messages import BaseMessage

from core.web.events import SeqCounter, build_event


SessionStatus = Literal[
    "pending", "running", "paused", "complete", "failed", "cancelled"
]
PauseReason = Literal["target_selection"]


@dataclass
class Session:
    id: str
    user_query: str
    dataset_filename: str
    csv_bytes: bytes
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    event_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    # Shared monotonic seq across everything the session emits — pipeline
    # runner, logger sink, and post-run Q&A all draw from this. Keeping a
    # single counter means QA events strictly follow pipeline events in
    # the stream, so the SSE reconnect dedup (`seq <= last_yielded`) works.
    seq_counter: SeqCounter = field(default_factory=SeqCounter)
    status: SessionStatus = "pending"
    paused_for: Optional[PauseReason] = None
    pause_payload: Optional[dict] = None
    resume_future: Optional[asyncio.Future] = None
    task: Optional[asyncio.Task] = None
    qa_task: Optional[asyncio.Task] = None
    artifacts: Optional[dict] = None
    qa_messages: list[BaseMessage] = field(default_factory=list)
    error: Optional[str] = None

    def record_event(self, event: dict) -> None:
        """Append to the ring buffer (used for SSE reconnect replay)."""
        self.event_buffer.append(event)

    async def emit(self, event_type: str, **fields: Any) -> dict:
        """
        Build an event with the session's shared seq counter, record it to
        the replay buffer, and publish it on the live queue. The single
        entry point used by the runner, the log sink, and the Q&A task so
        seq numbers stay monotonic and every event is replayable.
        """
        ev = build_event(event_type, self.seq_counter, **fields)
        self.record_event(ev)
        await self.queue.put(ev)
        # Diagnostic: mirror every session-level emit to stderr. Keeps QA
        # events (which don't go through the runner's own print) visible
        # in the same log stream.
        preview = {
            k: (
                v if not isinstance(v, str) or len(v) < 80
                else v[:80] + "…"
            )
            for k, v in ev.items()
            if k not in {"ts", "seq"}
        }
        print(
            f"[session {self.id[:8]}] emit seq={ev['seq']} qsize={self.queue.qsize()} {preview}",
            flush=True,
        )
        return ev

    def replay_since(self, last_seq: int):
        """
        Return events with seq > last_seq, in order.

        If last_seq is older than what the buffer still holds, return the
        sentinel string 'truncated' so the caller can emit replay_truncated.
        """
        if not self.event_buffer:
            return []
        oldest = self.event_buffer[0].get("seq", 0)
        if last_seq < oldest - 1:
            return "truncated"
        return [e for e in self.event_buffer if e.get("seq", 0) > last_seq]


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        csv_bytes: bytes,
        filename: str,
        user_query: str,
        *,
        buffer_size: int = 1000,
    ) -> Session:
        sid = uuid.uuid4().hex
        session = Session(
            id=sid,
            user_query=user_query,
            dataset_filename=filename,
            csv_bytes=csv_bytes,
        )
        # Override default buffer size if requested
        session.event_buffer = deque(maxlen=buffer_size)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]

    def resume(self, session_id: str, target_column: str) -> None:
        s = self.get(session_id)
        if s.status != "paused" or s.resume_future is None:
            raise RuntimeError(
                f"Session {session_id} is not paused (status={s.status})"
            )
        s.resume_future.set_result(target_column)

    def cancel(self, session_id: str) -> None:
        s = self.get(session_id)
        if s.task and not s.task.done():
            s.task.cancel()
        s.status = "cancelled"

    def delete(self, session_id: str) -> None:
        self.cancel(session_id)
        self._sessions.pop(session_id, None)
