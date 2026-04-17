"""
Asyncio coroutine that drives a session's pipeline run end-to-end.

Responsibilities:
  - Stream LangGraph events into the session's event queue + buffer
  - Translate node lifecycle into structured events (node_started/completed)
  - Capture interrupts (target selection) and pause the session
  - Resume the graph with Command(resume=<value>) when the user selects
  - Attach a QueueLogHandler to the autods logger so structured logs flow too
  - Emit pipeline_complete or pipeline_failed at the end
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from langgraph.types import Command

from core.web.events import EventTypes
from core.web.log_sink import QueueLogHandler
from core.web.session import Session


_AUTODS_LOG = logging.getLogger("autods")


class PipelineRunner:
    """One-shot runner for a single session."""

    def __init__(self, session: Session, graph_app: Any):
        self.session = session
        self.graph_app = graph_app
        self._loop = asyncio.get_event_loop()
        self._node_started_at: dict[str, float] = {}

    async def run(self, initial_state: dict) -> None:
        """Drive the graph. Catches all exceptions and surfaces them as events."""
        # Set web mode for the target_selector
        prior_mode = os.environ.get("AUTODS_INTERACTIVE_MODE")
        os.environ["AUTODS_INTERACTIVE_MODE"] = "web"
        sink = QueueLogHandler(
            self.session.queue, self.session.seq_counter, self._loop,
            record_to=self.session.record_event,
        )
        prior_level = _AUTODS_LOG.level
        # Ensure DEBUG records reach our sink regardless of prior setup.
        if _AUTODS_LOG.level == logging.NOTSET or _AUTODS_LOG.level > logging.DEBUG:
            _AUTODS_LOG.setLevel(logging.DEBUG)
        _AUTODS_LOG.addHandler(sink)

        try:
            await self._emit({
                "type": EventTypes.SESSION_STARTED,
                "session_id": self.session.id,
                "user_query": self.session.user_query,
                "dataset": {"filename": self.session.dataset_filename},
            })
            self.session.status = "running"

            # recursion_limit caps how many super-steps LangGraph will run.
            # Worst case here is ~234: MAX_PASSES (5) × (1 investigator +
            # MAX_TOOL_CALLS (20) tool loops + ~5 other nodes) + profiler +
            # target_selector + fe + autogluon + answer. 500 gives headroom
            # without hiding a real infinite loop. The CLI (app.invoke) uses
            # the default 25 because it happens not to exercise the worst
            # case, but the web path should be robust.
            config = {
                "configurable": {"thread_id": self.session.id},
                "recursion_limit": 500,
            }
            await self._stream_until_done(initial_state, config)

            # Capture artifacts from final state
            final = await self.graph_app.aget_state(config)
            artifacts = self._build_artifacts(final.values)
            self.session.artifacts = artifacts

            # The answer_agent stores its result in state (answer/final_answer)
            # rather than streaming tokens through astream_events, so no
            # final_answer_token events were emitted during the run. Emit one
            # now with the full text so the frontend's accumulator can render
            # it as part of the pipeline response card.
            if artifacts.get("final_answer"):
                await self._emit({
                    "type": EventTypes.FINAL_ANSWER_TOKEN,
                    "text": artifacts["final_answer"],
                })

            self.session.status = "complete"
            await self._emit({
                "type": EventTypes.PIPELINE_COMPLETE,
                "summary": self._summary(final.values),
            })
        except asyncio.CancelledError:
            self.session.status = "cancelled"
            raise
        except Exception as e:
            self.session.status = "failed"
            self.session.error = repr(e)
            await self._emit({
                "type": EventTypes.PIPELINE_FAILED,
                "error": repr(e),
            })
        finally:
            _AUTODS_LOG.removeHandler(sink)
            _AUTODS_LOG.setLevel(prior_level)
            if prior_mode is None:
                os.environ.pop("AUTODS_INTERACTIVE_MODE", None)
            else:
                os.environ["AUTODS_INTERACTIVE_MODE"] = prior_mode

    # ------------------------------------------------------------------
    # Streaming + interrupt handling
    # ------------------------------------------------------------------
    async def _stream_until_done(self, payload: Any, config: dict) -> None:
        """
        Stream events from the graph; if it pauses on an interrupt, wait for
        the user's resume value, then continue with Command(resume=...).
        Repeats until the graph reaches END.
        """
        current = payload
        while True:
            interrupted = await self._stream_once(current, config)
            if not interrupted:
                return
            # Wait for user selection
            self.session.resume_future = self._loop.create_future()
            self.session.status = "paused"
            self.session.paused_for = "target_selection"
            target = await self.session.resume_future
            await self._emit({
                "type": EventTypes.TARGET_SELECTION_RESOLVED,
                "target_column": target,
            })
            self.session.status = "running"
            self.session.paused_for = None
            current = Command(resume=target)

    async def _stream_once(self, payload: Any, config: dict) -> bool:
        """
        Stream events until the graph ends or pauses. Returns True if it
        paused on an interrupt (caller should wait for resume); False if it
        ran to completion.
        """
        async for event in self.graph_app.astream_events(
            payload, config, version="v2"
        ):
            await self._handle_graph_event(event)

        # After the stream ends, check if we're paused (interrupt) or done.
        state = await self.graph_app.aget_state(config)
        if state.tasks:
            for t in state.tasks:
                if getattr(t, "interrupts", None):
                    pause_payload = t.interrupts[0].value
                    self.session.pause_payload = pause_payload
                    await self._emit({
                        "type": EventTypes.TARGET_SELECTION_REQUIRED,
                        **pause_payload,
                    })
                    return True
        return False

    async def _handle_graph_event(self, event: dict) -> None:
        """Translate one LangGraph event into our event schema (if relevant)."""
        kind = event.get("event")
        name = event.get("name")
        if kind == "on_chain_start" and self._is_node(event):
            self._node_started_at[name] = time.monotonic()
            await self._emit({
                "type": EventTypes.NODE_STARTED,
                "node": name,
            })
        elif kind == "on_chain_end" and self._is_node(event):
            started = self._node_started_at.pop(name, None)
            duration_ms = (
                int((time.monotonic() - started) * 1000) if started else None
            )
            await self._emit({
                "type": EventTypes.NODE_COMPLETED,
                "node": name,
                "duration_ms": duration_ms,
            })

    @staticmethod
    def _is_node(event: dict) -> bool:
        # LangGraph emits chain events for many internal constructs; we only
        # want our top-level graph nodes. Most reliable signal on LangGraph v2
        # is metadata.langgraph_node — it's set when the chain represents a
        # registered node and matches the event name.
        meta = event.get("metadata", {}) or {}
        node = meta.get("langgraph_node")
        if node and node == event.get("name"):
            return True
        tags = event.get("tags", []) or []
        return any(t.startswith("graph:step:") for t in tags)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _emit(self, fields: dict) -> None:
        """Build, buffer, and enqueue an event through the shared session."""
        ev_type = fields.pop("type")
        await self.session.emit(ev_type, **fields)

    def _build_artifacts(self, state: dict) -> dict:
        """Extract the bits the Q&A agent needs."""
        findings = state.get("investigation_findings")
        return {
            "user_query": state.get("user_query"),
            "target_column": state.get("target_column"),
            "investigation_findings": (
                findings.model_dump()
                if findings and hasattr(findings, "model_dump")
                else findings
            ),
            "applied_steps": [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in state.get("applied_steps", [])
            ],
            "applied_fe_steps": [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in state.get("applied_fe_steps", [])
            ],
            # Pipeline activity log — gives the Q&A agent insight into the
            # moment-to-moment decisions (what tools it called, how the
            # router branched, free-text log messages).
            "activity_log": self._build_activity_log(),
            "model_metadata": state.get("model_metadata"),
            "final_answer": state.get("final_answer") or state.get("answer"),
        }

    def _summary(self, state: dict) -> dict:
        return {
            "pass_count": state.get("pass_count"),
            "target_column": state.get("target_column"),
            "rows_cleaned": (
                len(state["clean_df"])
                if state.get("clean_df") is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Activity log projection
    # ------------------------------------------------------------------
    # Event types worth showing to the Q&A agent. We skip heartbeats,
    # duplicate lifecycle chatter, and node lifecycle (the artifacts already
    # summarize what each node did structurally).
    _QA_LOG_TYPES = frozenset({
        EventTypes.ROUTE_DECISION,
        EventTypes.TOOL_CALL,
        EventTypes.INVESTIGATION_FINDINGS,
        EventTypes.CLEANING_RECIPE,
        EventTypes.PROFILE_SUMMARY,
        EventTypes.MODEL_METADATA,
        EventTypes.LOG,
        EventTypes.NODE_STARTED,
        EventTypes.NODE_COMPLETED,
        EventTypes.TARGET_SELECTION_RESOLVED,
    })

    # Cap on how many activity-log entries we send to the Q&A agent.
    # A 5-pass run with a full investigator tool loop can produce ~500
    # interesting events; context-window budget says roughly 300 is a
    # safe upper bound once every entry is trimmed. If we exceed this we
    # keep the most recent ones — the agent can always scroll the UI for
    # the full stream.
    _QA_LOG_MAX_ENTRIES = 300

    def _build_activity_log(self) -> list[dict]:
        """
        Compact, serializable projection of the session's event stream for
        the Q&A agent. Trims heavy payloads so the LLM context doesn't
        balloon on long runs.
        """
        out: list[dict] = []
        for ev in self.session.event_buffer:
            t = ev.get("type")
            if t not in self._QA_LOG_TYPES:
                continue
            entry = {"seq": ev.get("seq"), "type": t}
            # Carry type-specific payload fields
            if t == EventTypes.ROUTE_DECISION:
                entry["router"] = ev.get("router")
                entry["decision"] = ev.get("decision")
                if ev.get("context"):
                    entry["context"] = ev["context"]
            elif t == EventTypes.TOOL_CALL:
                entry["tool"] = ev.get("tool")
                entry["params"] = ev.get("params")
                preview = ev.get("result_preview") or ""
                entry["result_preview"] = preview[:300]
            elif t in (
                EventTypes.INVESTIGATION_FINDINGS,
                EventTypes.CLEANING_RECIPE,
                EventTypes.PROFILE_SUMMARY,
                EventTypes.MODEL_METADATA,
            ):
                raw = str(ev.get("raw", ""))
                entry["raw"] = raw[:1500]
            elif t == EventTypes.LOG:
                entry["level"] = ev.get("level")
                entry["node"] = ev.get("node")
                entry["message"] = ev.get("message")
                if ev.get("context"):
                    entry["context"] = ev["context"]
            elif t in (EventTypes.NODE_STARTED, EventTypes.NODE_COMPLETED):
                entry["node"] = ev.get("node")
                if ev.get("duration_ms") is not None:
                    entry["duration_ms"] = ev["duration_ms"]
            elif t == EventTypes.TARGET_SELECTION_RESOLVED:
                entry["target_column"] = ev.get("target_column")
            out.append(entry)
        # Keep only the most recent N entries; earlier detail is
        # represented structurally in applied_steps / findings / recipe.
        if len(out) > self._QA_LOG_MAX_ENTRIES:
            out = out[-self._QA_LOG_MAX_ENTRIES:]
        return out
