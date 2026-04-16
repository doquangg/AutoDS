"""
Web event schema for the AutoDS demo.

Single source of truth for event type constants and frame construction.
Every SSE frame is a JSON object with a `type` discriminator, a monotonic
`seq` (per session), and an ISO-8601 `ts`.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


class EventTypes:
    SESSION_STARTED = "session_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    ROUTE_DECISION = "route_decision"
    TOOL_CALL = "tool_call"
    INVESTIGATION_FINDINGS = "investigation_findings"
    CLEANING_RECIPE = "cleaning_recipe"
    PROFILE_SUMMARY = "profile_summary"
    MODEL_METADATA = "model_metadata"
    TARGET_SELECTION_REQUIRED = "target_selection_required"
    TARGET_SELECTION_RESOLVED = "target_selection_resolved"
    FINAL_ANSWER_TOKEN = "final_answer_token"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"
    QA_TOKEN = "qa_token"
    QA_COMPLETE = "qa_complete"
    HEARTBEAT = "heartbeat"
    REPLAY_TRUNCATED = "replay_truncated"
    LOG = "log"  # fallback for unrecognized log records


class SeqCounter:
    """Thread-safe monotonic counter, scoped per session."""
    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n


def build_event(event_type: str, seq: SeqCounter, **fields: Any) -> dict:
    """Build a fully-formed event dict ready to serialize to SSE JSON."""
    return {
        "type": event_type,
        "seq": seq.next(),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **fields,
    }
