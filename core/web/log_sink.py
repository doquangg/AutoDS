"""
A logging.Handler that turns AutoDS structured log records into web events.

The existing core/logger.py emits records with stable prefixes:
  - "ROUTE <router> -> <decision> (<context>)"
  - "TOOL_CALL <name>\n  params: ...\n  result: ..."
  - "NODE <name> | <message> | k=v ..."
  - "INVESTIGATION_FINDINGS\n  ...lines..."
  - "CLEANING_RECIPE (N steps):\n  ...lines..."
  - "PROFILE_SUMMARY rows=N columns=M\n  ..."
  - "MODEL_METADATA\n  ..."

This handler parses those prefixes and produces structured events. Anything
that doesn't match a known shape becomes a generic LOG event.

Why parsing rather than editing core/logger.py: the logger is a write-side
contract used by both the web layer AND the existing CLI. Parsing keeps the
sink a passive consumer with zero risk to the existing CLI behavior.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from core.web.events import EventTypes, SeqCounter, build_event


_ROUTE_RE = re.compile(r"^ROUTE\s+(\S+)\s*[\u2192>]+\s*(\S+)\s*(?:\((.*)\))?$")
_NODE_PIPE_RE = re.compile(r"^NODE\s+(\S+)\s*\|\s*(.+?)(?:\s*\|\s*(.*))?$")
_TOOL_CALL_RE = re.compile(r"^TOOL_CALL\s+(\S+)")


class QueueLogHandler(logging.Handler):
    """
    Routes autods logger records into a per-session asyncio queue as events.

    Args:
        queue:    target asyncio.Queue for events
        seq:      monotonic seq counter for this session
        loop:     event loop on which the queue lives (records may arrive
                  from worker threads, so we use call_soon_threadsafe)
        record_to: optional callable invoked with the event dict before it is
                  placed on the queue. The runner passes session.record_event
                  here so events also land in the replay ring buffer.
    """

    def __init__(self, queue: asyncio.Queue, seq: SeqCounter,
                 loop: asyncio.AbstractEventLoop,
                 record_to=None):
        super().__init__()
        self.setLevel(logging.DEBUG)
        self._queue = queue
        self._seq = seq
        self._loop = loop
        self._record_to = record_to

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = self._translate(record)
        except Exception:
            # Never let logging crash the pipeline.
            event = build_event(
                EventTypes.LOG, self._seq,
                level=record.levelname,
                message=f"[log_sink translation failed] {record.getMessage()}",
            )
        if self._record_to is not None:
            try:
                self._record_to(event)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def _translate(self, record: logging.LogRecord) -> dict:
        msg = record.getMessage()

        # ROUTE <router> -> <decision> (k=v, k=v)
        m = _ROUTE_RE.match(msg)
        if m:
            router, decision, ctx = m.group(1), m.group(2), m.group(3) or ""
            context = self._parse_kv_pairs(ctx)
            return build_event(
                EventTypes.ROUTE_DECISION, self._seq,
                router=router, decision=decision, context=context,
            )

        # TOOL_CALL <name>\n  params: <json>\n  result: <text>
        m = _TOOL_CALL_RE.match(msg)
        if m:
            tool = m.group(1)
            params = self._parse_labeled_field(msg, "params:")
            result_preview = self._parse_labeled_field(msg, "result:") or ""
            return build_event(
                EventTypes.TOOL_CALL, self._seq,
                tool=tool, params=params, result_preview=result_preview[:500],
            )

        # NODE <name> | <message> | <k=v ...>
        m = _NODE_PIPE_RE.match(msg)
        if m:
            node, message, ctx = m.group(1), m.group(2), m.group(3) or ""
            return build_event(
                EventTypes.LOG, self._seq,
                level=record.levelname,
                node=node,
                message=message,
                context=self._parse_kv_pairs(ctx),
            )

        # Multi-line structured payloads
        if msg.startswith("INVESTIGATION_FINDINGS"):
            return build_event(
                EventTypes.INVESTIGATION_FINDINGS, self._seq,
                raw=msg,
            )
        if msg.startswith("CLEANING_RECIPE"):
            return build_event(
                EventTypes.CLEANING_RECIPE, self._seq,
                raw=msg,
            )
        if msg.startswith("PROFILE_SUMMARY"):
            return build_event(
                EventTypes.PROFILE_SUMMARY, self._seq,
                raw=msg,
            )
        if msg.startswith("MODEL_METADATA"):
            return build_event(
                EventTypes.MODEL_METADATA, self._seq,
                raw=msg,
            )

        # Fallback: generic log event
        return build_event(
            EventTypes.LOG, self._seq,
            level=record.levelname,
            message=msg,
        )

    @staticmethod
    def _parse_kv_pairs(s: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not s:
            return out
        # Split on commas or whitespace; tolerate "k=v, k=v" or "k=v k=v".
        parts = re.split(r"[,\s]+", s.strip())
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _parse_labeled_field(msg: str, label: str) -> Any:
        """Pull a JSON-looking value following 'label:' on its own line."""
        for line in msg.splitlines():
            line = line.strip()
            if line.startswith(label):
                payload = line[len(label):].strip()
                try:
                    return json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    return payload
        return None
