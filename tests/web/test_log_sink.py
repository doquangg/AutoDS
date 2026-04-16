import asyncio
import logging

import pytest

from core.web.events import EventTypes, SeqCounter
from core.web.log_sink import QueueLogHandler


@pytest.fixture
def queue_and_handler():
    queue: asyncio.Queue = asyncio.Queue()
    seq = SeqCounter()
    loop = asyncio.new_event_loop()
    handler = QueueLogHandler(queue=queue, seq=seq, loop=loop)
    autods = logging.getLogger("autods")
    autods.setLevel(logging.DEBUG)
    autods.addHandler(handler)
    yield queue, autods, loop
    autods.removeHandler(handler)
    loop.close()


def _drain(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> list:
    out = []
    async def drain():
        while not queue.empty():
            out.append(await queue.get())
    loop.run_until_complete(drain())
    return out


def test_recognizes_route_log(queue_and_handler):
    queue, logger, loop = queue_and_handler
    logger.debug("ROUTE route_sandbox \u2192 evaluate (pass_count=0)")
    events = _drain(queue, loop)
    assert any(e["type"] == EventTypes.ROUTE_DECISION for e in events)
    ev = next(e for e in events if e["type"] == EventTypes.ROUTE_DECISION)
    assert ev["router"] == "route_sandbox"
    assert ev["decision"] == "evaluate"


def test_recognizes_tool_call_log(queue_and_handler):
    queue, logger, loop = queue_and_handler
    logger.debug("TOOL_CALL value_distribution\n  params: {\"col\": \"age\"}\n  result: {...}")
    events = _drain(queue, loop)
    ev = next(e for e in events if e["type"] == EventTypes.TOOL_CALL)
    assert ev["tool"] == "value_distribution"
    assert "params" in ev


def test_recognizes_node_event_log(queue_and_handler):
    queue, logger, loop = queue_and_handler
    logger.debug("NODE sandbox | pass complete | pass_number=1 violations_found=3 steps_executed=2 rows_after=842")
    events = _drain(queue, loop)
    ev = next(e for e in events if e["type"] == EventTypes.LOG)
    assert ev["node"] == "sandbox"
    assert "pass complete" in ev["message"]


def test_unrecognized_log_falls_back_to_log_event(queue_and_handler):
    queue, logger, loop = queue_and_handler
    logger.debug("Some unstructured message that doesn't match a prefix")
    events = _drain(queue, loop)
    ev = next(e for e in events if e["type"] == EventTypes.LOG)
    assert ev["message"] == "Some unstructured message that doesn't match a prefix"
    assert ev["level"] == "DEBUG"


def test_warning_level_passes_through(queue_and_handler):
    queue, logger, loop = queue_and_handler
    logger.warning("MODEL_TRAINING_FAILED error=oom")
    events = _drain(queue, loop)
    ev = next(e for e in events if e["type"] == EventTypes.LOG)
    assert ev["level"] == "WARNING"
