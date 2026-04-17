"""
Q&A runner: executes the QA agent and pushes its output onto the session's
shared event stream (queue + ring buffer) using the session's seq counter.

Running the QA agent this way — rather than streaming the response
directly from the POST /ask handler — means all client-visible events for
a session flow through a single SSE connection (the GET /events stream),
which is more reliable than trying to stream a POST response and also
means QA events replay on SSE reconnect just like pipeline events do.
"""
from __future__ import annotations

import sys
import traceback

from core.agents.qa_agent import run_qa_agent
from core.web.events import EventTypes
from core.web.session import Session


async def run_qa_task(session: Session, question: str) -> None:
    """
    Drive the QA agent. Emits QA_TOKEN per chunk, QA_COMPLETE on success,
    or QA_ERROR on any failure. Never raises — errors are both logged to
    stderr and surfaced to the UI as events.
    """
    history = (
        [(m["role"], m["content"]) for m in session.qa_messages]
        if session.qa_messages
        else []
    )
    print(
        f"[qa] question={question!r} history_turns={len(history)} "
        f"artifacts_keys={list((session.artifacts or {}).keys())}",
        flush=True,
        file=sys.stderr,
    )

    full_answer: list[str] = []
    try:
        async for token in run_qa_agent(session.artifacts, history, question):
            full_answer.append(token)
            await session.emit(EventTypes.QA_TOKEN, text=token)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(f"[qa] stream failed: {e!r}", flush=True, file=sys.stderr)
        await session.emit(
            EventTypes.QA_ERROR,
            error=f"{type(e).__name__}: {e}",
        )
        session.qa_messages.append({"role": "user", "content": question})
        session.qa_messages.append(
            {
                "role": "assistant",
                "content": f"[error: {type(e).__name__}: {e}]",
            }
        )
        return

    answer_text = "".join(full_answer)
    print(
        f"[qa] done tokens={len(full_answer)} chars={len(answer_text)}",
        flush=True,
        file=sys.stderr,
    )

    if not answer_text.strip():
        await session.emit(
            EventTypes.QA_ERROR,
            error=(
                "The Q&A agent returned an empty response. This usually "
                "means the model configuration is wrong (check QA_MODEL / "
                "ANSWER_MODEL / OPENAI_API_KEY / OPENAI_API_BASE)."
            ),
        )
        return

    session.qa_messages.append({"role": "user", "content": question})
    session.qa_messages.append({"role": "assistant", "content": answer_text})
    await session.emit(EventTypes.QA_COMPLETE, text=answer_text)
