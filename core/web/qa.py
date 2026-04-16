"""
Adapter from the QA agent (yields strings) to web events
(QA_TOKEN per chunk, QA_COMPLETE at end, QA_ERROR on failure). Also
updates the session's qa_messages history.
"""
from __future__ import annotations

import sys
import traceback
from typing import AsyncIterator

from core.agents.qa_agent import run_qa_agent
from core.web.events import EventTypes, SeqCounter, build_event
from core.web.session import Session


async def stream_qa_answer(
    session: Session, question: str
) -> AsyncIterator[dict]:
    """
    Yield event dicts (QA_TOKEN stream, QA_COMPLETE on success, or
    QA_ERROR on any failure). This function never raises — failures are
    surfaced to the client as an event and logged to stderr so the
    problem is visible in both places.
    """
    seq = SeqCounter()
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
            yield build_event(EventTypes.QA_TOKEN, seq, text=token)
    except Exception as e:  # noqa: BLE001 — broad by design; see note above
        traceback.print_exc(file=sys.stderr)
        print(f"[qa] stream failed: {e!r}", flush=True, file=sys.stderr)
        yield build_event(
            EventTypes.QA_ERROR,
            seq,
            error=f"{type(e).__name__}: {e}",
        )
        # Persist the user question + a failure marker so the history
        # doesn't end up half-written on the next turn.
        session.qa_messages.append({"role": "user", "content": question})
        session.qa_messages.append(
            {
                "role": "assistant",
                "content": f"[error: {type(e).__name__}: {e}]",
            }
        )
        return

    answer_text = "".join(full_answer)
    print(f"[qa] done tokens={len(full_answer)} chars={len(answer_text)}",
          flush=True, file=sys.stderr)

    if not answer_text.strip():
        # LLM returned nothing — treat as an error so the UI can show
        # something actionable instead of silently hanging.
        yield build_event(
            EventTypes.QA_ERROR,
            seq,
            error=(
                "The Q&A agent returned an empty response. This usually "
                "means the model configuration is wrong (check QA_MODEL / "
                "ANSWER_MODEL / OPENAI_API_KEY / OPENAI_API_BASE)."
            ),
        )
        return

    session.qa_messages.append({"role": "user", "content": question})
    session.qa_messages.append({"role": "assistant", "content": answer_text})
    yield build_event(EventTypes.QA_COMPLETE, seq, text=answer_text)
