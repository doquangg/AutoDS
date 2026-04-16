"""
Adapter from the QA agent (yields strings) to web events
(QA_TOKEN per chunk, QA_COMPLETE at end). Also updates the session's
qa_messages history.
"""
from __future__ import annotations

from typing import AsyncIterator

from core.agents.qa_agent import run_qa_agent
from core.web.events import EventTypes, SeqCounter, build_event
from core.web.session import Session


async def stream_qa_answer(
    session: Session, question: str
) -> AsyncIterator[dict]:
    """Yield event dicts (QA_TOKEN, then QA_COMPLETE)."""
    seq = SeqCounter()
    history = (
        [(m["role"], m["content"]) for m in session.qa_messages]
        if session.qa_messages
        else []
    )

    full_answer = []
    async for token in run_qa_agent(session.artifacts, history, question):
        full_answer.append(token)
        yield build_event(EventTypes.QA_TOKEN, seq, text=token)

    answer_text = "".join(full_answer)
    # Persist the turn in the session history (for follow-up follow-ups)
    session.qa_messages.append({"role": "user", "content": question})
    session.qa_messages.append({"role": "assistant", "content": answer_text})
    yield build_event(EventTypes.QA_COMPLETE, seq, text=answer_text)
