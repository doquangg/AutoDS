"""
Q&A agent that answers follow-up questions about a completed pipeline run,
grounded only in the saved artifacts. Streams tokens.
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.prompts import QA_SYSTEM_PROMPT


def get_qa_llm() -> ChatOpenAI:
    """ChatOpenAI for Q&A. Default model: same as ANSWER_MODEL."""
    return ChatOpenAI(
        model=os.environ.get(
            "QA_MODEL", os.environ.get("ANSWER_MODEL", "gpt-5.4-mini")
        ),
        temperature=0.0,
        streaming=True,
        base_url=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def build_qa_messages(
    artifacts: dict,
    history: list[tuple[str, str]],
    question: str,
) -> list[BaseMessage]:
    """Assemble system + user messages with artifacts as context."""
    artifacts_block = json.dumps(artifacts, indent=2, default=str)
    history_block = "\n\n".join(
        f"{role.upper()}: {text}" for role, text in history
    )
    user_content = (
        f"PIPELINE ARTIFACTS:\n{artifacts_block}\n\n"
        f"CONVERSATION SO FAR:\n{history_block or '(none)'}\n\n"
        f"NEW QUESTION:\n{question}"
    )
    return [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


async def run_qa_agent(
    artifacts: dict,
    history: list[tuple[str, str]],
    question: str,
) -> AsyncIterator[str]:
    """Stream the answer as text chunks."""
    llm = get_qa_llm()
    messages = build_qa_messages(artifacts, history, question)
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
