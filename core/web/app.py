"""
FastAPI app for the AutoDS web demo.

create_app(graph_app) returns a configured FastAPI instance. The default is
the real compiled LangGraph from core.pipeline.graph; tests inject a stub.
"""
from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from core.web.events import EventTypes
from core.web.runner import PipelineRunner
from core.web.session import SessionManager


def create_app(graph_app: Optional[Any] = None) -> FastAPI:
    """Build the FastAPI app. Inject graph_app for tests; defaults to real graph."""
    if graph_app is None:
        # Lazy import — keeps tests light and avoids importing AutoGluon paths.
        from core.pipeline.graph import app as real_app
        graph_app = real_app

    app = FastAPI(title="AutoDS Web Demo")
    sessions = SessionManager()

    # Dev CORS for Vite dev server on :5173
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/sessions")
    async def create_session(
        file: UploadFile = File(...),
        user_query: str = Form(...),
    ):
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400, detail="Only .csv files accepted"
            )
        csv_bytes = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"CSV parse error: {e}"
            )

        session = sessions.create(csv_bytes, file.filename, user_query)
        runner = PipelineRunner(session=session, graph_app=graph_app)
        initial_state = {
            "user_query": user_query,
            "working_df": df,
            "retry_count": 0,
            "tool_call_count": 0,
            "pass_count": 0,
            "target_column": None,
        }
        session.task = asyncio.create_task(runner.run(initial_state))
        return {
            "session_id": session.id,
            "filename": file.filename,
            "rows": len(df),
            "cols": len(df.columns),
        }

    @app.get("/sessions/{sid}")
    async def get_session(sid: str):
        try:
            s = sessions.get(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session")
        return {
            "session_id": s.id,
            "status": s.status,
            "paused_for": s.paused_for,
            "pause_payload": s.pause_payload,
            "user_query": s.user_query,
            "dataset_filename": s.dataset_filename,
            "error": s.error,
            "has_artifacts": s.artifacts is not None,
        }

    @app.delete("/sessions/{sid}")
    async def delete_session(sid: str):
        try:
            sessions.delete(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session")
        return {"ok": True}

    @app.post("/sessions/{sid}/resume")
    async def resume_session(sid: str, body: dict):
        try:
            s = sessions.get(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session")
        if s.status != "paused":
            raise HTTPException(
                status_code=409,
                detail=f"Session not paused (status={s.status})",
            )
        target = body.get("target_column")
        if not target:
            raise HTTPException(
                status_code=400, detail="target_column required"
            )
        try:
            sessions.resume(sid, target_column=target)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"ok": True}

    @app.get("/sessions/{sid}/events")
    async def stream_events(sid: str, request: Request):
        try:
            s = sessions.get(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session")

        last_event_id = request.headers.get("Last-Event-ID")
        last_seq = (
            int(last_event_id)
            if last_event_id and last_event_id.isdigit()
            else 0
        )

        async def event_generator():
            # Replay buffered events past last_seq first
            replay = s.replay_since(last_seq)
            if replay == "truncated":
                yield {
                    "event": "message",
                    "id": "0",
                    "data": json.dumps(
                        {"type": EventTypes.REPLAY_TRUNCATED}
                    ),
                }
            else:
                for ev in replay:
                    yield {
                        "event": "message",
                        "id": str(ev.get("seq", 0)),
                        "data": json.dumps(ev, default=str),
                    }
            # Then stream live
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(s.queue.get(), timeout=15.0)
                    yield {
                        "event": "message",
                        "id": str(ev.get("seq", 0)),
                        "data": json.dumps(ev, default=str),
                    }
                    if ev.get("type") in {
                        EventTypes.PIPELINE_COMPLETE,
                        EventTypes.PIPELINE_FAILED,
                    }:
                        return
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield {"event": "ping", "data": ""}

        return EventSourceResponse(event_generator())

    @app.post("/sessions/{sid}/ask")
    async def ask_followup(sid: str, body: dict, request: Request):
        from core.web.qa import stream_qa_answer  # lazy import
        try:
            s = sessions.get(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session")
        if s.artifacts is None:
            raise HTTPException(
                status_code=409, detail="Pipeline not yet complete"
            )
        question = body.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="question required")

        async def event_generator():
            async for chunk in stream_qa_answer(s, question):
                if await request.is_disconnected():
                    return
                yield {
                    "event": "message",
                    "data": json.dumps(chunk, default=str),
                }

        return EventSourceResponse(event_generator())

    return app
