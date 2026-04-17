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
            # Replay buffered events past last_seq first. The ring buffer is
            # the single source of truth for past events; the queue is the
            # live stream. After replay we drain any queue entries already
            # covered by the buffer so the client doesn't see duplicates.
            replay = s.replay_since(last_seq)
            if replay == "truncated":
                yield {
                    "event": "message",
                    "id": "0",
                    "data": json.dumps(
                        {"type": EventTypes.REPLAY_TRUNCATED}
                    ),
                }
                last_yielded = 0
            else:
                last_yielded = last_seq
                for ev in replay:
                    yield {
                        "event": "message",
                        "id": str(ev.get("seq", 0)),
                        "data": json.dumps(ev, default=str),
                    }
                    last_yielded = max(last_yielded, ev.get("seq", 0))

            # Live stream. sse-starlette handles keepalive pings via the
            # ping= parameter below.
            #
            # IMPORTANT: we do NOT close the stream on PIPELINE_COMPLETE or
            # PIPELINE_FAILED. After the pipeline finishes, the user may
            # ask follow-up questions via POST /ask, and the Q&A task
            # emits its events (QA_TOKEN / QA_COMPLETE / QA_ERROR) onto
            # this same queue. Closing here means those events have no
            # reader; EventSource auto-reconnect is unreliable in practice
            # (tab backgrounding, server close semantics, etc), and the
            # whole design is simpler if there's one long-lived connection
            # per session that only terminates when the client disconnects.
            while True:
                if await request.is_disconnected():
                    print(
                        f"[events] session={sid} client disconnected, "
                        f"closing stream (last_yielded={last_yielded})",
                        flush=True,
                    )
                    return
                ev = await s.queue.get()
                # Skip anything already delivered via replay.
                if ev.get("seq", 0) <= last_yielded:
                    continue
                print(
                    f"[events] session={sid} yield seq={ev.get('seq')} "
                    f"type={ev.get('type')}",
                    flush=True,
                )
                yield {
                    "event": "message",
                    "id": str(ev.get("seq", 0)),
                    "data": json.dumps(ev, default=str),
                }
                last_yielded = ev.get("seq", 0)

        return EventSourceResponse(event_generator(), ping=15)

    @app.post("/sessions/{sid}/ask")
    async def ask_followup(sid: str, body: dict):
        """
        Fire-and-forget: spawn the Q&A agent as a background task and
        return immediately. All output (QA_TOKEN / QA_COMPLETE / QA_ERROR)
        flows through the session's existing /events SSE stream, alongside
        pipeline events. This avoids the well-known flakiness of streaming
        a POST response body to the browser and keeps the frontend's event
        handling uniform.
        """
        from core.web.qa import run_qa_task  # lazy import
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

        # Don't start a second QA run while one is already in flight for
        # this session — emitted events would interleave.
        if s.qa_task is not None and not s.qa_task.done():
            raise HTTPException(
                status_code=409,
                detail="A Q&A request is already in progress for this session",
            )

        s.qa_task = asyncio.create_task(run_qa_task(s, question))
        return {"ok": True}

    return app
