"""
FastAPI app: serve the AutoDS UI and API.

Run from repo root:
  uvicorn web.main:app --reload --host 127.0.0.1 --port 8000

Requires OPENAI_API_KEY in the environment (same as scripts/run_graph.py).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web.service import (
    SAMPLE_CSV,
    ndjson_dumps,
    prepare_interactive,
    run_pipeline,
    run_pipeline_stream,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AutoDS", description="Web UI for the AutoDS ML pipeline")

# Allow the UI to call the API from another dev port (Live Preview, Vite, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _form_bool(raw: str) -> bool:
    return str(raw).lower().strip() in ("true", "1", "on", "yes")


async def _load_dataframe(
    use_sample: str = Form("false"),
    file: UploadFile | None = File(None),
) -> pd.DataFrame:
    if _form_bool(use_sample):
        if not SAMPLE_CSV.is_file():
            raise HTTPException(404, "Sample CSV not found on server")
        return pd.read_csv(SAMPLE_CSV)
    if file is None or not file.filename:
        raise HTTPException(400, "Upload a CSV file or enable 'Use sample data'")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 25 MB)")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}") from e


API_BASE_MARKER = "__AUTODS_API_BASE__"


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    """Serve HTML with API base injected so fetch() always hits this server."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(500, "Missing web/static/index.html")
    html = index_path.read_text(encoding="utf-8")
    base = str(request.base_url).rstrip("/")
    if API_BASE_MARKER not in html:
        raise HTTPException(500, "index.html missing API base placeholder")
    html = html.replace(API_BASE_MARKER, base)
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


@app.get("/favicon.ico")
async def favicon() -> Response:
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        b'<rect width="32" height="32" rx="6" fill="#3d8fd1"/>'
        b'<text x="16" y="21" text-anchor="middle" fill="white" '
        b'font-family="system-ui,sans-serif" font-size="14" font-weight="700">A</text>'
        b"</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/sample-columns")
async def sample_columns() -> dict:
    if not SAMPLE_CSV.is_file():
        raise HTTPException(404, "Sample data not found")
    df = pd.read_csv(SAMPLE_CSV, nrows=0)
    return {"columns": list(df.columns.astype(str))}


@app.post("/api/prepare")
async def api_prepare(
    user_query: str = Form(...),
    use_sample: str = Form("false"),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    df = await _load_dataframe(use_sample, file)
    if df.empty:
        raise HTTPException(400, "Dataset is empty")
    user_query = user_query.strip()
    if not user_query:
        raise HTTPException(400, "Question / query is required")
    try:
        payload = prepare_interactive(user_query, df)
    except Exception as e:
        raise HTTPException(500, f"Prepare failed: {type(e).__name__}: {e}") from e
    return JSONResponse(content=payload)


@app.post("/api/run")
async def api_run(
    user_query: str = Form(...),
    target_column: str = Form(...),
    use_sample: str = Form("false"),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    df = await _load_dataframe(use_sample, file)
    if df.empty:
        raise HTTPException(400, "Dataset is empty")

    target_column = target_column.strip()
    if not target_column:
        raise HTTPException(400, "Target column is required")
    if target_column not in df.columns:
        raise HTTPException(
            400,
            f"Target column {target_column!r} not in data. Columns: {list(df.columns.astype(str))}",
        )

    user_query = user_query.strip()
    if not user_query:
        raise HTTPException(400, "Question / query is required")

    payload = run_pipeline(user_query, df, target_column)
    status = 200 if payload.get("ok") else 500
    return JSONResponse(content=payload, status_code=status)


@app.post("/api/run/stream")
async def api_run_stream(
    user_query: str = Form(...),
    target_column: str = Form(...),
    use_sample: str = Form("false"),
    file: UploadFile | None = File(None),
) -> StreamingResponse:
    df = await _load_dataframe(use_sample, file)
    if df.empty:
        raise HTTPException(400, "Dataset is empty")

    target_column = target_column.strip()
    if not target_column:
        raise HTTPException(400, "Target column is required")
    if target_column not in df.columns:
        raise HTTPException(
            400,
            f"Target column {target_column!r} not in data. Columns: {list(df.columns.astype(str))}",
        )

    user_query = user_query.strip()
    if not user_query:
        raise HTTPException(400, "Question / query is required")

    uq, tc = user_query, target_column

    def byte_stream():
        for event in run_pipeline_stream(uq, df, tc):
            yield ndjson_dumps(event)

    return StreamingResponse(
        byte_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Mount static assets after API routes so /api/* is never shadowed.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
