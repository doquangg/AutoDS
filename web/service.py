"""
Run the LangGraph pipeline from the web layer: capture console output,
serialize results, profile + target suggestions for interactive UI, and
stream progress via NDJSON events.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_CSV = (
    REPO_ROOT
    / "data"
    / "sample_data"
    / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)


def _json_safe(obj: Any) -> Any:
    """Recursively convert objects for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if x != x else x
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, pd.DataFrame):
        return {
            "rows": int(len(obj)),
            "columns": list(obj.columns.astype(str)),
            "preview_records": obj.head(50).replace({float("nan"): None}).to_dict(orient="records"),
        }
    return str(obj)


def summarize_graph_state(state: dict) -> dict:
    """Compact, JSON-safe snapshot for streaming progress."""
    s: Dict[str, Any] = {}
    wf = state.get("working_df")
    if wf is not None and hasattr(wf, "shape"):
        s["rows"] = int(wf.shape[0])
        s["cols"] = int(wf.shape[1])
    s["pass_count"] = state.get("pass_count", 0)
    s["retry_count"] = state.get("retry_count", 0)
    s["target_column"] = state.get("target_column")
    prof = state.get("profile")
    if prof is not None:
        cols = getattr(prof, "columns", None)
        if cols is None and isinstance(prof, dict):
            cols = prof.get("columns")
        if cols is not None:
            s["profiled_columns"] = len(cols)
    inv = state.get("investigation_findings")
    if inv is not None:
        viol = getattr(inv, "violations", None)
        if viol is None and isinstance(inv, dict):
            viol = inv.get("violations") or []
        s["violations"] = len(viol) if viol else 0
    plan = state.get("current_plan")
    if plan is not None:
        steps = getattr(plan, "steps", None)
        if steps is None and isinstance(plan, dict):
            steps = plan.get("steps") or []
        s["cleaning_steps"] = len(steps) if steps else 0
    if state.get("final_answer"):
        s["has_final_answer"] = True
    if state.get("model_metadata"):
        s["has_model"] = True
    return s


def serialize_pipeline_result(result: dict) -> dict:
    """Turn graph invoke result into a JSON-friendly dict (no DataFrame blobs)."""
    out: Dict[str, Any] = {}

    findings = result.get("investigation_findings")
    if findings is not None:
        out["investigation_findings"] = _json_safe(findings)

    plan = result.get("current_plan")
    if plan is not None:
        out["current_plan"] = _json_safe(plan)

    history = result.get("cleaning_history", [])
    if history:
        out["cleaning_history"] = _json_safe(history)

    out["pass_count"] = result.get("pass_count", 0)
    out["pass_history"] = _json_safe(result.get("pass_history", []))
    out["retry_count"] = result.get("retry_count", 0)
    out["latest_error"] = result.get("latest_error")

    clean_df = result.get("clean_df")
    if clean_df is not None and isinstance(clean_df, pd.DataFrame):
        out["clean_dataframe"] = _json_safe(clean_df)
        out["clean_csv"] = clean_df.to_csv(index=False)

    meta = result.get("model_metadata")
    if meta is not None:
        out["model_metadata"] = _json_safe(meta)

    out["final_answer"] = result.get("final_answer")

    return out


def prepare_interactive(user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile the dataset and return column list + LLM-ranked target suggestions
    (same logic as CLI target selection, without input()).
    """
    sys.path.insert(0, str(REPO_ROOT))
    from plugins.profiler import generate_profile  # noqa: E402
    from core.agents.target_selector import rank_target_candidates  # noqa: E402

    profile = generate_profile(df, detailed_profiler=True, target_column=None)
    columns = profile.get("columns", [])
    all_names = [c["name"] for c in columns]
    columns_info = [
        {"name": c["name"], "type": c.get("inferred_type", "unknown")}
        for c in columns
    ]

    suggestions: List[Dict[str, str]] = []
    try:
        candidates = rank_target_candidates(user_query, columns_info)
        suggestions = [{"name": c.name, "rationale": c.rationale} for c in candidates]
    except Exception as e:
        for name in sorted(all_names)[:8]:
            suggestions.append(
                {"name": name, "rationale": f"(ranking unavailable: {type(e).__name__})"}
            )

    return {
        "ok": True,
        "columns": all_names,
        "suggestions": suggestions,
        "row_count": len(df),
        "column_count": len(all_names),
    }


def run_pipeline(
    user_query: str,
    df: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    """
    Invoke the compiled LangGraph app with stdout/stderr captured.

    target_column must be a valid column in df (sets state so the CLI
    target_selector does not call input()).
    """
    sys.path.insert(0, str(REPO_ROOT))
    from core.pipeline.graph import app  # noqa: E402
    from core.logger import setup_logger  # noqa: E402

    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(output_dir / "verbose.log"))

    initial_state = {
        "user_query": user_query,
        "working_df": df,
        "retry_count": 0,
        "tool_call_count": 0,
        "pass_count": 0,
        "target_column": target_column,
    }

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    error: Optional[str] = None
    result: Optional[dict] = None

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            result = app.invoke(initial_state)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    console_log = buf_out.getvalue() + buf_err.getvalue()

    if error:
        return {
            "ok": False,
            "error": error,
            "console_log": console_log,
        }

    assert result is not None
    payload = serialize_pipeline_result(result)
    payload["ok"] = True
    payload["console_log"] = console_log
    return payload


def run_pipeline_stream(
    user_query: str,
    df: pd.DataFrame,
    target_column: str,
) -> Iterator[Dict[str, Any]]:
    """
    Stream LangGraph state after each super-step (values mode) as progress events,
    then emit a final complete or error event.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from core.pipeline.graph import app  # noqa: E402
    from core.logger import setup_logger  # noqa: E402

    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(output_dir / "verbose.log"))

    initial_state = {
        "user_query": user_query,
        "working_df": df,
        "retry_count": 0,
        "tool_call_count": 0,
        "pass_count": 0,
        "target_column": target_column,
    }

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    final_state: Optional[dict] = None

    last_progress_sig: Optional[str] = None
    tick = 0
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            for state_update in app.stream(initial_state, stream_mode="values"):
                final_state = state_update
                tick += 1
                summ = summarize_graph_state(state_update)
                sig = json.dumps(summ, sort_keys=True, default=str)
                console = (buf_out.getvalue() + buf_err.getvalue())[-8000:]
                if sig != last_progress_sig or tick % 4 == 0:
                    last_progress_sig = sig
                    yield {
                        "type": "progress",
                        "summary": summ,
                        "console_snippet": console,
                    }
    except Exception as e:
        yield {
            "type": "error",
            "message": f"{type(e).__name__}: {e}",
            "console_log": buf_out.getvalue() + buf_err.getvalue(),
        }
        return

    console_log = buf_out.getvalue() + buf_err.getvalue()

    if final_state is None:
        yield {
            "type": "error",
            "message": "Pipeline produced no state",
            "console_log": console_log,
        }
        return

    try:
        payload = serialize_pipeline_result(final_state)
        payload["ok"] = True
        payload["console_log"] = console_log
        yield {"type": "complete", "result": payload}
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Serialize failed: {type(e).__name__}: {e}",
            "console_log": console_log,
        }


def ndjson_dumps(obj: dict) -> bytes:
    """One NDJSON line as UTF-8 bytes."""
    return (json.dumps(obj, default=str) + "\n").encode("utf-8")
