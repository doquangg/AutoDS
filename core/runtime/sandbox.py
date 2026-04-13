################################################################################
# Subprocess-based sandbox for executing LLM-generated cleaning code.
#
# Runs each CleaningStep's python_code in an isolated child process so that:
#   - Crashes in generated code don't take down the main process
#   - A timeout prevents infinite loops
#   - Per-step success/failure is logged for auditability
#
# FLOW:
#   Parent: serialize df → parquet, write cleaning script → temp dir
#   Child:  read parquet, exec each step in order, write output + log.json
#   Parent: read output parquet + log.json, return results
################################################################################

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.schemas import CleaningRecipe, CleaningStep, CleaningLogEntry


# Maximum wall-clock time (seconds) for the child process
SANDBOX_TIMEOUT = 120


def _sanitize_llm_python(code: str) -> str:
    """
    Replace common Unicode punctuation from LLM output with ASCII so execution
    is robust on Windows (avoids cp1252-only bytes in source / JSON edge cases).
    """
    if not code:
        return code
    repl = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\xa0": " ",
    }
    out = code
    for u, asc in repl.items():
        out = out.replace(u, asc)
    return out


def execute_cleaning_plan(
    df: pd.DataFrame,
    plan: CleaningRecipe,
) -> tuple[pd.DataFrame, list[CleaningLogEntry], Optional[str]]:
    """
    Execute a CleaningRecipe in an isolated subprocess.

    Args:
        df: The input DataFrame to clean.
        plan: A CleaningRecipe containing ordered CleaningSteps.

    Returns:
        (clean_df, log_entries, error)
        - clean_df: The cleaned DataFrame (original df if execution failed).
        - log_entries: Per-step audit trail.
        - error: None on success, or an error message string on failure.
    """
    with tempfile.TemporaryDirectory(prefix="autods_sandbox_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.parquet")
        output_path = os.path.join(tmpdir, "output.parquet")
        log_path = os.path.join(tmpdir, "log.json")
        script_path = os.path.join(tmpdir, "clean.py")
        steps_path = os.path.join(tmpdir, "steps.json")

        # --- Serialize inputs ---
        df.to_parquet(input_path)

        # Write steps as JSON so the child can read metadata for logging
        steps_data = []
        for step in plan.steps:
            d = step.model_dump()
            if "python_code" in d and isinstance(d["python_code"], str):
                d["python_code"] = _sanitize_llm_python(d["python_code"])
            steps_data.append(d)
        with open(steps_path, "w", encoding="utf-8") as f:
            json.dump(steps_data, f, ensure_ascii=True)

        # --- Generate the child script ---
        script = _build_child_script(input_path, output_path, log_path, steps_path)
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)

        # --- Run in subprocess ---
        child_env = os.environ.copy()
        if sys.platform == "win32":
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SANDBOX_TIMEOUT,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            logs = _read_partial_logs(log_path)
            return df, logs, f"Sandbox execution timed out ({SANDBOX_TIMEOUT}s)"

        # --- Read logs (always available, even on failure) ---
        logs = _read_partial_logs(log_path)

        # --- Handle failure ---
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Child process exited with code {result.returncode}"
            return df, logs, error_msg

        # --- Read cleaned DataFrame ---
        if not os.path.exists(output_path):
            return df, logs, "Child process succeeded but output file was not written"

        clean_df = pd.read_parquet(output_path)
        return clean_df, logs, None


def _build_child_script(
    input_path: str,
    output_path: str,
    log_path: str,
    steps_path: str,
) -> str:
    """
    Build the Python script that runs inside the child process.

    The script:
      1. Reads the input DataFrame from parquet
      2. Loads step metadata from JSON
      3. Executes each step's python_code in order, wrapped in try/except
      4. Writes per-step results to log.json
      5. Writes the final DataFrame to output parquet
    """
    return f'''\
# -*- coding: utf-8 -*-
import json
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd
import numpy as np

INPUT_PATH = {input_path!r}
OUTPUT_PATH = {output_path!r}
LOG_PATH = {log_path!r}
STEPS_PATH = {steps_path!r}


def main():
    df = pd.read_parquet(INPUT_PATH)

    with open(STEPS_PATH, encoding="utf-8") as f:
        steps = json.load(f)

    log_entries = []

    for step in steps:
        step_id = step["step_id"]
        operation = step["operation"]
        justification = step["justification"]
        code = step["python_code"]
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Execute the cleaning code in a namespace with df, pd, np
            namespace = {{"df": df, "pd": pd, "np": np}}
            exec(code, namespace)
            df = namespace["df"]

            log_entries.append({{
                "timestamp": timestamp,
                "step_id": step_id,
                "operation": operation,
                "justification": justification,
                "code_executed": code,
                "status": "SUCCESS",
            }})

        except Exception:
            err = traceback.format_exc()
            log_entries.append({{
                "timestamp": timestamp,
                "step_id": step_id,
                "operation": operation,
                "justification": justification,
                "code_executed": code,
                "status": "FAILED",
            }})

            # Write partial log before exiting
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2, ensure_ascii=True)

            # Print error to stderr for the parent to capture
            print(f"Step {{step_id}} ({{operation}}) failed:\\n{{err}}", file=sys.stderr)
            sys.exit(1)

    # All steps succeeded — write outputs
    df.to_parquet(OUTPUT_PATH)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
'''


def _read_partial_logs(log_path: str) -> list[CleaningLogEntry]:
    """Read log.json if it exists, returning CleaningLogEntry objects."""
    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [CleaningLogEntry(**entry) for entry in raw]
    except (json.JSONDecodeError, ValueError):
        return []
