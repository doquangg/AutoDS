"""
End-to-end test scaffold: real LangGraph + stubbed LLMs, healthcare CSV.

Intended to validate that:
  - POST /sessions kicks off the pipeline
  - target_selection_required arrives
  - POST /resume unblocks the pipeline
  - pipeline_complete eventually arrives
  - Q&A responds after completion

This is a placeholder to be fleshed out during manual demo verification
(Task 16). Keep the skip until the run has been verified manually.
"""
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEALTHCARE_CSV = (
    REPO_ROOT
    / "data"
    / "sample_data"
    / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)


@pytest.fixture(autouse=True)
def stub_all_llms(monkeypatch):
    """Keep the test offline, deterministic, and fast."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("AUTODS_INTERACTIVE_MODE", "web")
    # Skip AutoGluon — too slow and CPU-heavy for tests
    monkeypatch.setenv("AUTODS_SKIP_MODEL_TRAINING", "1")


@pytest.mark.asyncio
@pytest.mark.skipif(not HEALTHCARE_CSV.exists(), reason="sample CSV missing")
async def test_full_run_with_target_selection_and_qa():
    pytest.skip(
        "E2E with real graph + stubs — implement step-by-step during "
        "integration (Task 16)"
    )
