"""
Start the AutoDS web UI (FastAPI + Uvicorn).

From repo root, with OPENAI_API_KEY set:

  python scripts/run_web.py

Then open http://127.0.0.1:8000
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
