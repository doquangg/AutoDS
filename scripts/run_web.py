"""
Launch the AutoDS web demo.

Usage:
  # Dev: also run `cd web && npm run dev` in a second terminal
  python scripts/run_web.py

  # Prod-ish: build the frontend first, then this serves dist/ at /
  cd web && npm run build && cd ..
  python scripts/run_web.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import uvicorn
from fastapi.staticfiles import StaticFiles

from core.web.app import create_app
from core.logger import setup_logger


def main() -> None:
    setup_logger()
    app = create_app()

    # If the frontend has been built, serve it at /
    dist = REPO_ROOT / "web" / "dist"
    if dist.exists():
        app.mount(
            "/", StaticFiles(directory=str(dist), html=True), name="frontend"
        )
    else:
        @app.get("/")
        def hint():
            return {
                "hint": (
                    "Frontend not built. Run `cd web && npm run dev` "
                    "(visit :5173) or `npm run build` (visit :8000)."
                )
            }

    host = os.environ.get("AUTODS_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTODS_WEB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
