"""Shared pytest fixtures for the web layer tests."""
import asyncio
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each async test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_autods_logger():
    """Make sure the autods logger has no leftover handlers between tests."""
    logger = logging.getLogger("autods")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)
