"""
Prompt loader for the AutoDS pipeline.

Each prompt lives in its own .txt file in this directory. Prompts that use
Python str.format() placeholders (e.g. {pass_number}) are loaded as-is —
callers are responsible for calling .format() on them.
"""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Read and return the contents of ``<name>.txt`` from the prompts directory."""
    return (_PROMPT_DIR / f"{name}.txt").read_text()


INVESTIGATOR_SYSTEM_PROMPT = load_prompt("investigator_system")
INVESTIGATOR_REEXAM_PROMPT = load_prompt("investigator_reexam")
CODEGEN_SYSTEM_PROMPT = load_prompt("codegen_system")
ANSWER_SYSTEM_PROMPT = load_prompt("answer_system")
TARGET_RANKING_SYSTEM_PROMPT = load_prompt("target_ranking_system")
