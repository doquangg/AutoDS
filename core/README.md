## `core/`

Logic, state, and prompts for the LLM agents that drive the pipeline.

### Subpackages

- **`agents/`** — One module per LLM-driven role.
  - `agents.py` — investigator, cleaning code generator, and final answer agents.
  - `feature_engineering.py` — FE planner + codegen + the LangGraph node that loops up to 3 rounds of planner → codegen → sandbox → re-profile, with a defensive target-leakage lint.
  - `qa_agent.py` — streaming Q&A agent for follow-up questions about a completed run, grounded in the run's saved artifacts.
  - `target_selector.py` — LLM-ranked target column selection with a CLI prompt branch and a web/interrupt branch (gated by `AUTODS_INTERACTIVE_MODE`).

- **`pipeline/`** — LangGraph state machine.
  - `graph.py` — graph assembly, conditional edges, multi-pass termination logic. Compiled with a `MemorySaver` checkpointer (`pickle_fallback=True`) so DataFrames survive the human-in-the-loop interrupt at target selection.
  - `state.py` — `AgentState` TypedDict, the shared state schema for all nodes.

- **`runtime/`** — In-process tooling for agents.
  - `sandbox.py` — subprocess sandbox for LLM-generated cleaning and FE code (120s timeout).
  - `tools.py` — seven read-only investigation tools: `inspect_rows`, `cross_column_frequency`, `temporal_ordering_check`, `value_distribution`, `null_co_occurrence`, `correlation_scan`, `web_search`.

- **`web/`** — FastAPI backend for the chat UI.
  - `app.py` — REST + SSE endpoints (upload, run, resume, Q&A).
  - `runner.py` — async pipeline runner; emits typed events and supports interrupt-resume.
  - `session.py` — in-memory session manager with a ring-buffer for SSE replay on reconnect.
  - `qa.py` — Q&A adapter that streams through the session's shared SSE stream.
  - `events.py`, `log_sink.py` — typed event schema + bridge from autods log records to events.

### Top-level files

- `schemas.py` — Pydantic schemas for every data contract (profile, investigation findings, cleaning recipe, feature recipe, etc.).
- `logger.py` — centralized verbose logging that writes to `output/verbose.log` when `AUTODS_VERBOSE` is set.
- `prompts/` — system prompts for each agent as plain `.txt` files.
