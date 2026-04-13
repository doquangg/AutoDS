# AutoDS

A library for taking unclean data from raw to insights, via conversation.

## Intended Usage

AutoDS is designed for questions that require **supervised machine learning** — any question where you want to understand, explain, or predict a target variable. It automates the full pipeline from raw, messy data to business-friendly insights: data profiling → target selection → data quality investigation → cleaning → feature engineering → model training (via AutoGluon) → answer generation.

**Supported task types:**
- **Regression**: "What drives high hospital bills?" / "Can we forecast monthly revenue?"
- **Binary classification**: "What predicts patient readmission?" / "Which customers will churn?"
- **Multiclass classification**: "What determines a patient's risk category?"

**Out of scope:** Simple descriptive or aggregation queries that don't require ML modeling — e.g., "What's the average bill?", "How many patients visited last month?", "Show me all visits in January." These are better served by SQL or BI tools.

## Architecture

AutoDS is built as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine with multi-pass cleaning and a feature engineering stage. The pipeline uses separate LLM agents for investigation, code generation, and answer synthesis — each with its own message history and system prompt.

### Pipeline Flow

```
START
  │
  ▼
profiler ◄──────────────────────────────┐
  │                                     │
  ▼                                     │
target_selector (human-in-the-loop)     │
  │                                     │
  ▼                                     │
investigator ◄──┐                       │
  │              │                      │
  ├─[tools?]─► tools                    │
  │                                     │
  ▼                                     │
code_generator                          │
  │                                     │
  ▼                                     │
sandbox                                 │
  │                                     │
  ├─[error?]─► code_generator (retry)   │
  │                                     │
  ▼                                     │
re_profile                              │
  │                                     │
  ├─[done]──► feature_engineering       │
  │              │                      │
  │              ▼                      │
  │           autogluon                 │
  │              │                      │
  │           answer_agent              │
  │              │                      │
  │            END                      │
  │                                     │
  └─[next_pass]──► pass_reset ──────────┘
```

### Key Design Decisions

1. **Split agents**: The investigator (diagnosis) and code generator (Python code) are separate agents. Investigation happens once per pass; only the code generator retries on sandbox failure.
2. **Tool loop cap**: The investigator can call at most 20 tools per pass to prevent runaway loops.
3. **Retry isolation**: On sandbox failure, only the code generator re-runs. Investigation findings are preserved within a pass (up to 3 retries).
4. **Multi-pass cleaning**: After each successful sandbox execution, the pipeline re-profiles the data. If critical violations remain (and max passes haven't been reached), a new cleaning pass begins. Up to 5 passes.
5. **Feature engineering**: Runs once after cleaning completes. Internally loops through up to 3 rounds of planner → codegen → sandbox → re-profile, with a defensive target-leakage lint.
6. **Subprocess sandbox**: LLM-generated code executes in an isolated subprocess with a 120-second timeout, so crashes don't take down the main process.
7. **Human-in-the-loop target selection**: After profiling, an LLM ranks candidate target columns and the user confirms via CLI prompt (or `AUTO_TARGET_COLUMN` env var for non-interactive mode).

## Project Structure

```
AutoDS/
├── core/
│   ├── agents/
│   │   ├── agents.py               # Investigator, Code Generator, and Answer agents
│   │   ├── feature_engineering.py   # FE planner + codegen agents and the FE graph node
│   │   └── target_selector.py       # Human-in-the-loop target column selection
│   ├── pipeline/
│   │   ├── graph.py                 # LangGraph state machine assembly
│   │   └── state.py                 # AgentState TypedDict (shared state schema)
│   ├── runtime/
│   │   ├── sandbox.py               # Subprocess-based sandbox for LLM-generated code
│   │   └── tools.py                 # Read-only investigation tools for the investigator
│   ├── prompts/                     # System prompts for each agent (*.txt files)
│   ├── schemas.py                   # Pydantic schemas for all data contracts
│   └── logger.py                    # Centralized verbose logging
├── plugins/
│   ├── profiler.py                  # Data profiling (column stats, ydata-profiling, quality scores)
│   └── modeller.py                  # AutoGluon model training wrapper
├── scripts/
│   ├── run_graph.py                 # Main entry point — run the full pipeline
│   ├── run_graph_orcheval.py        # Run with OrchEval tracing and reporting
│   ├── download_benchmarks.py       # Download UCI benchmark datasets
│   └── evaluate_benchmarks.py       # End-to-end benchmark evaluation harness
├── data/
│   ├── sample_data/healthcare/      # Sample dirty healthcare dataset
│   └── benchmark_metadata.yaml      # UCI benchmark dataset configurations
├── env.yml                          # Conda environment specification
└── output/                          # Pipeline outputs (cleaned CSV, model, logs)
```

## Setup

1. Create a conda environment:
```bash
conda env create -f env.yml
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Configuration

### Model Selection

Override the default model for any agent role via environment variables:

| Environment Variable     | Default          | Agent Role                  |
|--------------------------|------------------|-----------------------------|
| `INVESTIGATOR_MODEL`     | `gpt-5.4`       | Data quality investigator   |
| `CODEGEN_MODEL`          | `gpt-5.4-mini`  | Cleaning code generator     |
| `ANSWER_MODEL`           | `gpt-5.4-mini`  | Final answer synthesis      |
| `TARGET_SELECTOR_MODEL`  | `gpt-5.4-nano`  | Target column ranking       |
| `FE_PLANNER_MODEL`       | `gpt-5.4`       | Feature engineering planner |
| `FE_CODEGEN_MODEL`       | `gpt-5.4-mini`  | Feature engineering codegen |

To use a local OpenAI-compatible server (Ollama, vLLM, etc.):
```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

### AutoGluon Training

| Environment Variable     | Default            | Description                              |
|--------------------------|--------------------|------------------------------------------|
| `AUTOGLUON_TIME_LIMIT`   | `300`              | Training budget in seconds               |
| `AUTOGLUON_PRESETS`      | `medium_quality`   | AutoGluon preset string                  |

### Optional Integrations

| Environment Variable   | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `TAVILY_API_KEY`       | Enables web search for semantic verification of data values        |
| `AUTO_TARGET_COLUMN`   | Bypass interactive target column selection (set to column name)    |

## Running

### Standard Execution

```bash
python scripts/run_graph.py
```

### Verbose Mode

```bash
AUTODS_VERBOSE=1 python scripts/run_graph.py      # verbose output, with truncation
AUTODS_VERBOSE=full python scripts/run_graph.py    # verbose output, no truncation
```

When verbose mode is enabled, logger output is automatically written to `output/verbose.log`.

### OrchEval Tracing

Run the pipeline with [OrchEval](https://github.com/doquangg/orcheval) tracing to capture structured trace events and generate cost, timeline, routing, convergence, and retry reports:

```bash
pip install orcheval
python scripts/run_graph_orcheval.py
```

Outputs are saved to `output/orcheval_trace.json` and `output/orcheval_report.json`.

## Benchmarking

AutoDS includes a benchmark harness that evaluates the pipeline end-to-end against UCI datasets with published baselines:

```bash
# Download benchmark datasets (requires ucimlrepo)
python scripts/download_benchmarks.py

# Run all benchmarks (or a single dataset with --only)
python scripts/evaluate_benchmarks.py
python scripts/evaluate_benchmarks.py --only ai4i_2020
```

The harness performs a stratified 80/10/10 train/val/test split, runs the AutoDS pipeline on the train fold, replays the cleaning and feature engineering recipes on the held-out folds, trains a fresh AutoGluon model with explicit validation data, and scores on the test fold. Results are saved to `output/benchmarks/results.json`.

Available benchmark datasets are configured in `data/benchmark_metadata.yaml`.

## Investigation Tools

The investigator agent has access to seven read-only tools for inspecting data:

| Tool                       | Purpose                                                    |
|----------------------------|------------------------------------------------------------|
| `inspect_rows`             | Sample rows matching a pandas query expression             |
| `cross_column_frequency`   | Crosstab of top value combinations between two columns     |
| `temporal_ordering_check`  | Check whether dates maintain causal ordering               |
| `value_distribution`       | Histogram (numeric) or value counts (categorical)          |
| `null_co_occurrence`       | Find columns that tend to be null together                 |
| `correlation_scan`         | Top N columns most correlated with a target column         |
| `web_search`               | Verify semantic plausibility via web search (requires Tavily) |
