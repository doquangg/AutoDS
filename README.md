# AutoDS
A library for taking unclean data from raw to insights, via conversation.

# Intended Usage
AutoDS is designed for questions that require **supervised machine learning** — any question where you want to understand, explain, or predict a target variable. It automates the full pipeline from raw, messy data to business-friendly insights: data profiling → quality investigation → cleaning → model training (via AutoGluon) → answer generation.

**Supported task types:**
- **Regression**: "What drives high hospital bills?" / "Can we forecast monthly revenue?"
- **Binary classification**: "What predicts patient readmission?" / "Which customers will churn?"
- **Multiclass classification**: "What determines a patient's risk category?"

**Out of scope:** Simple descriptive or aggregation queries that don't require ML modeling — e.g., "What's the average bill?", "How many patients visited last month?", "Show me all visits in January." These are better served by SQL or BI tools.

# Instructions
1) Create a conda environment via: `conda env create -f env.yml` from the root directory.
2) Set your OpenAI API key:
```
export OPENAI_API_KEY=sk-...
```
3) Optionally override the default models per agent role:
```
export INVESTIGATOR_MODEL=gpt-5.4         # default
export CODEGEN_MODEL=gpt-5.4-mini         # default
export ANSWER_MODEL=gpt-5.4-mini          # default
export TARGET_SELECTOR_MODEL=gpt-5.4-nano # default
```
4) To run the system, use one of the three commands below:
```
python scripts/run_graph.py (no change in output) # default
AUTODS_VERBOSE=1 python scripts/run_graph.py # verbose output, with truncation
AUTODS_VERBOSE=full python scripts/run_graph.py # verbose output, no truncation
```
When verbose mode is enabled, logger output is automatically written to `output/verbose.log`.
If you also want shell-level stderr redirection, you can still append `2>output/verbose.log`.
