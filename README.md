# AutoDS
A library for taking unclean data from raw to insights, via conversation.

# Instructions
1) Create a conda environment via: `conda env create -f env.yml` from the root directory.
2) Set your OpenAI API key:
```
export OPENAI_API_KEY=sk-...
```
3) Optionally override the default models per agent role:
```
export INVESTIGATOR_MODEL=gpt-5.1-2025-11-13   # default
export CODEGEN_MODEL=gpt-5.1-2025-11-13     # default
export ANSWER_MODEL=gpt-5.1-2025-11-13      # default
```
4) To run the system:
Normal run: python scripts/run_graph.py (no change in output)
Verbose run: AUTODS_VERBOSE=1 python scripts/run_graph.py
Save to file: AUTODS_VERBOSE=1 python scripts/run_graph.py 2>verbose.log
