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
4) To run the system, use one of the three commands below:
```
python scripts/run_graph.py (no change in output) # default
AUTODS_VERBOSE=1 python scripts/run_graph.py # verbose output, with truncation
AUTODS_VERBOSE=full python scripts/run_graph.py # verbose output, no truncation
```
Append `2>verbose.log` to the end of the line to output to `verbose.log`
