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
export INVESTIGATOR_MODEL=gpt-5-2025-08-07   # default
export CODEGEN_MODEL=gpt-4.1-2025-04-14      # default
export ANSWER_MODEL=gpt-4.1-2025-04-14       # default
```