# AutoDS
A library for taking unclean data from raw to insights, via conversation.

# Instructions
1) Create a conda enviornment via: `conda env create -f environment.yml` from the root directory.
2) Instanitate a VLLM instance with 
```
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser=hermes \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85
```