# Enhancing LLM Reasoning with Policy Guided Tree Search

## Serve vllm

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8004 --speculative-model meta-llama/Llama-3.2-1B-Instruct --speculative-max-model-len 8192 --max-model-len 8192 --num-speculative-tokens 5 --enable-prefix-caching
```

## Known Issues

1. If vllm complains about "libnvJitLink.so.12", run the following command:

```bash
export LD_LIBRARY_PATH=/opt/conda/envs/llm_reasoning/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

2. When using judges, speculative decoding and prefix caching give "Internal Service" errors.