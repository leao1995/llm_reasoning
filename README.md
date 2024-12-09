
## Known Issues

1. If vllm complains about "libnvJitLink.so.12", run the following command:

```bash
export LD_LIBRARY_PATH=/opt/conda/envs/llm_reasoning/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```