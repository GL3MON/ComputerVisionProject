vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --max-model-len 10880 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.35