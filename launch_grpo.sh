CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b \
BASE_MODEL=Qwen/Qwen3-VL-2B-Thinking \
SFT_CHECKPOINT=/home/harishr/Med/outputs/qwen3-vl-2b-medical-lora/checkpoint-4828 \
VERIFIER_URL=http://127.0.0.1:8000/v1/completions \
VERIFIER_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=./outputs/qwen3-vl-2b-medical-grpo \
NUM_GENERATIONS=4 \
MAX_COMPLETION_LENGTH=2048 \
WANDB_PROJECT=medical-o1-grpo \
WANDB_RUN_NAME=qwen3-vl-2b-medical-grpo \
accelerate launch \
    --config_file accelerate_grpo.yaml \
    grpo.py



# # Terminal 1 — policy vLLM
# CUDA_VISIBLE_DEVICES=1 \
# trl vllm-serve \
#     --model Qwen/Qwen3-VL-2B-Instruct \
#     --port 8001 \
#     --max-model-len 4096 \
#     --kv-cache-dtype fp8 \
#     --gpu-memory-utilization 0.65


# CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1\
# python -c "
# from vllm.platforms.cuda import NvmlCudaPlatform
# import os
# print('UUID:', os.environ['CUDA_VISIBLE_DEVICES'])
# print('Physical ID:', NvmlCudaPlatform.device_control_id_to_physical_device_id(os.environ['CUDA_VISIBLE_DEVICES']))
# "