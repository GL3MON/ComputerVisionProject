"""Build a portable vLLM command with the validated Triton configuration."""

import argparse
import os
import shlex
import subprocess
import sys
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root


def server_command(config):
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config["model"],
        "--served-model-name",
        config.get("served_name", config["model"]),
        "--host",
        config.get("host", "127.0.0.1"),
        "--port",
        str(config.get("port", 8000)),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.get("max_model_len", 8192)),
        "--gpu-memory-utilization",
        str(config.get("gpu_memory_utilization", 0.65)),
        "--max-num-seqs",
        str(config.get("max_num_seqs", 16)),
        "--attention-config",
        '{"backend":"TRITON_ATTN"}',
        "--mm-encoder-attn-backend",
        "TORCH_SDPA",
        "--enforce-eager",
    ]
    if config.get("kv_cache_dtype"):
        command.extend(["--kv-cache-dtype", config["kv_cache_dtype"]])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/serving/model.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--served-name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
        config["served_name"] = args.model
    if args.served_name:
        config["served_name"] = args.served_name
    if args.port:
        config["port"] = args.port
    command = server_command(config)
    settings = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
        "OMP_NUM_THREADS": "2",
    }
    if args.dry_run:
        print(shlex.join(command))
        print(settings)
        return
    environment = os.environ.copy()
    environment.update(settings)
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
