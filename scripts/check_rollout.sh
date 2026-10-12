#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$PROJECT_ROOT/scripts/check_rollouts.py" --model Qwen/Qwen3-VL-2B-Instruct --num-generations 8 --max-tokens 4096 --limit 1 "$@"
