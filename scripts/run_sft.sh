#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec accelerate launch --config_file "$PROJECT_ROOT/configs/accelerate/sft.yaml" "$PROJECT_ROOT/scripts/train_sft.py" --config "$PROJECT_ROOT/configs/training/sft.yaml" "$@"
