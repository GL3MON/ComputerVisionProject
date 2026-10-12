#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$PROJECT_ROOT/scripts/serve_model.py" --config "$PROJECT_ROOT/configs/serving/verifier.yaml" "$@"
