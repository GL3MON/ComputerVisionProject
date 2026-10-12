# Checkpoint 2100: PMC-VQA evaluation

Evaluated all 2,000 test examples. Pass@1: **41.35%**; pass@2: **56.85%**. Failed examples count as incorrect.

## Comparison with saved runs

| Model | Examples | Pass@1 | Pass@2 | Failed |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,000 | 30.40% | 41.05% | 5 |
| SFT | 2,000 | 41.55% | 56.00% | 5 |
| GRPO checkpoint-2100 | 2,000 | 41.35% | 56.85% | 0 |

Relative to the saved SFT run: -0.20 percentage points at pass@1 and +0.85 percentage points at pass@2.

Every historical baseline question, options, and ground-truth answer was checked against the current run by example index. These are historical comparisons: sampling is stochastic and the baseline result files do not record their server configuration. The differences are not a controlled causal estimate of the effect of GRPO.

## Checkpoint and protocol

- Checkpoint: `/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100`.
- Base model: `GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged`.
- Vision and language LoRA adapters were merged; representative weights from both towers were checked for finite values and changes from the base in [merge_validation.json](merge_validation.json).
- Dataset: `OctoMed/PMC-VQA`, `test` split; fingerprint `6ecd63043d85c3a1`.
- Original benchmark prompt, image JPEG quality 95, thinking enabled, two attempts per example.
- Attempt temperatures: 0.6 and 0.75; top-p 0.8; top-k 20; up to 4,096 completion tokens per attempt.
- Scoring: extract `<answer>...</answer>` when present and normalize the option letter. Pass@1 uses the first attempt; pass@2 counts an example correct when either attempt is correct.
- vLLM: Triton language attention, PyTorch vision attention, bfloat16 weights, 8,192-token context, 16 concurrent benchmark workers. FlashInfer sampling and DeepGEMM warmup disabled.

## Run diagnostics

- Failed examples: 0 / 2,000.
- Truncated attempts (finish reason `length`): 0.
- Completion tokens: 1,373,739.
- Prompt tokens: 1,235,914.
- Evaluation wall time: 39.01 minutes.
- Completion throughput: 586.88 tokens/second.
- Mean example latency (both attempts, successful examples): 17.68 seconds. Requests ran concurrently, so this differs from wall time per example.

## Artifacts

- [results.json](results.json): per-example predictions and both attempts.
- [results_summary.json](results_summary.json): final metrics and evaluation settings.
- [comparison_summary.json](comparison_summary.json): baseline and checkpoint scores.
- [server_metadata.json](server_metadata.json): inference command and GPU/backend settings.
- [README.md](README.md): artifact guide and reproduction/resume command.

Existing base and SFT results were preserved. Merged weights and runtime logs remain local and are ignored by Git.
