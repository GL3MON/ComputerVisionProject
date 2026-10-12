# Checkpoint 2100 PMC-VQA evaluation

Checkpoint: `/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100`.
Base model: `GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged`.

Both vision and language LoRA adapters are merged for inference. The local merged weights in `merged/` are ignored by Git. The full evaluation uses the cached `OctoMed/PMC-VQA` test split (2,000 examples), the original benchmark prompt, two attempts with temperatures 0.6 and 0.75, top-p 0.8, top-k 20, thinking enabled, and a maximum of 4,096 completion tokens per attempt. Pass@1 scores the first attempt; pass@2 scores whether either attempt matches the normalized correct option. Failed examples count as incorrect.

## Artifacts

| File | Purpose |
| --- | --- |
| `merge_checkpoint.py` | Reproduce the local merge of both adapters; refuses to overwrite an existing export. |
| `merge_metadata.json` | Source checkpoint, base model, merge targets, and dtype. |
| `merge_validation.json` | Finite representative vision and language weights and verified changes from the base. |
| `server_metadata.json` | Server command, GPU identifier, and backend configuration. |
| `run_med_benchmark.original.py` | Original benchmark script preserved before adding configurable runs. |
| `baseline_comparison.json` | Existing base and SFT scores, example counts, failures, and question alignment. |
| `smoke_results.json`, `smoke_results_summary.json` | Small inference check; these are not full benchmark scores. |
| `results.json`, `results_summary.json` | Full-run predictions and saved progress/metrics. Check `complete` and `evaluated_examples` in the summary before treating the scores as final. |
| `REPORT.md` | Final evaluation findings, generated after the full run completes. |
| `build_report.py`, `evaluation_validation.json` | Recompute scores and verify full coverage and baseline alignment. |
| `comparison_summary.json` | Final baseline and checkpoint scores. |

Runtime logs live in `outputs/logs/benchmark_grpo_checkpoint_2100/` and are ignored by Git. Existing base and SFT benchmark files are preserved.

## Run or resume the benchmark

Start the server with the command and environment flags recorded in `server_metadata.json`. The server uses port 8002, Triton language attention, PyTorch vision attention, and disabled FlashInfer sampling and DeepGEMM warmup.

From the project root, run:

```bash
/home/harishr/miniconda3/envs/med/bin/python run_med_benchmark.py \
  --base-url http://127.0.0.1:8002/v1 \
  --model medical-grpo-checkpoint-2100 \
  --checkpoint /home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100 \
  --output outputs/benchmarks/pmc_vqa/grpo_checkpoint_2100/results.json \
  --workers 16 --offline
```

Add `--resume` to continue an interrupted run with the same settings. The script refuses to overwrite existing result files without that flag.

Comparisons to the saved base and SFT runs are historical comparisons. Sampling is stochastic, and their server configuration is not recorded in those result files, so differences should not be interpreted as a controlled causal estimate of the checkpoint's effect.
