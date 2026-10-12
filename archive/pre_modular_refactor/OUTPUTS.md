# Output folders

Generated results live under [`outputs/`](outputs/). This guide explains what each folder contains and where to start when reviewing an experiment. Some folders exist only in the local workspace because datasets, checkpoints, weights, caches, and logs are excluded by [`.gitignore`](.gitignore).

## Folder guide

| Folder | Contents and purpose |
| --- | --- |
| [`outputs/benchmarks/pmc_vqa/`](outputs/benchmarks/pmc_vqa/) | PMC-VQA benchmark result JSON files for the base and supervised fine-tuned (SFT) models, plus [`grpo_checkpoint_2100/`](outputs/benchmarks/pmc_vqa/grpo_checkpoint_2100/README.md) for the new checkpoint evaluation, predictions, summaries, and report. |
| [`outputs/analysis/attention/`](outputs/analysis/attention/) | Vision attention figures in `qwen3vl_attention_maps/`. |
| [`outputs/analysis/attribution/`](outputs/analysis/attribution/) | Separate `base/` and `medical/` folders containing per-image attribution figures and `attribution_metrics.csv`. |
| [`outputs/analysis/comparisons/`](outputs/analysis/comparisons/) | `medical_vs_base/` contains model comparison figures; `attribution_maps/` contains paired attribution figures; `attribution_differences/` contains difference figures and `attribution_differences.csv`. |
| [`outputs/classifiers/`](outputs/classifiers/) | Separate `base/` and `medical/` linear classifier runs. Each has classifier weights and an `embeddings/` folder with train, validation, and test features and labels. Weights are ignored; embeddings and labels remain included. |
| [`outputs/experiments/`](outputs/experiments/) | Self-contained experiment folders with reports, scripts, metrics, predictions, visualizations, provenance, and validation files. See the checkpoint experiment below. |
| [`outputs/rollouts/`](outputs/rollouts/) | `rollouts.jsonl` contains rollout checker results; `grpo-medical-o1/completions/` contains generated completions in Parquet format. |
| [`outputs/qwen3-vl-2b-medical-lora/`](outputs/qwen3-vl-2b-medical-lora/) | SFT adapter metadata and training checkpoint folders. Weights, serialized training state, checkpoint folders, and merged model exports are ignored. |
| [`outputs/qwen3-vl-2b-medical-grpo/`](outputs/qwen3-vl-2b-medical-grpo/) | GRPO training outputs. Checkpoint folders are ignored; generated completion Parquet files under `completions/` remain included. |
| [`outputs/organization/`](outputs/organization/) | Records from organizing the workspace: `move_manifest.json` maps old paths to new paths; `verification.json` records preservation checks; `source_backups/` preserves original versions of modified scripts. |
| `outputs/logs/wandb/` | Local W&B run history and logs. The entire `outputs/logs/` tree is ignored. |
| `outputs/cache/vision_probe_cache/` | Local vision probe cache. The entire `outputs/cache/` tree is ignored. |

## Checkpoint 2100 experiment

Start with [`outputs/experiments/grpo_checkpoint_2100/REPORT.md`](outputs/experiments/grpo_checkpoint_2100/REPORT.md) for the experiment setup, classifier results, attribution method, and limitations.

| Location within `outputs/experiments/grpo_checkpoint_2100/` | What it contains |
| --- | --- |
| `encoder/` | Exported vision encoder weights, configuration, processor files, and merge verification. Encoder weights are ignored; configuration and verification remain included. |
| `classifier/` | Classifier metrics, predictions, embeddings, labels, and weights. Classifier weights are ignored; the other results remain included. |
| `attribution/` | Per-image PNG visualizations, raw NumPy attribution arrays, CSV metrics, and a JSON summary. |
| `comparison_base_sft_rl/` | Comparisons of base, SFT, and reinforcement-learning (RL) models: a report, summary, metrics, overview figure, per-image maps, raw arrays, and supporting scripts. |
| `comparison_base_sft_rl/base/classifier/` and `comparison_base_sft_rl/sft/classifier/` | Supporting classifier runs for the base and SFT comparisons. |
| `comparison_base_sft_rl/sft_wrong_rl_right/` | Selected prediction cases, comparison maps, raw arrays, CSV results, selection metadata, summary, validation, and a separate report. |
| `manifest.json`, `status.json`, `validation.json` | Experiment provenance, execution status, and recorded validation checks. |
| `train_probe.py`, `analyze_attribution.py` | Scripts for reproducing the classifier and attribution stages. |

For the comparison results, read [`comparison_base_sft_rl/REPORT.md`](outputs/experiments/grpo_checkpoint_2100/comparison_base_sft_rl/REPORT.md). For the selected prediction cases, read [`sft_wrong_rl_right/REPORT.md`](outputs/experiments/grpo_checkpoint_2100/comparison_base_sft_rl/sft_wrong_rl_right/REPORT.md).

## Files kept in Git

Reports, source scripts, figures, CSV metrics, benchmark JSON, rollout JSONL, generated completion Parquet files, raw attribution `.npy` arrays, prediction `.npz` files, embeddings, labels, and organization records remain eligible for tracking.

Downloaded datasets, checkpoint folders, merged model exports, model and classifier weights, serialized training state, caches, Python bytecode, local environments, logs, W&B run files, and local credentials are ignored. File formats such as `.pt`, `.npy`, `.npz`, and `.parquet` are not excluded wholesale, so generated artifacts in those formats remain available.

Ignore rules affect Git tracking; they do not delete local files. Files already tracked by Git are not removed from tracking simply by adding an ignore rule.

## Paths in older artifacts

Historical logs and metadata were preserved during the reorganization, so they may still mention earlier locations. Use [`outputs/organization/move_manifest.json`](outputs/organization/move_manifest.json) to find their current paths. The original folder index is also available in [`outputs/README.md`](outputs/README.md).

See the [combined PMC-VQA report](outputs/benchmarks/pmc_vqa/COMBINED_REPORT.md) for all three models, question and prompt counts, token usage, accuracy, and latency. CSV and JSON exports accompany the report.

Read the [PMC-VQA analysis report](outputs/benchmarks/pmc_vqa/ANALYSIS_REPORT.md) for interpretation of model accuracy, prompt recovery, token usage, paired regressions, and comparison limits.
