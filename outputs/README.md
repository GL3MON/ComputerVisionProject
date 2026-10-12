# Output index

All generated artifacts are grouped here. Every existing file was retained; the original names and folder structure within each artifact group are preserved.

| Location | Contents |
| --- | --- |
| `benchmarks/pmc_vqa/` | Base and SFT PMC-VQA benchmark result JSON files; [checkpoint 2100 evaluation](benchmarks/pmc_vqa/grpo_checkpoint_2100/README.md) contains the new run and report |
| `analysis/attention/qwen3vl_attention_maps/` | Vision attention figures |
| `analysis/attribution/base/` | Base-model attribution figures and metrics |
| `analysis/attribution/medical/` | Medical-model attribution figures and metrics |
| `analysis/comparisons/medical_vs_base/` | Medical-versus-base comparison figures |
| `analysis/comparisons/attribution_maps/` | Paired attribution comparison figures |
| `analysis/comparisons/attribution_differences/` | Attribution difference figures and CSV |
| `classifiers/base/` | Base linear classifier weights, embeddings, and labels |
| `classifiers/medical/` | Medical linear classifier weights, embeddings, and labels |
| `experiments/grpo_checkpoint_2100/` | GRPO encoder experiment, report, scripts, classifiers, attribution, comparisons, and validation |
| `rollouts/rollouts.jsonl` | Rollout checker results |
| `rollouts/grpo-medical-o1/completions/` | GRPO completion parquet files |
| `logs/wandb/` | All existing W&B runs, logs, and symbolic links |
| `cache/vision_probe_cache/` | Vision probe cache directory |
| `qwen3-vl-2b-medical-lora/` | SFT adapter and training checkpoints |
| `qwen3-vl-2b-medical-grpo/` | GRPO checkpoints and completions |
| `organization/` | Move manifest, preservation verification, and original copies of modified scripts |

Training checkpoint directories retain their established paths. Scripts now use the grouped paths for generated outputs and classifier inputs. SFT and GRPO scripts default W&B storage to `outputs/logs/`, while respecting an explicitly configured `WANDB_DIR`. Benchmark and rollout scripts create their destination folders when needed.

See [the checkpoint experiment report](experiments/grpo_checkpoint_2100/REPORT.md) for its results. Existing reports, metadata, serialized artifacts, and logs were kept unchanged, so historical path strings inside them may refer to the original locations. Consult [the move manifest](organization/move_manifest.json) to find the corresponding current paths. Paths in runnable project scripts were updated.

`organization/source_backups/` contains the exact original bytes of every modified script, organized by its former path. No original data or log was removed or overwritten by this reorganization.

See the [combined PMC-VQA report](benchmarks/pmc_vqa/COMBINED_REPORT.md) for benchmark metrics across all three models, including prompt counts and token usage.

Read the [PMC-VQA analysis report](benchmarks/pmc_vqa/ANALYSIS_REPORT.md) for interpretation of the combined metrics and paired outcomes.
