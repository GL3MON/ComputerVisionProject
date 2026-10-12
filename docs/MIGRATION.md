# Migration from the original scripts

Every original root Python, shell, and YAML file is preserved under `archive/pre_modular_refactor/`, with checksums in its manifest. Historical experiment implementations under `outputs/experiments/` remain provenance artifacts.

| Original root script | Modular command |
| --- | --- |
| `sft.py` | `medical-vlm train-sft` or `python scripts/train_sft.py` |
| `grpo.py` | `medical-vlm train-grpo` or `python scripts/train_grpo.py` |
| `run_med_benchmark.py` | `medical-vlm benchmark` or `python scripts/run_med_benchmark.py` |
| `check_rollouts.py` | `medical-vlm rollouts` |
| `classifier_train.py` | `medical-vlm probe` |
| `generate_analysis_maps.py` | `medical-vlm attribution` |
| `compare_attribution_diff.py` | `medical-vlm compare-attribution` |
| `generate_attention_maps.py` | `medical-vlm attention` |
| `merge_and_push.py` | `medical-vlm merge`, followed by `medical-vlm upload`; merge also accepts explicit `--repo` |
| `test_checkpoint.py` | `medical-vlm inspect-checkpoint` |
| `test_sft_checkpoint.py` | `medical-vlm smoke` |
| `test_med_sft_checkpoint.py` | `medical-vlm smoke --medical` |
| `run_sft.sh`, `launch_grpo.sh`, `check_rollout.sh`, `launch_verifier.sh` | Portable counterparts under `scripts/` |
| `sft_training_args.yaml` | `configs/training/sft.yaml` |
| `qwen_sft.yaml`, `accelerate_grpo.yaml` | `configs/accelerate/sft.yaml`, `configs/accelerate/grpo.yaml` |
| Report builders under `outputs/benchmarks/pmc_vqa/` | `medical-vlm report`; small compatibility launchers preserve the old builder paths |

## Behavior retained

PMC-VQA prompting, option normalization, two-attempt defaults, failure-inclusive accuracy, resume guards, and generation settings remain stable. Training hyperparameters and strict verifier checklist semantics are retained. GRPO initialization from SFT is separate from resuming a GRPO run. No new training or benchmark inference is run during the refactor.

## Deliberate corrections

- Imports no longer load weights, download datasets, create W&B runs, or execute training.
- Rollout JSONL now writes one complete JSON object per line; the original script wrote multi-line JSON inside a `.jsonl` file. Existing rollout files are preserved.
- Feature extraction uses `pooler_output` when available and supports the historical tuple interface.
- Attribution restores packed patch order and normalizes top-fraction mass. Historical figures and metrics are unchanged; older unnormalized metrics are not numerically interchangeable with new results.
- Attention captures actual eager visual attention weights and reports mean received attention. It does not assume a CLS token, discard arbitrary tokens, or claim to compute attention rollout.
- CPU vision inference uses float32 rather than float16.
- GPU/MIG identifiers and absolute home-directory paths are configuration/environment values, not launch-script constants.

`mig_patch.sh` is archived because it targets vLLM 0.17 and rewrites installed package code. The current serving command uses validated vLLM 0.22 Triton settings and applies no installed-package patches.

Historical reports may contain older root-script reproduction commands. Use the command mapping above. Existing output indexes and model-upload provenance remain available in their original locations.
