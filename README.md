# Medical vision-language research

Modular workflows for Qwen3-VL supervised training, GRPO verification, PMC-VQA benchmarking, frozen vision probes, attribution, and model export. Existing outputs and historical scripts are preserved.

## Repository layout

```text
src/medical_vlm/
  core/         configuration, paths, atomic persistence
  data/         reasoning examples, image encoding, MedMNIST
  training/     SFT, GRPO, callbacks, verifier rewards
  evaluation/   inference clients, scoring, benchmarks, smoke checks, rollouts
  models/       model loading, inspection, merge, upload, Triton serving
  vision/       features, classifiers, probes, attribution, attention, comparisons
  reports/      metric aggregation, paired statistics, rendering, report CLI
configs/        training, Accelerate, vision, serving, report profiles
scripts/        thin Python entry points and portable shell launchers
tests/          offline regression checks and synthetic vision tests
docs/           architecture and migration guide
outputs/        existing artifacts; heavyweight files and logs remain ignored
archive/        exact original root scripts/configs and historical report builders
```

## Install

Use the existing `med` environment on this machine:

```bash
conda activate med
python -m pip install -e . --no-deps --no-build-isolation
medical-vlm --help
```

For a new environment, install the extras needed for your workflow, for example `python -m pip install -e '.[training,vision,dev]'`. Choose a CUDA-compatible PyTorch environment before adding serving dependencies; the existing environment uses PyTorch 2.11.0+cu129, Transformers 5.6.2, and vLLM 0.22.0.

Commands can also run directly from the checkout, for example `python scripts/run_med_benchmark.py --help`, without an editable installation. Project-relative paths resolve against this checkout rather than the caller's working directory. `MEDICAL_VLM_ROOT` can select another project root.

## Workflows

### Serve and evaluate

```bash
medical-vlm serve --dry-run
medical-vlm serve
medical-vlm benchmark --base-url http://127.0.0.1:8000/v1 \
  --model medical-grpo-checkpoint-2100 \
  --output outputs/benchmarks/pmc_vqa/new_run.json --workers 16 --offline
```

The server defaults to the published GRPO model and the validated Triton settings. Set `CUDA_VISIBLE_DEVICES` in your environment to choose a GPU or MIG device. `--offline` requires a cached dataset. Benchmark outputs must be new unless `--resume` is supplied; failed examples count as incorrect. Model names, endpoints, sample limits, generation limits, and concurrency are command arguments.

### Train

```bash
bash scripts/run_sft.sh
bash scripts/launch_grpo.sh
# Or select a profile directly:
medical-vlm train-sft --config configs/training/sft.yaml
medical-vlm train-grpo --config configs/training/grpo.yaml
```

Inspect the profiles before training. The GRPO profile initializes the existing SFT adapter and expects a verifier at its configured endpoint. `--resume` resumes a GRPO run separately from SFT initialization. Explicit legacy environment overrides remain supported for GRPO. Importing modules or requesting help does not launch training, load model weights, or contact inference services.

### Vision analysis

```bash
medical-vlm probe --config configs/vision/probe.yaml
medical-vlm attribution --config configs/vision/attribution.yaml
medical-vlm compare-attribution --config configs/vision/comparison.yaml
medical-vlm attention --config configs/vision/attention.yaml
```

Probe, attribution, and attention accept adapter paths where appropriate. Shared feature pooling supports the current Transformers `pooler_output` interface and older tuple outputs. New attribution maps restore Qwen patch packing to raster order, and top-fraction mass is normalized. Attention maps show mean received visual self-attention, without a CLS-token assumption. See [migration notes](docs/MIGRATION.md) for differences from historical figures.

### Generate reports from saved predictions

```bash
medical-vlm report --config configs/reports/pmc_vqa.yaml
```

This writes regenerated reports under `outputs/benchmarks/pmc_vqa/reports/`, leaving existing reports untouched. Use `--output-dir` for another destination or explicitly `--overwrite` for generated files. Run specifications, labels, prediction paths, and summary paths live in the report profile.

Read the existing [combined report](outputs/benchmarks/pmc_vqa/COMBINED_REPORT.md), [analysis report](outputs/benchmarks/pmc_vqa/ANALYSIS_REPORT.md), and [output guide](OUTPUTS.md).

### Merge and upload

```bash
medical-vlm merge --adapter /path/to/checkpoint --output outputs/models/my_run/merged
medical-vlm upload --folder outputs/models/my_run/merged --repo YOUR_ACCOUNT/YOUR_MODEL
medical-vlm inspect-checkpoint --checkpoint /path/to/adapter
```

Merge resolves the base from adapter metadata unless `--base-model` is supplied and includes every LoRA target, including vision targets. Export refuses to overwrite an existing folder. Upload happens only when explicitly invoked; authenticate with `hf auth login` first. The previously uploaded checkpoint remains at [GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged](https://huggingface.co/GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged).

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests use recorded outputs or synthetic CPU tensors and mocked clients; they do not start GPU training or publish models. See [architecture](docs/ARCHITECTURE.md) for module responsibilities and [migration](docs/MIGRATION.md) for old-to-new commands.
