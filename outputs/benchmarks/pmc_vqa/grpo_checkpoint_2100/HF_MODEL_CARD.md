---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged
license: apache-2.0
tags:
- qwen3-vl
- grpo
- medical
- vision-language
- pmc-vqa
---

# Qwen3-VL-2B Medical PMC-VQA GRPO — merged checkpoint 2100

This repository contains the full merged Qwen3-VL model from GRPO checkpoint **2100**, initialized from [GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged](https://huggingface.co/GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged). Both vision and language LoRA adapters were merged into the model weights. An additional PEFT adapter is not required for inference.

The repository name follows the SFT convention with `grpo` replacing `sft`. The PMC-VQA name comes from the SFT base and evaluation benchmark; the metrics below describe evaluation, not a claim about the GRPO training dataset.

## PMC-VQA evaluation

Evaluated on all **2,000** `OctoMed/PMC-VQA` test examples with two attempts per question.

| Model | Pass@1 | Pass@2 | Failed examples |
| --- | ---: | ---: | ---: |
| Saved base run | 30.40% | 41.05% | 5 |
| Saved SFT run | 41.55% | 56.00% | 5 |
| GRPO checkpoint-2100 | 41.35% | 56.85% | 0 |

The checkpoint produced **4,000 retained successful inference prompts**, using **1,235,914 input tokens** and **1,373,739 output tokens**, with **0 generation truncations**.

Pass@1 checks the first answer. Pass@2 checks whether either attempt matches the ground-truth option; it is an oracle-style metric and does not provide a method for selecting the correct answer at inference time. Failed examples count as incorrect.

The two attempt temperatures were 0.6 and 0.75, with top-p 0.8, top-k 20, thinking enabled, and at most 4,096 completion tokens per attempt. Images were sent as JPEG at quality 95. The server used bfloat16, an 8,192-token context, Triton language attention, PyTorch vision attention, and 16 concurrent benchmark workers.

These are historical comparisons from stochastic runs. Input-token totals and processor/server configurations differ, so the small accuracy differences do not establish a controlled improvement from GRPO.

See [evaluation summary](evaluation/results_summary.json), [combined metrics](evaluation/combined_metrics.json), and [adapter-merge checks](evaluation/merge_validation.json).

## Load the model

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

repo = "GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged"
processor = AutoProcessor.from_pretrained(repo)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    repo, dtype=torch.bfloat16, device_map="auto"
)
```

Use the processor's chat template for image-and-text inputs. The model and processor were exported with Transformers 5.6.2. The merged checkpoint was validated for finite representative vision and language weights and for updates from the SFT base.
