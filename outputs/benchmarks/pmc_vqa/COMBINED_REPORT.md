# Combined PMC-VQA benchmark report

Compares the saved Base, SFT, and GRPO checkpoint-2100 runs on the same **2,000 question/image examples**. Questions, options, and ground-truth answers were checked for exact alignment by example index.

## Accuracy and number of prompts

A **question** is one dataset example. A **prompt** here is one generation attempt sent for that example. Each run planned two attempts per question; the count of saved successful prompts excludes failed or discarded attempts.

| Model | Questions | Planned prompts | Saved successful prompts | Failed questions | Pass@1 | Pass@2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 2,000 | 4,000 | 3,990 | 5 | 30.40% | 41.05% |
| SFT | 2,000 | 4,000 | 3,990 | 5 | 41.55% | 56.00% |
| GRPO checkpoint-2100 | 2,000 | 4,000 | 4,000 | 0 | 41.35% | 56.85% |

Checkpoint-2100 versus saved SFT: **-0.20 percentage points** at pass@1 and **+0.85 percentage points** at pass@2. These small differences come from historical runs with stochastic sampling; they do not establish a controlled improvement from GRPO.

Across the three runs there are **6,000 model-example evaluations** on the same 2,000 dataset examples, **12,000 planned prompts**, and **11,980 saved successful prompts**.

The five failed examples in each historical run have no retained attempts, leaving ten planned attempts unretained per run. This is not a count of ten failed HTTP requests. Actual network request totals were not recorded and can include retries or interrupted work. Smoke-check prompts are excluded from this report.

## Token usage

Prompt count and prompt-token count measure different things: one inference prompt can contain hundreds of input tokens. Token totals below sum server-reported usage over saved attempts.

| Metric | Base | SFT | GRPO checkpoint-2100 |
| --- | ---: | ---: | ---: |
| Prompt/input tokens | 2,181,130 | 2,181,130 | 1,235,914 |
| Completion/output tokens | 9,507,746 | 1,344,522 | 1,373,739 |
| Total tokens | 11,688,876 | 3,525,652 | 2,609,653 |
| Mean input tokens per saved prompt | 546.65 | 546.65 | 308.98 |
| Mean output tokens per saved prompt | 2,382.89 | 336.97 | 343.43 |
| Correct first-attempt questions | 608 | 831 | 827 |
| Questions correct in either attempt | 821 | 1,120 | 1,137 |

Combined token usage: **5,598,174 input tokens**, **12,226,007 output tokens**, **17,824,181 total tokens**.

Input-token totals differ despite matched questions and options. Historical server/processor configurations are not fully recorded, so matching examples does not establish identical image preprocessing or tokenization. Token differences cannot be attributed solely to GRPO.

## Latency and generation diagnostics

| Metric | Base | SFT | GRPO checkpoint-2100 |
| --- | ---: | ---: | ---: |
| Mean example latency, both attempts (seconds) | 46.55 | 6.21 | 17.68 |
| Median example latency (seconds) | 48.53 | 6.23 | 17.79 |
| 95th-percentile example latency (seconds) | 78.95 | 8.49 | 23.78 |
| Mean saved-prompt latency (seconds) | 23.28 | 3.10 | 8.84 |
| Summed example latencies (hours) | 25.80 | 3.44 | 9.82 |
| Recorded evaluation wall time (minutes) | Not recorded | Not recorded | 39.01 |
| Output tokens per wall second | Not recorded | Not recorded | 586.88 |
| Attempts with exactly 4,096 output tokens | 1347 | 0 | 0 |
| Confirmed truncated attempts | Not recorded | Not recorded | 0 |

Latency averages exclude failed examples. The checkpoint run used 16 concurrent workers, so summed request latencies overlap and are not elapsed wall time. Historical wall time is unavailable; these rows should not be treated as a controlled speed comparison. Reaching exactly 4,096 output tokens does not by itself confirm truncation: baseline files did not save finish reasons.

## Common successful subset

All three models have saved successful results for **1,995 common examples**. This table excludes the same five examples from every model, removing failure-coverage differences from the denominator.

| Model | Common examples | Pass@1 | Pass@2 |
| --- | ---: | ---: | ---: |
| Base | 1,995 | 30.48% | 41.15% |
| SFT | 1,995 | 41.65% | 56.14% |
| GRPO checkpoint-2100 | 1,995 | 41.40% | 56.89% |

Baseline failures occurred at indices `961` through `965`. The Base run rejected inputs longer than its 8,192-token context; the SFT run reported image-processor failures. Checkpoint-2100 succeeded on those examples. The headline accuracy table includes all 2,000 examples and counts failures as incorrect.

## Sources and reproduction

- [Base predictions](pmc_vqa_base_vllm_results.json).
- [SFT predictions](pmc_vqa_sft_vllm_results.json).
- [Checkpoint-2100 predictions](grpo_checkpoint_2100/results.json) and [summary](grpo_checkpoint_2100/results_summary.json).
- [Checkpoint evaluation report](grpo_checkpoint_2100/REPORT.md) and [protocol](grpo_checkpoint_2100/README.md).
- [CSV metrics](combined_metrics.csv) and [JSON metrics with definitions](combined_metrics.json).

Regenerate from the project root with:

```bash
python outputs/benchmarks/pmc_vqa/build_combined_report.py
```

No inference is rerun. Original prediction files are read without modification.
