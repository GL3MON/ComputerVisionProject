# Analysis of the PMC-VQA benchmark results

The largest observed improvement is between the saved Base and SFT runs. GRPO checkpoint-2100 performs close to SFT: it answers four fewer questions correctly on the first attempt and covers 17 more questions when either of two attempts is allowed. Its run also completes the five examples that failed in the historical runs. These results support describing the checkpoint as similar in first-answer accuracy, with a small observed gain in two-attempt coverage and complete request coverage; they do not establish a consistent accuracy advantage from GRPO.

## Evaluation basis

All three runs evaluate the same 2,000 question/image examples. Questions, answer options, and ground-truth answers match by example index. Headline scores count failed examples as incorrect. Each run plans two generation attempts per example. The GRPO run uses temperatures 0.6 and 0.75, top-p 0.8, top-k 20, thinking enabled, and a 4,096-token completion allowance.

| Model | First attempt correct | Correct in either attempt | Pass@1 | Pass@2 | Failed examples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 608 | 821 | 30.40% | 41.05% | 5 |
| SFT | 831 | 1,120 | 41.55% | 56.00% | 5 |
| GRPO checkpoint-2100 | 827 | 1,137 | 41.35% | 56.85% | 0 |

## SFT accounts for the main observed change

SFT increases pass@1 by **11.15 percentage points** and pass@2 by **14.95 percentage points** relative to Base. In question counts, that is 223 additional first-attempt successes and 299 additional two-attempt successes.

At the same time, mean output length falls from 2,382.89 to 336.97 tokens per saved prompt, a **85.86% reduction**. The adapted model therefore produces much shorter responses while achieving higher measured accuracy. This is an observed association across saved runs; it does not prove that shorter reasoning causes better answers.

The Base run has 1,347 attempts with exactly 4,096 output tokens (33.76% of saved prompts). Neither adapted run has an attempt at that length. This is consistent with more pressure on the Base generation budget, but historical finish reasons were not saved, so those attempts cannot all be classified as confirmed truncations.

## GRPO versus SFT: a small net difference with substantial turnover

The first-attempt score moves from 41.55% to 41.35% (**-0.20 percentage points**). The two-attempt score moves from 56.00% to 56.85% (**+0.85 percentage points**). These net changes conceal many questions on which the models disagree:

| Paired outcome across 2,000 examples | Pass@1 | Pass@2 |
| --- | ---: | ---: |
| GRPO correct, SFT incorrect | 295 | 297 |
| SFT correct, GRPO incorrect | 299 | 280 |
| Both correct | 532 | 840 |
| Both incorrect | 874 | 583 |
| Net change in correct questions | -4 | +17 |

GRPO does not simply preserve every SFT success and add new ones. At pass@2 it gains 297 questions and loses 280, leaving the net gain of 17. This makes per-example regression analysis more useful than relying on the small headline difference alone.

An exploratory exact, two-sided McNemar test gives **p=0.9020** for pass@1 and **p=0.5054** for pass@2. Neither difference reaches the conventional 0.05 threshold under this question-level test. The exact test uses discordant matched outcomes and a binomial distribution; see the [statsmodels method documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html). These calculations do not establish model equivalence or account for shared-image clustering, generation-seed variability, training-seed variability, or differing inference configurations.

## What the second prompt adds

| Model | Planned prompts | Saved successful prompts | Additional questions recovered by attempt 2 | Pass@2 minus pass@1 |
| --- | ---: | ---: | ---: | ---: |
| Base | 4,000 | 3,990 | 213 | 10.65 pp |
| SFT | 4,000 | 3,990 | 289 | 14.45 pp |
| GRPO checkpoint-2100 | 4,000 | 4,000 | 310 | 15.50 pp |

GRPO's second attempt recovers 310 first-attempt misses, compared with 289 for SFT and 213 for Base. This explains why GRPO can have a slightly lower first-attempt score but a slightly higher pass@2 score. It recovers more examples with another sample, rather than improving the first answer consistently.

Pass@2 is an **oracle-style coverage metric**: the evaluator uses the known correct answer to decide whether either response succeeded. It is not the accuracy of an automatically selected final answer. A system that produces one answer needs a separately evaluated selection or verification method before it can claim the pass@2 score as its final-answer accuracy.

For GRPO, the second round uses 1,308,456 recorded input-plus-output tokens and recovers 310 additional questions, or approximately 4,221 second-round tokens per additional covered question. This describes the benchmark's extra token budget, not a monetary cost or a guaranteed inference-time benefit.

The number of prompts is separate from the number of prompt tokens. Each run evaluates 2,000 questions and plans 4,000 prompts. The Base and SFT files each retain 3,990 successful attempts because five examples have no saved attempts; GRPO retains all 4,000. Across models, this is 12,000 planned prompts and 11,980 retained successful prompts, on the same 2,000 dataset examples. These counts exclude the smoke check and do not reconstruct actual network requests, retries, or unretained work from the interrupted evaluation.

## Token savings require separating inputs from outputs

| Token metric | Base | SFT | GRPO checkpoint-2100 |
| --- | ---: | ---: | ---: |
| Input tokens | 2,181,130 | 2,181,130 | 1,235,914 |
| Output tokens | 9,507,746 | 1,344,522 | 1,373,739 |
| Total tokens | 11,688,876 | 3,525,652 | 2,609,653 |
| Mean output tokens per saved prompt | 2382.89 | 336.97 | 343.43 |

GRPO uses **25.98% fewer total recorded tokens** than SFT, but its input-token total is **43.34% lower**, while its output length is slightly higher. Mean output length rises from 336.97 to 343.43 tokens per saved prompt (**+1.92%**). The lower total is therefore driven by the input side, not shorter GRPO answers.

The examples match, but historical server/processor configurations are incomplete. Different image preprocessing or tokenization could change the input budget. The recorded totals do not establish that GRPO itself reduces compute or cost. On the output side, SFT and GRPO are very close in generation length.

## Answer formatting also affects the measured scores

| Response diagnostic | Base | SFT | GRPO checkpoint-2100 |
| --- | ---: | ---: | ---: |
| Saved responses without an answer tag | 1225 | 0 | 0 |
| Normalized predictions outside the supplied option labels | 1376 | 12 | 24 |

Base has 1,225 saved responses without an `<answer>` tag and 1,376 normalized predictions that do not match a supplied option label. The adapted runs always include an answer tag, but SFT still has 12 predictions outside the supplied option labels and GRPO has 24. A missing tag alone does not guarantee an incorrect score: a bare option letter can still be scored correctly by the existing extractor.

This benchmark measures both answering and adherence to the extraction/scoring format. Some Base errors may reflect unfinished or noncanonical answers rather than solely a failure of image reasoning. These counts do not establish that the underlying medical answer was correct; that requires inspecting the responses. Any alternative answer parser should be fixed in advance and applied equally to all runs.

## Failure coverage and latency

Base and SFT each fail on examples 961–965. Base reports input lengths exceeding an 8,192-token context, and SFT reports image-processor failures. GRPO processes all five examples successfully, with one additional first-attempt success and two additional pass@2 successes among those examples. This is a real improvement in this run's request coverage, but it cannot be assigned to learned weights without controlling preprocessing and serving settings.

Removing those same five examples from every model leaves 1,995 common successful examples:

| Model | Common-subset pass@1 | Common-subset pass@2 |
| --- | ---: | ---: |
| Base | 30.48% | 41.15% |
| SFT | 41.65% | 56.14% |
| GRPO checkpoint-2100 | 41.40% | 56.89% |

The common-subset GRPO–SFT differences are about -0.25 percentage points at pass@1 and +0.75 percentage points at pass@2. Excluding request failures therefore does not change the main interpretation: first-attempt performance is similar, with a small observed two-attempt coverage gain.

Mean example latency is 46.55 seconds for Base, 6.21 for SFT, and 17.68 for GRPO. GRPO's recorded wall time is 39.01 minutes with 16 concurrent workers. Historical wall times are unavailable. Per-request latency and summed latency are affected by serving load and overlapping requests; these results do not establish that SFT is intrinsically faster or that GRPO is intrinsically slower.

## Findings and next experiments

The evidence supports three findings: SFT has the largest observed accuracy and response-length change from Base; checkpoint-2100 largely preserves SFT-level first-answer accuracy; and the new run has slightly higher two-attempt coverage with complete request coverage. A stable GRPO accuracy advantage remains unproven.

1. Re-evaluate SFT and checkpoint-2100 with the same processor settings, image-resolution limits, context budget, server version, hardware allocation, generation parameters, and concurrency. This addresses the observed input-token and failure-coverage differences.
2. Repeat sampling with multiple recorded seeds and retain per-example outcomes. Report variability alongside paired comparisons; account for questions sharing an image when estimating uncertainty.
3. Inspect the 297 pass@2 gains and 280 regressions by question type and image. The net score alone obscures substantial changes in which examples succeed.
4. Evaluate response selection or verification if two prompts are intended for practical use. Report the accuracy of the selected answer and its extra token and latency budget.
5. Save finish reasons, processor configuration, image-token settings, elapsed wall time, and actual request/retry counts for every future run so reliability and efficiency comparisons are reproducible.

## Published checkpoint

The full merged checkpoint, processor, model card, and evaluation metrics are available at [GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged](https://huggingface.co/GL3MON/qwen3-vl-2b-medical-pmcvqa-grpo-merged). Both vision and language adapters are included in the merged weights. Upload provenance is recorded in [hf_upload_metadata.json](grpo_checkpoint_2100/hf_upload_metadata.json).

## Evidence and reproducibility

- [Combined report](COMBINED_REPORT.md) and [combined metrics](combined_metrics.json).
- [Derived analysis metrics](analysis_metrics.json): paired outcomes, recovery counts, token changes, and exact test values.
- [Checkpoint evaluation report](grpo_checkpoint_2100/REPORT.md).
- [Base predictions](pmc_vqa_base_vllm_results.json), [SFT predictions](pmc_vqa_sft_vllm_results.json), and [GRPO predictions](grpo_checkpoint_2100/results.json).

Regenerate with `python outputs/benchmarks/pmc_vqa/build_analysis_report.py`. The analysis reads the existing results; it does not rerun inference or modify prediction files.
