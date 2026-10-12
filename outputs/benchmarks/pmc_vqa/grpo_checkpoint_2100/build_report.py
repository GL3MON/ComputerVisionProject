"""Validate full-run artifacts and write the checkpoint evaluation report."""
import json
from pathlib import Path

run = Path(__file__).resolve().parent
summary = json.loads((run / "results_summary.json").read_text())
results = json.loads((run / "results.json").read_text())
assert summary["complete"], "Full evaluation is still in progress"
assert summary["requested_examples"] == summary["test_split_size"] == len(results) == 2000
assert sorted(r["index"] for r in results) == list(range(2000))
assert summary["evaluated_examples"] == len(results)
for r in results:
    if "error" not in r:
        assert len(r["attempts"]) == summary["pass_at_k"] == 2
        correct = [a["normalized_prediction"] == r["normalized_ground_truth"] for a in r["attempts"]]
        assert correct == [a["correct"] for a in r["attempts"]]
        assert r["correct_pass1"] == correct[0]
        assert r["correct_passk"] == any(correct)
    else:
        assert not r["correct_pass1"] and not r["correct_passk"]
for metric in ["correct_pass1", "correct_passk"]:
    key = "accuracy_pass1" if metric == "correct_pass1" else "accuracy_passk"
    assert abs(sum(r[metric] for r in results) / len(results) - summary[key]) < 1e-12
failures = [r for r in results if "error" in r]
assert len(failures) == summary["failed_examples"]
baselines = json.loads((run / "baseline_comparison.json").read_text())
by_index = {r["index"]: r for r in results}
comparison = []
for baseline in baselines:
    baseline_rows = json.loads((run.parent / baseline["result_file"]).read_text())
    aligned = baseline["examples"] == len(results) and all(
        row["index"] in by_index and all(by_index[row["index"]][key] == row[key]
        for key in ["question", "options", "ground_truth"])
        for row in baseline_rows
    )
    assert aligned, "Historical baseline examples do not match the current benchmark"
    comparison.append({k:v for k,v in baseline.items() if k != "dataset_alignment"})
comparison.append({"label":"GRPO checkpoint-2100", "examples":len(results),
                   "accuracy_pass1":summary["accuracy_pass1"], "accuracy_passk":summary["accuracy_passk"],
                   "failed_examples":len(failures)})
report = ["# Checkpoint 2100: PMC-VQA evaluation", "",
          f"Evaluated all {len(results):,} test examples. Pass@1: **{summary['accuracy_pass1']:.2%}**; "
          f"pass@2: **{summary['accuracy_passk']:.2%}**. Failed examples count as incorrect.", "",
          "## Comparison with saved runs", "",
          "| Model | Examples | Pass@1 | Pass@2 | Failed |",
          "| --- | ---: | ---: | ---: | ---: |"]
for row in comparison:
    report.append(f"| {row['label']} | {row['examples']:,} | {row['accuracy_pass1']:.2%} | "
                  f"{row['accuracy_passk']:.2%} | {row['failed_examples']} |")
sft = next(r for r in comparison if r["label"] == "SFT")
d1 = (summary["accuracy_pass1"] - sft["accuracy_pass1"]) * 100
d2 = (summary["accuracy_passk"] - sft["accuracy_passk"]) * 100
report += ["", f"Relative to the saved SFT run: {d1:+.2f} percentage points at pass@1 and "
           f"{d2:+.2f} percentage points at pass@2.", "",
           "Every historical baseline question, options, and ground-truth answer was checked against the current run by example index. "
           "These are historical comparisons: sampling is stochastic and the baseline result files do not record their server "
           "configuration. The differences are not a controlled causal estimate of the effect of GRPO.", "",
           "## Checkpoint and protocol", "",
           f"- Checkpoint: `{summary['checkpoint']}`.",
           "- Base model: `GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged`.",
           "- Vision and language LoRA adapters were merged; representative weights from both towers were checked "
           "for finite values and changes from the base in [merge_validation.json](merge_validation.json).",
           f"- Dataset: `{summary['dataset']}`, `{summary['split']}` split; fingerprint `{summary['dataset_fingerprint']}`.",
           "- Original benchmark prompt, image JPEG quality 95, thinking enabled, two attempts per example.",
           "- Attempt temperatures: 0.6 and 0.75; top-p 0.8; top-k 20; up to 4,096 completion tokens per attempt.",
           "- Scoring: extract `<answer>...</answer>` when present and normalize the option letter. Pass@1 uses "
           "the first attempt; pass@2 counts an example correct when either attempt is correct.",
           "- vLLM: Triton language attention, PyTorch vision attention, bfloat16 weights, 8,192-token context, "
           "16 concurrent benchmark workers. FlashInfer sampling and DeepGEMM warmup disabled.", "",
           "## Run diagnostics", "",
           f"- Failed examples: {len(failures)} / {len(results):,}.",
           f"- Truncated attempts (finish reason `length`): {summary['truncated_attempts']}.",
           f"- Completion tokens: {summary['completion_tokens']:,}.",
           f"- Prompt tokens: {summary['prompt_tokens']:,}.",
           f"- Evaluation wall time: {summary['wall_time_seconds']/60:.2f} minutes.",
           f"- Completion throughput: {summary['completion_tokens_per_second']:.2f} tokens/second."]
if summary["average_example_latency_seconds"] is not None:
    report.append(f"- Mean example latency (both attempts, successful examples): "
                  f"{summary['average_example_latency_seconds']:.2f} seconds. Requests ran concurrently, "
                  "so this differs from wall time per example.")
if failures:
    report += ["", "### Request failures", "", "| Example index | Error |", "| ---: | --- |"]
    for row in failures:
        error = row["error"].replace("|", chr(92) + "|").replace(chr(10), " ")
        report.append(f"| {row['index']} | {error} |")
report += ["", "## Artifacts", "",
           "- [results.json](results.json): per-example predictions and both attempts.",
           "- [results_summary.json](results_summary.json): final metrics and evaluation settings.",
           "- [comparison_summary.json](comparison_summary.json): baseline and checkpoint scores.",
           "- [server_metadata.json](server_metadata.json): inference command and GPU/backend settings.",
           "- [README.md](README.md): artifact guide and reproduction/resume command.", "",
           "Existing base and SFT results were preserved. Merged weights and runtime logs remain local and are ignored by Git."]
(run / "REPORT.md").write_text("\n".join(report) + "\n")
(run / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
validation = {"complete_test_split":True,"examples":len(results),"unique_indices":True,
              "scoring_recomputed":True,"baseline_questions_options_and_answers_aligned":True,
              "failed_examples":len(failures),"truncated_attempts":summary["truncated_attempts"]}
(run / "evaluation_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print("Report saved:",run / "REPORT.md")
print(json.dumps(comparison,indent=2))
