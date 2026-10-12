"""Combine saved PMC-VQA benchmark metrics without rerunning inference."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
RUNS = [
    ("Base", "pmc_vqa_base_vllm_results.json"),
    ("SFT", "pmc_vqa_sft_vllm_results.json"),
    ("GRPO checkpoint-2100", "grpo_checkpoint_2100/results.json"),
]


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def percent(numerator, denominator):
    return numerator / denominator if denominator else None


def collect(label, filename, rows):
    assert len(rows) == 2000
    assert sorted(r["index"] for r in rows) == list(range(2000))
    attempts = [a for r in rows for a in r["attempts"]]
    successful = [r for r in rows if "error" not in r]
    failures = [r for r in rows if "error" in r]
    assert all(len(r["attempts"]) == 2 for r in successful)
    assert all(not r["attempts"] for r in failures)
    for row in successful:
        correctness = [a["normalized_prediction"] == row["normalized_ground_truth"] for a in row["attempts"]]
        assert correctness == [a["correct"] for a in row["attempts"]]
        assert row["correct_pass1"] == correctness[0]
        assert row["correct_passk"] == any(correctness)
        for key in ["prompt_tokens", "completion_tokens"]:
            assert row[key] == sum(a.get(key) or 0 for a in row["attempts"])
    for row in failures:
        assert not row["correct_pass1"] and not row["correct_passk"]
    prompt_tokens = sum(a.get("prompt_tokens") or 0 for a in attempts)
    completion_tokens = sum(a.get("completion_tokens") or 0 for a in attempts)
    assert prompt_tokens == sum(r.get("prompt_tokens") or 0 for r in rows)
    assert completion_tokens == sum(r.get("completion_tokens") or 0 for r in rows)
    latencies = [r["latency"] for r in successful if r.get("latency") is not None]
    attempt_latencies = [a["latency"] for a in attempts if a.get("latency") is not None]
    finish_reasons_recorded = bool(attempts) and all("finish_reason" in a for a in attempts)
    result = {
        "model": label, "source_file": filename,
        "question_examples": len(rows), "attempts_planned_per_question": 2,
        "planned_inference_prompts": len(rows) * 2,
        "recorded_successful_prompts": len(attempts),
        "unretained_planned_attempts": len(rows) * 2 - len(attempts),
        "actual_network_request_count": None,
        "successful_question_examples": len(successful), "failed_question_examples": len(failures),
        "failed_example_indices": [r["index"] for r in failures],
        "correct_pass1_count": sum(r["correct_pass1"] for r in rows),
        "correct_pass2_count": sum(r["correct_passk"] for r in rows),
        "accuracy_pass1": percent(sum(r["correct_pass1"] for r in rows), len(rows)),
        "accuracy_pass2": percent(sum(r["correct_passk"] for r in rows), len(rows)),
        "accuracy_pass1_successful_only": percent(sum(r["correct_pass1"] for r in successful), len(successful)),
        "accuracy_pass2_successful_only": percent(sum(r["correct_passk"] for r in successful), len(successful)),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "mean_prompt_tokens_per_recorded_prompt": percent(prompt_tokens, len(attempts)),
        "mean_completion_tokens_per_recorded_prompt": percent(completion_tokens, len(attempts)),
        "mean_example_latency_seconds": percent(sum(latencies), len(latencies)),
        "median_example_latency_seconds": median(latencies) if latencies else None,
        "p95_example_latency_seconds": quantile(latencies, 0.95),
        "mean_attempt_latency_seconds": percent(sum(attempt_latencies), len(attempt_latencies)),
        "sum_recorded_example_latency_seconds": sum(latencies),
        "wall_time_seconds": None, "completion_tokens_per_wall_second": None,
        "attempts_with_4096_completion_tokens": sum(a.get("completion_tokens") == 4096 for a in attempts),
        "finish_reasons_recorded": finish_reasons_recorded,
        "confirmed_truncated_attempts": sum(a.get("finish_reason") == "length" for a in attempts) if finish_reasons_recorded else None,
    }
    if label == "GRPO checkpoint-2100":
        summary = json.loads((ROOT / "grpo_checkpoint_2100/results_summary.json").read_text())
        assert summary["complete"] and summary["evaluated_examples"] == len(rows)
        assert summary["accuracy_pass1"] == result["accuracy_pass1"]
        assert summary["accuracy_passk"] == result["accuracy_pass2"]
        assert summary["prompt_tokens"] == prompt_tokens
        assert summary["completion_tokens"] == completion_tokens
        assert summary["truncated_attempts"] == result["confirmed_truncated_attempts"]
        result["wall_time_seconds"] = summary["wall_time_seconds"]
        result["completion_tokens_per_wall_second"] = summary["completion_tokens_per_second"]
    return result


def main():
    all_rows = [json.loads((ROOT / filename).read_text()) for _, filename in RUNS]
    reference = {r["index"]: r for r in all_rows[0]}
    for rows in all_rows[1:]:
        assert all(all(row[key] == reference[row["index"]][key]
                       for key in ["question", "options", "ground_truth"])
                   for row in rows), "Benchmark questions/options/answers do not align"
    metrics = [collect(label, filename, rows) for (label, filename), rows in zip(RUNS, all_rows)]
    common = set.intersection(*[{r["index"] for r in rows if "error" not in r} for rows in all_rows])
    for metric, rows in zip(metrics, all_rows):
        shared = [r for r in rows if r["index"] in common]
        metric["common_successful_examples"] = len(shared)
        metric["accuracy_pass1_common_subset"] = sum(r["correct_pass1"] for r in shared) / len(shared)
        metric["accuracy_pass2_common_subset"] = sum(r["correct_passk"] for r in shared) / len(shared)
    aggregate = {
        "distinct_dataset_examples": len(reference),
        "model_example_evaluations": sum(m["question_examples"] for m in metrics),
        "planned_inference_prompts": sum(m["planned_inference_prompts"] for m in metrics),
        "recorded_successful_prompts": sum(m["recorded_successful_prompts"] for m in metrics),
        "failed_model_example_evaluations": sum(m["failed_question_examples"] for m in metrics),
        "prompt_tokens": sum(m["prompt_tokens"] for m in metrics),
        "completion_tokens": sum(m["completion_tokens"] for m in metrics),
        "total_tokens": sum(m["total_tokens"] for m in metrics),
    }
    definitions = {
        "question_examples": "Dataset question/image examples evaluated once by each model; repeated generation attempts do not add questions.",
        "planned_inference_prompts": "Question examples multiplied by two planned generation attempts.",
        "recorded_successful_prompts": "Number of retained attempt objects in the result files. Each is one completed inference prompt.",
        "actual_network_request_count": "Not recorded. Retries, successful attempts discarded when another attempt failed, interrupted/unretained work, and smoke checks are not captured by the retained-attempt count.",
        "prompt_tokens": "Server-reported input tokens summed over retained attempts; distinct from the number of prompts.",
        "accuracy_pass1": "First attempt correct, divided by all 2,000 examples, including failed examples as incorrect.",
        "accuracy_pass2": "At least one of two attempts correct, divided by all 2,000 examples.",
        "mean_example_latency_seconds": "Mean sum of both attempt latencies on successful examples; concurrent request latencies overlap.",
        "wall_time_seconds": "Recorded evaluator elapsed time, available only for the checkpoint run; not inferred from summed latencies.",
        "confirmed_truncated_attempts": "Count of finish_reason=length when recorded; null means finish reasons were not saved.",
    }
    payload = {"created_at":datetime.now(timezone.utc).isoformat(), "dataset":"OctoMed/PMC-VQA", "split":"test",
               "aligned_questions_options_answers":True, "definitions":definitions,
               "aggregate":aggregate, "models":metrics}
    (ROOT / "combined_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (ROOT / "combined_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        for metric in metrics:
            row = dict(metric)
            row["failed_example_indices"] = json.dumps(row["failed_example_indices"])
            writer.writerow(row)
    report = ["# Combined PMC-VQA benchmark report", "",
              "Compares the saved Base, SFT, and GRPO checkpoint-2100 runs on the same **2,000 question/image examples**. "
              "Questions, options, and ground-truth answers were checked for exact alignment by example index.", "",
              "## Accuracy and number of prompts", "",
              "A **question** is one dataset example. A **prompt** here is one generation attempt sent for that example. "
              "Each run planned two attempts per question; the count of saved successful prompts excludes failed or discarded attempts.", "",
              "| Model | Questions | Planned prompts | Saved successful prompts | Failed questions | Pass@1 | Pass@2 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in metrics:
        report.append(f"| {m['model']} | {m['question_examples']:,} | {m['planned_inference_prompts']:,} | "
                      f"{m['recorded_successful_prompts']:,} | {m['failed_question_examples']} | "
                      f"{m['accuracy_pass1']:.2%} | {m['accuracy_pass2']:.2%} |")
    new, sft, base = metrics[2], metrics[1], metrics[0]
    report += ["", f"Checkpoint-2100 versus saved SFT: **{(new['accuracy_pass1']-sft['accuracy_pass1'])*100:+.2f} "
               f"percentage points** at pass@1 and **{(new['accuracy_pass2']-sft['accuracy_pass2'])*100:+.2f} "
               "percentage points** at pass@2. These small differences come from historical runs with stochastic sampling; "
               "they do not establish a controlled improvement from GRPO.", "",
               f"Across the three runs there are **{aggregate['model_example_evaluations']:,} model-example evaluations** "
               f"on the same {aggregate['distinct_dataset_examples']:,} dataset examples, "
               f"**{aggregate['planned_inference_prompts']:,} planned prompts**, and "
               f"**{aggregate['recorded_successful_prompts']:,} saved successful prompts**.", "",
               "The five failed examples in each historical run have no retained attempts, leaving ten planned attempts "
               "unretained per run. This is not a count of ten failed HTTP requests. Actual network request totals were "
               "not recorded and can include retries or interrupted work. Smoke-check prompts are excluded from this report.", "",
               "## Token usage", "",
               "Prompt count and prompt-token count measure different things: one inference prompt can contain hundreds "
               "of input tokens. Token totals below sum server-reported usage over saved attempts.", "",
               "| Metric | Base | SFT | GRPO checkpoint-2100 |", "| --- | ---: | ---: | ---: |"]
    for label,key,kind in [
        ("Prompt/input tokens","prompt_tokens","int"),
        ("Completion/output tokens","completion_tokens","int"),
        ("Total tokens","total_tokens","int"),
        ("Mean input tokens per saved prompt","mean_prompt_tokens_per_recorded_prompt","float"),
        ("Mean output tokens per saved prompt","mean_completion_tokens_per_recorded_prompt","float"),
        ("Correct first-attempt questions","correct_pass1_count","int"),
        ("Questions correct in either attempt","correct_pass2_count","int"),
    ]:
        values=[f"{m[key]:,}" if kind=="int" else f"{m[key]:,.2f}" for m in metrics]
        report.append("| "+label+" | "+" | ".join(values)+" |")
    report += ["", f"Combined token usage: **{aggregate['prompt_tokens']:,} input tokens**, "
               f"**{aggregate['completion_tokens']:,} output tokens**, "
               f"**{aggregate['total_tokens']:,} total tokens**.", "",
               "Input-token totals differ despite matched questions and options. Historical server/processor "
               "configurations are not fully recorded, so matching examples does not establish identical image "
               "preprocessing or tokenization. Token differences cannot be attributed solely to GRPO.", "",
               "## Latency and generation diagnostics", "",
               "| Metric | Base | SFT | GRPO checkpoint-2100 |", "| --- | ---: | ---: | ---: |"]
    for label,key in [
        ("Mean example latency, both attempts (seconds)","mean_example_latency_seconds"),
        ("Median example latency (seconds)","median_example_latency_seconds"),
        ("95th-percentile example latency (seconds)","p95_example_latency_seconds"),
        ("Mean saved-prompt latency (seconds)","mean_attempt_latency_seconds"),
    ]:
        report.append("| "+label+" | "+" | ".join(f"{m[key]:.2f}" for m in metrics)+" |")
    report.append("| Summed example latencies (hours) | "+" | ".join(f"{m['sum_recorded_example_latency_seconds']/3600:.2f}" for m in metrics)+" |")
    report.append(f"| Recorded evaluation wall time (minutes) | Not recorded | Not recorded | {new['wall_time_seconds']/60:.2f} |")
    report.append(f"| Output tokens per wall second | Not recorded | Not recorded | {new['completion_tokens_per_wall_second']:.2f} |")
    report.append("| Attempts with exactly 4,096 output tokens | "+" | ".join(str(m['attempts_with_4096_completion_tokens']) for m in metrics)+" |")
    report.append("| Confirmed truncated attempts | "+" | ".join("Not recorded" if m['confirmed_truncated_attempts'] is None else str(m['confirmed_truncated_attempts']) for m in metrics)+" |")
    report += ["", "Latency averages exclude failed examples. The checkpoint run used 16 concurrent workers, "
               "so summed request latencies overlap and are not elapsed wall time. Historical wall time is unavailable; "
               "these rows should not be treated as a controlled speed comparison. Reaching exactly 4,096 output tokens "
               "does not by itself confirm truncation: baseline files did not save finish reasons.", "",
               "## Common successful subset", "",
               f"All three models have saved successful results for **{len(common):,} common examples**. This table "
               "excludes the same five examples from every model, removing failure-coverage differences from the denominator.", "",
               "| Model | Common examples | Pass@1 | Pass@2 |", "| --- | ---: | ---: | ---: |"]
    for m in metrics:
        report.append(f"| {m['model']} | {m['common_successful_examples']:,} | "
                      f"{m['accuracy_pass1_common_subset']:.2%} | {m['accuracy_pass2_common_subset']:.2%} |")
    report += ["", "Baseline failures occurred at indices `961` through `965`. The Base run rejected inputs longer "
               "than its 8,192-token context; the SFT run reported image-processor failures. Checkpoint-2100 succeeded "
               "on those examples. The headline accuracy table includes all 2,000 examples and counts failures as incorrect.", "",
               "## Sources and reproduction", "",
               "- [Base predictions](pmc_vqa_base_vllm_results.json).",
               "- [SFT predictions](pmc_vqa_sft_vllm_results.json).",
               "- [Checkpoint-2100 predictions](grpo_checkpoint_2100/results.json) and [summary](grpo_checkpoint_2100/results_summary.json).",
               "- [Checkpoint evaluation report](grpo_checkpoint_2100/REPORT.md) and [protocol](grpo_checkpoint_2100/README.md).",
               "- [CSV metrics](combined_metrics.csv) and [JSON metrics with definitions](combined_metrics.json).", "",
               "Regenerate from the project root with:", "", "```bash",
               "python outputs/benchmarks/pmc_vqa/build_combined_report.py", "```", "",
               "No inference is rerun. Original prediction files are read without modification."]
    (ROOT / "COMBINED_REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"aggregate":aggregate,"models":[{k:m[k] for k in ["model","planned_inference_prompts","recorded_successful_prompts","accuracy_pass1","accuracy_pass2","prompt_tokens"]} for m in metrics]},indent=2))
    print("Report:", ROOT / "COMBINED_REPORT.md")


if __name__ == "__main__":
    main()
