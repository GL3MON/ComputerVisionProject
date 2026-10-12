"""Pure aggregation of saved attempt records; no inference or report writes."""

import json
from pathlib import Path
from statistics import median


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


def collect(label, filename, rows, summary_file=None):
    assert rows, "Results cannot be empty"
    assert sorted(r["index"] for r in rows) == list(range(len(rows)))
    attempts = [a for r in rows for a in r["attempts"]]
    successful = [r for r in rows if "error" not in r]
    failures = [r for r in rows if "error" in r]
    attempts_per_question = max((len(r["attempts"]) for r in successful), default=2)
    assert all(len(r["attempts"]) == attempts_per_question for r in successful)
    assert all(not r["attempts"] for r in failures)
    for row in successful:
        correctness = [
            a["normalized_prediction"] == row["normalized_ground_truth"] for a in row["attempts"]
        ]
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
        "model": label,
        "source_file": filename,
        "question_examples": len(rows),
        "attempts_planned_per_question": attempts_per_question,
        "planned_inference_prompts": len(rows) * attempts_per_question,
        "recorded_successful_prompts": len(attempts),
        "unretained_planned_attempts": len(rows) * attempts_per_question - len(attempts),
        "actual_network_request_count": None,
        "successful_question_examples": len(successful),
        "failed_question_examples": len(failures),
        "failed_example_indices": [r["index"] for r in failures],
        "correct_pass1_count": sum(r["correct_pass1"] for r in rows),
        "correct_pass2_count": sum(r["correct_passk"] for r in rows),
        "accuracy_pass1": percent(sum(r["correct_pass1"] for r in rows), len(rows)),
        "accuracy_pass2": percent(sum(r["correct_passk"] for r in rows), len(rows)),
        "accuracy_pass1_successful_only": percent(
            sum(r["correct_pass1"] for r in successful), len(successful)
        ),
        "accuracy_pass2_successful_only": percent(
            sum(r["correct_passk"] for r in successful), len(successful)
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "mean_prompt_tokens_per_recorded_prompt": percent(prompt_tokens, len(attempts)),
        "mean_completion_tokens_per_recorded_prompt": percent(completion_tokens, len(attempts)),
        "mean_example_latency_seconds": percent(sum(latencies), len(latencies)),
        "median_example_latency_seconds": median(latencies) if latencies else None,
        "p95_example_latency_seconds": quantile(latencies, 0.95),
        "mean_attempt_latency_seconds": percent(sum(attempt_latencies), len(attempt_latencies)),
        "sum_recorded_example_latency_seconds": sum(latencies),
        "wall_time_seconds": None,
        "completion_tokens_per_wall_second": None,
        "attempts_with_4096_completion_tokens": sum(
            a.get("completion_tokens") == 4096 for a in attempts
        ),
        "finish_reasons_recorded": finish_reasons_recorded,
        "confirmed_truncated_attempts": sum(a.get("finish_reason") == "length" for a in attempts)
        if finish_reasons_recorded
        else None,
    }
    if summary_file is not None:
        summary = json.loads(Path(summary_file).read_text())
        assert summary["complete"] and summary["evaluated_examples"] == len(rows)
        assert summary["accuracy_pass1"] == result["accuracy_pass1"]
        assert summary["accuracy_passk"] == result["accuracy_pass2"]
        assert summary["prompt_tokens"] == prompt_tokens
        assert summary["completion_tokens"] == completion_tokens
        assert summary["truncated_attempts"] == result["confirmed_truncated_attempts"]
        result["wall_time_seconds"] = summary["wall_time_seconds"]
        result["completion_tokens_per_wall_second"] = summary["completion_tokens_per_second"]
    return result
