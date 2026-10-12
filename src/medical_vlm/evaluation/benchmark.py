"""Resumable, bounded-concurrency PMC-VQA evaluation."""

import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from tqdm.auto import tqdm
from medical_vlm.core.paths import output_root, resolve_path
from medical_vlm.core.io import write_json
from medical_vlm.evaluation.client import ChatInference, InferenceSettings, create_client
from medical_vlm.evaluation.scoring import normalize_answer

CHECKPOINT_EVERY = 50


def evaluate_sample(item, service, k=2):
    i, sample = item
    normalized_ground_truth = normalize_answer(sample["answer"])
    record = {
        "index": i,
        "id": sample.get("id"),
        "question": sample["question"],
        "options": sample["options"],
        "ground_truth": sample["answer"],
        "normalized_ground_truth": normalized_ground_truth,
    }
    try:
        attempts = service.generate_n(sample, k=k)
        correctness = [a["normalized"] == normalized_ground_truth for a in attempts]
        record.update(
            {
                "attempts": [
                    {
                        "attempt": j,
                        "reasoning": a["reasoning"],
                        "response": a["response"],
                        "predicted_answer": a["predicted"],
                        "normalized_prediction": a["normalized"],
                        "correct": correctness[j],
                        "latency": a["latency"],
                        "finish_reason": a.get("finish_reason"),
                        "prompt_tokens": a["prompt_tokens"],
                        "completion_tokens": a["completion_tokens"],
                    }
                    for j, a in enumerate(attempts)
                ],
                "correct_pass1": correctness[0],
                "correct_passk": any(correctness),
                "latency": sum(a["latency"] for a in attempts),
                "prompt_tokens": sum(a["prompt_tokens"] or 0 for a in attempts),
                "completion_tokens": sum(a["completion_tokens"] or 0 for a in attempts),
            }
        )
    except Exception as error:
        record.update(
            {
                "attempts": [],
                "correct_pass1": False,
                "correct_passk": False,
                "latency": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "error": str(error),
            }
        )
    return record


def summarize(results, metadata, elapsed, complete):
    valid = [r for r in results if r["latency"] is not None]
    count = len(results)
    tokens = sum(r["completion_tokens"] or 0 for r in valid)
    return {
        **metadata,
        "complete": complete,
        "evaluated_examples": count,
        "failed_examples": count - len(valid),
        "accuracy_pass1": sum(r["correct_pass1"] for r in results) / count if count else None,
        "accuracy_passk": sum(r["correct_passk"] for r in results) / count if count else None,
        "average_example_latency_seconds": sum(r["latency"] for r in valid) / len(valid)
        if valid
        else None,
        "completion_tokens": tokens,
        "truncated_attempts": sum(
            a.get("finish_reason") == "length" for r in results for a in r["attempts"]
        ),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in valid),
        "wall_time_seconds": elapsed,
        "completion_tokens_per_second": tokens / elapsed if elapsed else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    VLLM_BASE_URL, VLLM_API_KEY = "http://localhost:8000/v1", "EMPTY"
    PASS_AT_K, MAX_COMPLETION_TOKENS = 2, 4096
    parser = argparse.ArgumentParser(
        description="Evaluate a served model on PMC-VQA without overwriting previous runs."
    )
    parser.add_argument("--base-url", default=VLLM_BASE_URL)
    parser.add_argument(
        "--model", help="Served model ID; required if multiple models are available"
    )
    parser.add_argument(
        "--output", type=Path, help="Result JSON path (must be new unless --resume is supplied)"
    )
    parser.add_argument("--checkpoint", help="Checkpoint path recorded as provenance")
    parser.add_argument("--pass-at-k", type=int, default=PASS_AT_K)
    parser.add_argument("--max-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, help="Evaluate a prefix for a smoke test; omit for the full test split"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached Hugging Face data only")
    args = parser.parse_args()
    if min(args.pass_at_k, args.max_tokens, args.workers) < 1 or (
        args.limit is not None and args.limit < 1
    ):
        parser.error("pass-at-k, max-tokens, workers and limit must be positive")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    client = create_client(args.base_url, os.environ.get("VLLM_API_KEY", VLLM_API_KEY))
    models = [m.id for m in client.models.list().data]
    if args.model:
        if args.model not in models:
            parser.error(f"Model {args.model!r} is not served; available: {models}")
        MODEL_NAME = args.model
    elif len(models) == 1:
        MODEL_NAME = models[0]
    else:
        parser.error(f"Specify --model; available: {models}")
    PASS_AT_K, MAX_COMPLETION_TOKENS = args.pass_at_k, args.max_tokens
    service = ChatInference(client, InferenceSettings(MODEL_NAME, MAX_COMPLETION_TOKENS))
    output = args.output or output_root() / "benchmarks/pmc_vqa" / (
        re.sub(r"[^A-Za-z0-9_.-]", "_", MODEL_NAME)
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
    output = resolve_path(output)
    summary_path = output.with_name(output.stem + "_summary.json")
    if not args.resume and (output.exists() or summary_path.exists()):
        parser.error(f"Results already exist: {output}; choose a new output or use --resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Import after offline flags are set so datasets reads the requested configuration.
    from datasets import load_dataset

    test_dataset = load_dataset("OctoMed/PMC-VQA")["test"]
    total = min(args.limit, len(test_dataset)) if args.limit else len(test_dataset)
    metadata = {
        "model": MODEL_NAME,
        "checkpoint": args.checkpoint,
        "base_url": args.base_url,
        "dataset": "OctoMed/PMC-VQA",
        "split": "test",
        "dataset_fingerprint": test_dataset._fingerprint,
        "test_split_size": len(test_dataset),
        "requested_examples": total,
        "pass_at_k": PASS_AT_K,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperatures": [0.6 if j == 0 else min(0.6 + j * 0.15, 1.0) for j in range(PASS_AT_K)],
        "top_p": 0.8,
        "top_k": 20,
        "enable_thinking": True,
        "workers": args.workers,
        "scoring": "Exact normalized option letter; failures count as incorrect; pass@k is any correct attempt",
        "output_file": str(output),
    }
    results, previous_elapsed = [], 0.0
    if args.resume:
        if not output.exists() or not summary_path.exists():
            parser.error("Resume requires both the result JSON and its summary")
        previous = json.loads(summary_path.read_text())
        for key in [
            "model",
            "checkpoint",
            "dataset",
            "split",
            "dataset_fingerprint",
            "requested_examples",
            "pass_at_k",
            "max_completion_tokens",
            "temperatures",
            "top_p",
            "top_k",
            "enable_thinking",
        ]:
            if previous.get(key) != metadata[key]:
                parser.error(f"Resume configuration differs for {key}")
        results = json.loads(output.read_text())
        indices = [r["index"] for r in results]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= total for i in indices):
            parser.error("Existing results have duplicate or out-of-range indices")
        previous_elapsed = previous.get("wall_time_seconds", 0.0)
    completed = {r["index"] for r in results}
    items = ((i, test_dataset[i]) for i in range(total) if i not in completed)
    start = time.perf_counter()

    def save(complete=False):
        results.sort(key=lambda r: r["index"])
        write_json(output, results)
        summary = summarize(
            results, metadata, previous_elapsed + time.perf_counter() - start, complete
        )
        write_json(summary_path, summary)
        return summary

    print(
        f"Model: {MODEL_NAME}; examples: {total}; attempts: {PASS_AT_K}; workers: {args.workers}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)
    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            with tqdm(total=total - len(completed)) as progress:
                pending = set()

                def submit_next():
                    item = next(items, None)
                    if item is not None:
                        pending.add(pool.submit(evaluate_sample, item, service, PASS_AT_K))

                for _ in range(args.workers * 2):
                    submit_next()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        results.append(future.result())
                        progress.update(1)
                        submit_next()
                        if len(results) % CHECKPOINT_EVERY == 0:
                            summary = save()
                            print(
                                f"Processed {len(results)}/{total}: pass@1={summary['accuracy_pass1']:.2%}, "
                                f"pass@{PASS_AT_K}={summary['accuracy_passk']:.2%}, failed={summary['failed_examples']}",
                                flush=True,
                            )
    finally:
        summary = save(complete=len(results) == total)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed_examples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
