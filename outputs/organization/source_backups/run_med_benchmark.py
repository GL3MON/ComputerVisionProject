import base64
import io
import json
import os
import re
import time

from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from tqdm.auto import tqdm


VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

OUTPUT_FILE = "pmc_vqa_base_vllm_results.json"

MAX_COMPLETION_TOKENS = 4096
CHECKPOINT_EVERY = 50
PASS_AT_K = 2


client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)


models = client.models.list()
MODEL_NAME = models.data[0].id

print("Using model:", MODEL_NAME)


dataset = load_dataset("OctoMed/PMC-VQA")

test_dataset = dataset["test"]

print("Test examples:", len(test_dataset))

def image_to_base64(image):
    buffer = io.BytesIO()

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def format_question(sample):
    options = "\n".join(sample["options"])

    return f"""{sample["question"]}

{options}

Select the correct answer.

Analyze the medical image carefully and determine the correct answer and put your final answer inside <answer></answer>."""

def extract_answer(response):
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return response.strip()

def normalize_answer(answer):
    answer = str(answer).strip()

    match = re.match(
        r"^([A-E])(?:\s*[:.)-]|\s+|$)",
        answer,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return answer.lower()

def run_inference(sample, temperature=0.7):

    image_b64 = image_to_base64(sample["image"])
    prompt = format_question(sample)

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=temperature,
        top_p=0.8,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
        },
    )

    latency = time.perf_counter() - start

    message = response.choices[0].message

    reasoning = getattr(
        message,
        "reasoning",
        None,
    )

    final_answer = message.content or ""

    if not final_answer and reasoning:
        final_answer = reasoning

    return {
        "reasoning": reasoning,
        "response": final_answer,
        "latency": latency,
        "completion_tokens": getattr(
            response.usage,
            "completion_tokens",
            None,
        ),
        "prompt_tokens": getattr(
            response.usage,
            "prompt_tokens",
            None,
        ),
    }


def run_inference_pass_at_k(sample, k=PASS_AT_K):
    attempts = []
    for attempt in range(k):
        temp = 0.6 if attempt == 0 else min(0.6 + attempt * 0.15, 1.0)
        result = run_inference(sample, temperature=temp)
        predicted = extract_answer(result["response"])
        normalized = normalize_answer(predicted)
        result["predicted"] = predicted
        result["normalized"] = normalized
        attempts.append(result)
    return attempts

results = []

correct_pass1 = 0
correct_passk = 0
failed = 0

total_start = time.perf_counter()

for i, sample in enumerate(
    tqdm(test_dataset, total=len(test_dataset))
):

    try:
        attempts = run_inference_pass_at_k(sample, k=PASS_AT_K)

        normalized_ground_truth = normalize_answer(sample["answer"])

        is_correct_per_attempt = [
            a["normalized"] == normalized_ground_truth for a in attempts
        ]

        any_correct = any(is_correct_per_attempt)

        if is_correct_per_attempt[0]:
            correct_pass1 += 1
        if any_correct:
            correct_passk += 1

        results.append({
            "index": i,
            "question": sample["question"],
            "options": sample["options"],
            "ground_truth": sample["answer"],
            "normalized_ground_truth": normalized_ground_truth,
            "attempts": [
                {
                    "attempt": j,
                    "reasoning": a["reasoning"],
                    "response": a["response"],
                    "predicted_answer": a["predicted"],
                    "normalized_prediction": a["normalized"],
                    "correct": is_correct_per_attempt[j],
                    "latency": a["latency"],
                    "prompt_tokens": a["prompt_tokens"],
                    "completion_tokens": a["completion_tokens"],
                }
                for j, a in enumerate(attempts)
            ],
            "correct_pass1": is_correct_per_attempt[0],
            "correct_passk": any_correct,
            "latency": sum(a["latency"] for a in attempts),
            "prompt_tokens": sum(a["prompt_tokens"] or 0 for a in attempts),
            "completion_tokens": sum(a["completion_tokens"] or 0 for a in attempts),
        })

    except Exception as e:

        failed += 1

        results.append({
            "index": i,
            "question": sample["question"],
            "options": sample["options"],
            "ground_truth": sample["answer"],
            "normalized_ground_truth": normalize_answer(sample["answer"]),
            "attempts": [],
            "correct_pass1": False,
            "correct_passk": False,
            "latency": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": str(e),
        })

    if (i + 1) % CHECKPOINT_EVERY == 0:

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        current_accuracy_pass1 = correct_pass1 / (i + 1)
        current_accuracy_passk = correct_passk / (i + 1)

        print(f"\nProcessed: {i + 1}/{len(test_dataset)}")
        print(f"Pass@1 Accuracy: {current_accuracy_pass1 * 100:.2f}%")
        print(f"Pass@{PASS_AT_K} Accuracy: {current_accuracy_passk * 100:.2f}%")
        print(f"Failed: {failed}")




total_time = time.perf_counter() - total_start

valid_results = [
    r for r in results
    if r["latency"] is not None
]

accuracy_pass1 = sum(r["correct_pass1"] for r in results) / len(results)
accuracy_passk = sum(r["correct_passk"] for r in results) / len(results)

avg_latency = (
    sum(r["latency"] for r in valid_results)
    / len(valid_results)
)

total_output_tokens = sum(
    r["completion_tokens"] or 0
    for r in valid_results
)

total_prompt_tokens = sum(
    r["prompt_tokens"] or 0
    for r in valid_results
)

print("=" * 60)
print("PMC-VQA SFT MODEL BENCHMARK")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Examples: {len(results)}")
print(f"Failed: {failed}")
print(f"Pass@1 Accuracy: {accuracy_pass1 * 100:.2f}%")
print(f"Pass@{PASS_AT_K} Accuracy: {accuracy_passk * 100:.2f}%")

print(f"Average latency: {avg_latency:.3f} sec")

print(f"Total time: {total_time / 60:.2f} min")

print(f"Prompt tokens: {total_prompt_tokens:,}")

print(f"Completion tokens: {total_output_tokens:,}")

if total_output_tokens > 0:
    print(f"Output tokens/sec: {total_output_tokens / total_time:.2f}")


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Saved to {OUTPUT_FILE}")