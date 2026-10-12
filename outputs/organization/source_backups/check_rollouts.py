import argparse
import json

import requests
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="FreedomIntelligence/medical-o1-verifiable-problem",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="rollouts.jsonl",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1/completions",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def generate_group(
    url,
    model,
    prompt,
    num_generations,
    max_tokens,
    temperature,
    top_p,
):
    prompts = [prompt] * num_generations

    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=600,
    )

    response.raise_for_status()

    data = response.json()

    choices = sorted(
        data["choices"],
        key=lambda x: x["index"],
    )

    completions = [
        choice["text"]
        for choice in choices
    ]

    if len(completions) != num_generations:
        raise RuntimeError(
            f"Expected {num_generations} completions, "
            f"got {len(completions)}"
        )

    return completions


def main():
    args = parse_args()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    print(f"Loaded {len(dataset)} examples")

    with open(args.output, "w") as f:
        for i, example in enumerate(dataset):
            question = example["Open-ended Verifiable Question"]
            ground_truth = example["Ground-True Answer"]

            completions = generate_group(
                url=args.url,
                model=args.model,
                prompt=question,
                num_generations=args.num_generations,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "prompt": question,
                "ground_truth": ground_truth,
                "completions": completions,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    indent=4,
                )
                + "\n"
            )

            if (i + 1) % 10 == 0:
                print(
                    f"Generated "
                    f"{i + 1}/{len(dataset)} groups"
                )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()