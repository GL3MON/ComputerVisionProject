"""Small text or image inference checks against an existing server."""

import argparse
from medical_vlm.evaluation.client import create_client, ChatInference, InferenceSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--medical", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--question", default="What is 17 × 24? Think carefully and give the final answer."
    )
    args = parser.parse_args()
    client = create_client(args.base_url)
    models = [m.id for m in client.models.list().data]
    model = args.model or (models[0] if len(models) == 1 else None)
    if model not in models:
        parser.error(f"Specify --model from {models}")
    if args.medical:
        from datasets import load_dataset

        sample = load_dataset("OctoMed/PMC-VQA", split="test")[args.index]
        print("Question:", sample["question"], "Ground truth:", sample["answer"])
        print(ChatInference(client, InferenceSettings(model)).generate(sample))
    else:
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.question}],
            max_completion_tokens=512,
            temperature=0.6,
            top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
        )
        print(result.choices[0].message)
        print(result.usage)


if __name__ == "__main__":
    main()
