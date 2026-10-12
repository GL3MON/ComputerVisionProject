import argparse
import os

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Thinking",
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--private",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Destination: {args.repo}")

    processor = AutoProcessor.from_pretrained(
        args.base_model,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )

    print("Merging LoRA adapter...")

    model = model.merge_and_unload()

    model.eval()

    print("Saving merged model locally...")

    local_dir = os.path.join(
        args.adapter,
        "merged",
    )

    os.makedirs(
        local_dir,
        exist_ok=True,
    )

    model.save_pretrained(
        local_dir,
        safe_serialization=True,
    )

    processor.save_pretrained(
        local_dir,
    )

    print(
        f"Uploading merged model to "
        f"hf.co/{args.repo}"
    )

    model.push_to_hub(
        args.repo,
        private=args.private,
    )

    processor.push_to_hub(
        args.repo,
        private=args.private,
    )

    print("Done.")


if __name__ == "__main__":
    main()