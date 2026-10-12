"""Inspect a local adapter without starting inference on import."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a LoRA checkpoint and generate from a question, "
        "printing the raw output including special tokens."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter dir (e.g. ./outputs/qwen3-vl-2b-medical-lora).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model to load. Defaults to whatever the "
        "adapter's adapter_config.json says it was trained from.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask. If omitted, you'll be prompted interactively.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    question = args.question
    if question is None:
        question = input("Question: ")

    # the adapter config records which base model it was trained from,
    # so we don't need to ask the user to repeat it unless they want to
    # override it
    base_model_name = args.base_model
    if base_model_name is None:
        peft_config = PeftConfig.from_pretrained(args.checkpoint)
        base_model_name = peft_config.base_model_name_or_path

    print(f"Loading base model: {base_model_name}")

    # fix: flash_attention_2 raises ImportError if flash-attn isn't installed
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        print("flash-attn not found, falling back to sdpa attention")
        attn_impl = "sdpa"

    processor = AutoProcessor.from_pretrained(args.checkpoint)

    if processor.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    base_model.config.pad_token_id = processor.pad_token_id

    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()
    model.config.use_cache = True

    prompt_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]

    # skip_special_tokens=False so you can see <think>/<|im_end|>/etc. exactly
    # as the model produced them
    generated_text = processor.decode(generated_ids, skip_special_tokens=False)

    # also show the full sequence (prompt + generation) with special tokens,
    # useful for sanity-checking the chat template formatting itself
    full_text = processor.decode(output_ids[0], skip_special_tokens=False)

    print("\n" + "=" * 80)
    print("PROMPT (as sent to the model, with special tokens):")
    print("=" * 80)
    print(prompt_text)

    print("\n" + "=" * 80)
    print("GENERATION ONLY (with special tokens):")
    print("=" * 80)
    print(generated_text)

    print("\n" + "=" * 80)
    print("FULL SEQUENCE (prompt + generation, with special tokens):")
    print("=" * 80)
    print(full_text)


if __name__ == "__main__":
    main()
