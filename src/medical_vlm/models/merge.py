"""Merge every LoRA target, including vision targets, into a local export."""

import argparse
from pathlib import Path
from medical_vlm.core.paths import output_root, resolve_path
from medical_vlm.core.io import write_json
from medical_vlm.models.loading import model_source


def merge_checkpoint(adapter, output, base_model=None, dtype="bfloat16", device="cpu"):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    adapter = model_source(adapter)
    output = resolve_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model export: {output}")
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model = base_model or peft_config.base_model_name_or_path
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_source(base_model),
        dtype=getattr(torch, dtype),
        device_map=device,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Unmerged LoRA parameters remain")
    try:
        processor = AutoProcessor.from_pretrained(adapter)
    except OSError:
        processor = AutoProcessor.from_pretrained(model_source(base_model))
    model.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    write_json(
        output.parent / "merge_metadata.json",
        {
            "adapter": adapter,
            "base_model": base_model,
            "output": str(output),
            "dtype": dtype,
            "all_lora_modules_merged": True,
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo", help="Optional explicit Hugging Face upload after merging")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    output = args.output or output_root() / "models" / Path(args.adapter).name / "merged"
    output = merge_checkpoint(args.adapter, output, args.base_model, args.dtype, args.device)
    print(f"Merged model: {output}")
    if args.repo:
        from medical_vlm.models.hub import upload_model

        print(upload_model(output, args.repo, args.private))


if __name__ == "__main__":
    main()
