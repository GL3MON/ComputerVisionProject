"""Shared checkpoint and processor loading."""

from medical_vlm.core.paths import resolve_path


def model_source(source):
    candidate = resolve_path(source)
    return str(candidate) if candidate.exists() else str(source)


def load_vision_model(model_name, adapter=None, attention="sdpa", device="auto"):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    base = model_source(model_name)
    processor_source = model_source(adapter) if adapter else base
    processor = AutoProcessor.from_pretrained(processor_source)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=dtype, device_map=device, attn_implementation=attention
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_source(adapter)).merge_and_unload(
            safe_merge=True
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor
