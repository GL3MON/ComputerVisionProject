"""Merge both language and vision adapters for local benchmark inference."""
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

checkpoint = Path("/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100")
output = Path(__file__).resolve().parent / "merged"
assert not output.exists(), f"Refusing to overwrite {output}"
config = json.loads((checkpoint / "adapter_config.json").read_text())
base = config["base_model_name_or_path"]
print(f"Loading base: {base}", flush=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base, dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    attn_implementation="sdpa",
)
print("Loading vision and language LoRA adapters", flush=True)
model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
print("Merging all adapters", flush=True)
model = model.merge_and_unload(safe_merge=True)
assert not any("lora_" in name for name, _ in model.named_parameters())
model.save_pretrained(output, safe_serialization=True)
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
processor.save_pretrained(output)
metadata = {"checkpoint": str(checkpoint), "base_model": base, "merged_model": str(output),
            "adapter_target_modules": config["target_modules"], "dtype": "bfloat16",
            "all_lora_modules_merged": True}
(Path(__file__).resolve().parent / "merge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Saved merged checkpoint: {output}", flush=True)
