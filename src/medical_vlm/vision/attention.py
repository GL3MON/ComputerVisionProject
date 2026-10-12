"""Capture actual received self-attention mass from the Qwen3-VL visual tower."""

import argparse
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.core.io import write_json


def attention_for_image(model, processor, image):
    import torch
    from unittest.mock import patch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation
    from medical_vlm.vision.features import restore_patch_grid

    inputs = processor.image_processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    grid = inputs["image_grid_thw"][0].tolist()
    if int(grid[0]) != 1:
        raise ValueError("Attention capture supports one static-image temporal grid")
    captured = []
    original = implementation.eager_attention_forward

    def capture(*args, **kwargs):
        output, weights = original(*args, **kwargs)
        captured.append(weights.detach().float().mean(dim=tuple(range(weights.ndim - 1))).cpu())
        return output, weights

    with patch.object(implementation, "eager_attention_forward", capture), torch.inference_mode():
        model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    expected = int(grid[1]) * int(grid[2])
    if not captured or any(values.numel() != expected for values in captured):
        raise RuntimeError(
            "Attention shapes do not match the patch grid; no truncation or shape guessing is performed"
        )
    received = torch.stack(captured).mean(dim=0)
    return restore_patch_grid(received, grid, processor.image_processor.merge_size).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/vision/attention.yaml"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.samples is not None:
        config["samples"] = args.samples
    if args.output:
        config["output_dir"] = args.output
    if config.get("samples", 100) < 1:
        parser.error("samples must be positive")
    import numpy as np
    from medical_vlm.models.loading import load_vision_model
    from medical_vlm.data.medmnist import MedMNISTDataset
    from medical_vlm.vision.maps import save_overlay

    model, processor = load_vision_model(config["model"], config.get("adapter"), attention="eager")
    dataset = MedMNISTDataset(
        config.get("dataset", "pneumoniamnist"), "val", config.get("image_size", 224)
    )
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    count = min(config.get("samples", 100), len(dataset))
    for index in range(count):
        image, label = dataset[index]
        values = attention_for_image(model, processor, image)
        np.save(output / f"attention_{index:05d}_raw.npy", values)
        save_overlay(
            image,
            values,
            output / f"attention_{index:05d}_label_{label}.png",
            f"Mean received visual self-attention: {index}",
        )
    write_json(
        output / "summary.json",
        {
            "samples": count,
            "config": config,
            "method": "Mean key-side self-attention mass across visual layers, heads, and queries; no CLS-token assumption or attention rollout.",
        },
    )


if __name__ == "__main__":
    main()
