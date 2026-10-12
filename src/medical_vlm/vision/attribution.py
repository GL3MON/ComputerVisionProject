"""Predicted-class gradient-times-input attribution for a frozen linear probe."""

import argparse
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.core.io import write_json


def attribution_for_image(model, processor, classifier, image):
    import torch
    from medical_vlm.vision.features import image_embeddings, restore_patch_grid

    device = next(model.parameters()).device
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pixels = inputs["pixel_values"].detach().clone().requires_grad_(True)
    inputs["pixel_values"] = pixels
    embedding = image_embeddings(model, inputs).float().to(next(classifier.parameters()).device)
    logits = classifier(embedding)
    predicted = int(logits.argmax(-1).item())
    gradient = torch.autograd.grad(logits[0, predicted], pixels)[0]
    packed = (gradient * pixels).abs().mean(dim=1)
    merge = processor.image_processor.merge_size
    grid = restore_patch_grid(packed, inputs["image_grid_thw"][0].tolist(), merge)
    return (
        grid.detach().float().cpu().numpy(),
        predicted,
        float(logits.softmax(-1)[0, predicted].item()),
    )


def load_attribution_components(config):
    import torch
    from medical_vlm.models.loading import load_vision_model
    from medical_vlm.vision.classifier import LinearClassifier

    model, processor = load_vision_model(config["model"], config.get("adapter"))
    checkpoint = torch.load(
        resolve_path(config["classifier"]), map_location="cpu", weights_only=False
    )
    classifier = LinearClassifier(checkpoint["input_dim"], checkpoint["num_classes"])
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(next(model.parameters()).device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return model, processor, classifier


def run_attribution(config):
    import csv
    import random
    import numpy as np
    from medical_vlm.data.medmnist import MedMNISTDataset
    from medical_vlm.vision.maps import normalize_map, summarize_map, save_overlay

    model, processor, classifier = load_attribution_components(config)
    dataset = MedMNISTDataset(
        config.get("dataset", "pneumoniamnist"), "val", config.get("image_size", 224)
    )
    indices = random.Random(config.get("seed", 3407)).sample(
        range(len(dataset)), min(config.get("samples", 25), len(dataset))
    )
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        image, label = dataset[index]
        raw, prediction, confidence = attribution_for_image(model, processor, classifier, image)
        values = normalize_map(raw)
        record = {
            "index": index,
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "confidence": confidence,
            **summarize_map(values),
        }
        records.append(record)
        np.save(output / f"sample_{index:04d}_raw.npy", raw)
        save_overlay(
            image,
            values,
            output / f"sample_{index:04d}.png",
            f"Sample {index}: label={label}, prediction={prediction}",
        )
    with (output / "attribution_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "samples": len(records),
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "config": config,
        "method": "Predicted-class absolute gradient times input; packed patches restored to raster order; top-fraction mass normalized to total mass.",
    }
    for key in ["confidence", "attribution_entropy", "top_10_percent_mass", "center_x", "center_y"]:
        summary[key] = float(np.mean([r[key] for r in records]))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/vision/attribution.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--classifier")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["model", "adapter", "classifier", "samples"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output:
        config["output_dir"] = args.output
    if config.get("samples", 25) < 1:
        parser.error("samples must be positive")
    print(run_attribution(config))


if __name__ == "__main__":
    main()
