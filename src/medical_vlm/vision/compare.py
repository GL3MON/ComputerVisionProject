"""Matched-image attribution comparison using shared loading and metrics."""

import argparse
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.core.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/vision/comparison.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    import csv
    import random
    import numpy as np
    from PIL import Image
    from medical_vlm.data.medmnist import MedMNISTDataset
    from medical_vlm.vision.attribution import load_attribution_components, attribution_for_image
    from medical_vlm.vision.maps import normalize_map, save_overlay

    dataset = MedMNISTDataset(
        config.get("dataset", "pneumoniamnist"), "val", config.get("image_size", 224)
    )
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    indices = random.Random(config.get("seed", 3407)).sample(
        range(len(dataset)), min(config.get("samples", 25), len(dataset))
    )
    models = {
        name: load_attribution_components(values) for name, values in config["models"].items()
    }
    if len(models) != 2:
        raise ValueError("Comparison requires exactly two model/classifier configurations")
    names = list(models)
    records = []
    for index in indices:
        image, label = dataset[index]
        maps = {}
        record = {"index": index, "label": label}
        for name, components in models.items():
            values, prediction, confidence = attribution_for_image(*components, image)
            maps[name] = np.asarray(
                Image.fromarray(normalize_map(values).astype("float32"), mode="F").resize(
                    image.size
                )
            )
            np.save(output / f"{name}_{index:04d}_raw.npy", values)
            save_overlay(
                image,
                values,
                output / f"{name}_{index:04d}.png",
                f"{name}: label={label}, prediction={prediction}",
            )
            record.update(
                {
                    f"{name}_prediction": prediction,
                    f"{name}_correct": prediction == label,
                    f"{name}_confidence": confidence,
                }
            )
        difference = maps[names[1]] - maps[names[0]]
        np.save(output / f"difference_{index:04d}.npy", difference)
        record["mean_absolute_difference"] = float(np.abs(difference).mean())
        records.append(record)
    with (output / "attribution_differences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        output / "summary.json",
        {
            "models": names,
            "samples": len(records),
            "config": config,
            "map_comparison": "Each model uses its own processor; min-max-normalized maps resized to the original image size.",
        },
    )


if __name__ == "__main__":
    main()
