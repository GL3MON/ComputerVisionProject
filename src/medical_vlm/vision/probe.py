"""Frozen vision feature extraction and supervised linear probing."""

import argparse
from medical_vlm.core.config import load_config
from medical_vlm.core.paths import project_root, resolve_path
from medical_vlm.core.io import write_json


def extract_split(model, processor, dataset, batch_size, device):
    import torch
    from torch.utils.data import DataLoader
    from medical_vlm.vision.features import image_embeddings

    def collate(batch):
        images, labels = zip(*batch)
        return processor.image_processor(images=list(images), return_tensors="pt"), torch.tensor(
            labels
        )

    features, labels = [], []
    for inputs, batch_labels in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    ):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            features.append(image_embeddings(model, inputs).float().cpu())
        labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def train_probe(config):
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from medical_vlm.data.medmnist import MedMNISTDataset
    from medical_vlm.models.loading import load_vision_model
    from medical_vlm.vision.classifier import LinearClassifier, evaluate_classifier

    seed = config.get("seed", 3407)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = resolve_path(config["output_dir"])
    embeddings = output / "embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    model, processor = load_vision_model(config["model"], config.get("adapter"))
    model_device = next(model.parameters()).device
    splits = {}
    for split in ["train", "val", "test"]:
        dataset = MedMNISTDataset(
            config.get("dataset", "pneumoniamnist"), split, config.get("image_size", 224)
        )
        limit = config.get("limit")
        if limit:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        x, y = extract_split(model, processor, dataset, config.get("batch_size", 16), model_device)
        torch.save(x, embeddings / f"{split}_features.pt")
        torch.save(y, embeddings / f"{split}_labels.pt")
        splits[split] = TensorDataset(x, y)
    input_dim = splits["train"].tensors[0].shape[1]
    classes = (
        len(dataset.dataset.info["label"])
        if isinstance(dataset, Subset)
        else len(dataset.info["label"])
    )
    classifier = LinearClassifier(input_dim, classes).to(device)
    loaders = {
        name: DataLoader(
            data, batch_size=config.get("classifier_batch_size", 256), shuffle=name == "train"
        )
        for name, data in splits.items()
    }
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
    )
    history = []
    best = -1
    best_epoch = 0
    for epoch in range(1, config.get("epochs", 20) + 1):
        classifier.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(classifier(x), y)
            loss.backward()
            optimizer.step()
        metrics, _, _ = evaluate_classifier(classifier, loaders["val"], device)
        history.append({"epoch": epoch, **metrics})
        if metrics["accuracy"] > best:
            best, best_epoch = metrics["accuracy"], epoch
            torch.save(
                {
                    "model_state_dict": classifier.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": classes,
                    "model_name": config["model"],
                    "dataset_name": config.get("dataset", "pneumoniamnist"),
                    "image_size": config.get("image_size", 224),
                    "best_val_accuracy": best,
                    "best_epoch": epoch,
                },
                output / "linear_classifier_best.pt",
            )
        print(f"Epoch {epoch}: validation accuracy {metrics['accuracy']:.2%}")
    checkpoint = torch.load(
        output / "linear_classifier_best.pt", map_location=device, weights_only=False
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    metrics, targets, predictions = evaluate_classifier(classifier, loaders["test"], device)
    torch.save(
        {
            **checkpoint,
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["weighted_f1"],
            "history": history,
        },
        output / "linear_classifier_final.pt",
    )
    np.savez(output / "test_predictions.npz", targets=targets, predictions=predictions)
    write_json(
        output / "metrics.json",
        {
            "best_epoch": best_epoch,
            "validation_accuracy": best,
            "test": metrics,
            "history": history,
            "config": config,
        },
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(project_root() / "configs/vision/probe.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["model", "adapter", "limit"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output:
        config["output_dir"] = args.output
    if config.get("epochs", 20) < 1 or (config.get("limit") is not None and config["limit"] < 1):
        parser.error("Epochs and limit must be positive")
    print(train_probe(config))


if __name__ == "__main__":
    main()
