"""Shared linear probe architecture and evaluation."""

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


def evaluate_classifier(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    losses, total, predictions, targets = 0.0, 0, [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            losses += nn.functional.cross_entropy(logits, labels).item() * len(labels)
            total += len(labels)
            predictions.extend(logits.argmax(-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if not total:
        raise ValueError("Cannot evaluate an empty feature split")
    return (
        {
            "loss": losses / total,
            "accuracy": float(accuracy_score(targets, predictions)),
            "weighted_f1": float(
                f1_score(targets, predictions, average="weighted", zero_division=0)
            ),
        },
        targets,
        predictions,
    )
