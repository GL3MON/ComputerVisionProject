"""Normalized attribution summaries with defined edge-case behavior."""

import numpy as np


def normalize_map(values):
    values = np.asarray(values, dtype=float)
    values = values - values.min()
    return values / values.max() if values.max() > 1e-12 else np.zeros_like(values)


def summarize_map(values, fraction=0.1):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    values = np.maximum(np.asarray(values, dtype=float), 0)
    total = values.sum()
    probability = values / (total + 1e-12)
    entropy = (
        -np.sum(probability * np.log(probability + 1e-12)) / np.log(values.size)
        if values.size > 1 and total > 0
        else 0.0
    )
    n = max(1, int(values.size * fraction))
    mass = np.sort(values.ravel())[-n:].sum() / (total + 1e-12)
    height, width = values.shape
    ys, xs = np.mgrid[:height, :width]
    return {
        "attribution_entropy": float(entropy),
        "top_10_percent_mass": float(mass),
        "center_x": float((xs * probability).sum() / max(1, width - 1)),
        "center_y": float((ys * probability).sum() / max(1, height - 1)),
    }


def save_overlay(image, values, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    overlay = np.asarray(
        Image.fromarray(normalize_map(values).astype("float32"), mode="F").resize(
            image.size, Image.Resampling.BILINEAR
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(overlay, cmap="jet", alpha=0.5)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
