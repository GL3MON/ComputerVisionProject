import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import medmnist
import matplotlib.pyplot as plt

from PIL import Image
from medmnist import INFO
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


import json
from pathlib import Path
torch.set_num_threads(2)
CHECKPOINT_PATH = '/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100'

MODEL_NAME = "GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged"
CLASSIFIER_PATH = '/home/harishr/Med/experiments/grpo_checkpoint_2100/classifier/linear_classifier_best.pt'

DATASET_NAME = "pneumoniamnist"
IMAGE_SIZE = 224
NUM_SAMPLES = 25

OUTPUT_DIR = '/home/harishr/Med/experiments/grpo_checkpoint_2100/attribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 3407

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

print("Device:", device)
print("Dtype:", dtype)
print("Model:", MODEL_NAME)


class MedMNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = Image.fromarray(
            np.asarray(image)
        ).convert("RGB")

        label = int(
            np.asarray(label).squeeze()
        )

        return image, label


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, x):
        return self.classifier(x)


dataset_info = INFO[DATASET_NAME]

DatasetClass = getattr(
    medmnist,
    dataset_info["python_class"],
)

val_dataset = MedMNISTWrapper(
    DatasetClass(
        split="val",
        size=IMAGE_SIZE,
        download=True,
        as_rgb=True,
    )
)


processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, local_files_only=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
)
from peft import PeftModel
from safetensors.torch import save_file
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=False).merge_and_unload()
model = model.to(device)
print("Loaded and merged checkpoint:", CHECKPOINT_PATH, flush=True)


model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)


checkpoint = torch.load(
    CLASSIFIER_PATH,
    map_location="cpu",
)

classifier = LinearClassifier(
    checkpoint["input_dim"],
    checkpoint["num_classes"],
)

classifier.load_state_dict(
    checkpoint["model_state_dict"]
)

classifier = classifier.to(device)
classifier.eval()

for parameter in classifier.parameters():
    parameter.requires_grad_(False)

print(
    "Classifier input dimension:",
    checkpoint["input_dim"],
)

print(
    "Best validation accuracy:",
    checkpoint.get(
        "best_val_accuracy",
        "N/A",
    ),
)


def get_embedding(model, inputs):
    features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )

    feature = features.pooler_output[0]

    if feature.ndim == 2:
        return feature.mean(dim=0)

    if feature.ndim == 3:
        return feature[0].mean(dim=0)

    raise ValueError(
        f"Unexpected feature shape: {feature.shape}"
    )


def get_attribution(image):
    inputs = processor.image_processor(
        images=[image],
        return_tensors="pt",
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(
            model_device,
            non_blocking=True,
        )
        for key, value in inputs.items()
    }

    pixel_values = inputs["pixel_values"]

    pixel_values = pixel_values.clone().detach()
    pixel_values.requires_grad_(True)

    inputs["pixel_values"] = pixel_values

    embedding = get_embedding(
        model,
        inputs,
    ).float()

    logits = classifier(
        embedding.unsqueeze(0)
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predicted_class = logits.argmax(
        dim=-1
    ).item()

    predicted_score = logits[
        0,
        predicted_class
    ]

    model.zero_grad(set_to_none=True)

    gradient = torch.autograd.grad(
        predicted_score,
        pixel_values,
        retain_graph=False,
        create_graph=False,
    )[0]

    attribution = (
        gradient * pixel_values
    ).abs()

    attribution = attribution.mean(
        dim=1
    )

    attribution = attribution.squeeze(0)

    grid = (
        inputs["image_grid_thw"][0]
        .detach()
        .cpu()
        .tolist()
    )

    temporal = int(grid[0])
    grid_h = int(grid[1])
    grid_w = int(grid[2])

    num_patches = attribution.shape[0]

    expected = (
        temporal
        * grid_h
        * grid_w
    )

    if num_patches != expected:
        raise ValueError(
            f"Patch/grid mismatch: "
            f"{num_patches} vs {expected}"
        )

    merge = processor.image_processor.merge_size
    attribution = attribution.reshape(temporal, grid_h // merge, grid_w // merge, merge, merge)
    attribution = attribution.permute(0, 1, 3, 2, 4).reshape(temporal, grid_h, grid_w)

    attribution = attribution.mean(
        dim=0
    )

    attribution = (
        attribution
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return (
        attribution,
        predicted_class,
        probabilities[
            0,
            predicted_class
        ].item(),
    )


def normalize_map(x):
    x = x - x.min()

    maximum = x.max()

    if maximum < 1e-12:
        return x

    return x / maximum


def attribution_entropy(x):
    values = x.flatten()

    values = values / (
        values.sum() + 1e-12
    )

    entropy = -(
        values
        * np.log(values + 1e-12)
    ).sum()

    max_entropy = np.log(
        len(values)
    )

    return entropy / max_entropy


def top_fraction_mass(
    x,
    fraction=0.10,
):
    values = np.sort(
        x.flatten()
    )[::-1]

    n = max(
        1,
        int(
            len(values)
            * fraction
        ),
    )

    return values[:n].sum() / (values.sum() + 1e-12)


def center_of_mass(x):
    x = x / (
        x.sum() + 1e-12
    )

    height, width = x.shape

    y_coords, x_coords = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij",
    )

    center_x = (
        x_coords * x
    ).sum() / (width - 1)

    center_y = (
        y_coords * x
    ).sum() / (height - 1)

    return center_x, center_y


def save_visualization(
    image,
    attribution,
    sample_index,
    label,
    prediction,
    confidence,
):
    attribution = normalize_map(
        attribution
    )

    attribution_image = np.array(
        Image.fromarray(
            (
                attribution * 255
            ).astype(np.uint8)
        ).resize(
            image.size,
            Image.Resampling.BILINEAR,
        )
    ) / 255.0

    threshold = np.quantile(
        attribution,
        0.90,
    )

    top10_mask = (
        attribution >= threshold
    ).astype(float)

    top10_mask = np.array(
        Image.fromarray(
            (
                top10_mask * 255
            ).astype(np.uint8)
        ).resize(
            image.size,
            Image.Resampling.NEAREST,
        )
    ) / 255.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(image)

    axes[0].set_title(
        f"Original\n"
        f"True: {label} | "
        f"Pred: {prediction}"
    )

    axes[1].imshow(image)

    axes[1].imshow(
        attribution_image,
        alpha=0.55,
        cmap="jet",
    )

    axes[1].set_title(
        f"Class Attribution\n"
        f"Confidence: {confidence:.3f}"
    )

    axes[2].imshow(image)

    axes[2].imshow(
        top10_mask,
        alpha=0.55,
        cmap="jet",
    )

    axes[2].set_title(
        "Top 10% Most Important Regions"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_index:04d}.png",
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return attribution


indices = random.sample(
    range(len(val_dataset)),
    min(
        NUM_SAMPLES,
        len(val_dataset),
    ),
)

metrics = []

for sample_index in tqdm(
    indices,
    desc="Analyzing validation images",
):

    image, label = val_dataset[
        sample_index
    ]

    (
        attribution,
        prediction,
        confidence,
    ) = get_attribution(image)

    np.save(os.path.join(OUTPUT_DIR, f"sample_{sample_index:04d}_raw.npy"), attribution)

    entropy = attribution_entropy(
        attribution
    )

    top10 = top_fraction_mass(
        attribution,
        fraction=0.10,
    )

    center_x, center_y = center_of_mass(
        attribution
    )

    save_visualization(
        image=image,
        attribution=attribution,
        sample_index=sample_index,
        label=label,
        prediction=prediction,
        confidence=confidence,
    )

    metrics.append(
        {
            "sample": sample_index,
            "true_label": label,
            "prediction": prediction,
            "correct": int(
                label == prediction
            ),
            "confidence": confidence,
            "attribution_entropy": entropy,
            "top_10_percent_mass": top10,
            "center_x": center_x,
            "center_y": center_y,
        }
    )


metrics_df = pd.DataFrame(
    metrics
)

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "attribution_metrics.csv",
    ),
    index=False,
)

print()
print("========== ATTRIBUTION ANALYSIS ==========")

print(
    f"Samples: "
    f"{len(metrics_df)}"
)

print(
    f"Accuracy: "
    f"{metrics_df['correct'].mean():.4f}"
)

print(
    f"Mean confidence: "
    f"{metrics_df['confidence'].mean():.4f}"
)

print(
    f"Mean attribution entropy: "
    f"{metrics_df['attribution_entropy'].mean():.4f}"
)

print(
    f"Mean top-10% attribution mass: "
    f"{metrics_df['top_10_percent_mass'].mean():.4f}"
)

print(
    f"Mean attribution center: "
    f"x={metrics_df['center_x'].mean():.3f}, "
    f"y={metrics_df['center_y'].mean():.3f}"
)

print()
print(
    f"Saved figures to: "
    f"{OUTPUT_DIR}"
)

print(
    f"Saved metrics to: "
    f"{OUTPUT_DIR}/attribution_metrics.csv"
)
summary = {"checkpoint": CHECKPOINT_PATH, "classifier": CLASSIFIER_PATH, "split": "val", "samples": indices,
    "method": "absolute gradient times normalized input, predicted class logit; mean over patch elements",
    "patch_order": "Qwen merge blocks restored to spatial raster order", "mass_definition": "top 10 percent of patches divided by total raw attribution",
    "mean_metrics": metrics_df.select_dtypes(include="number").mean().to_dict()}
(Path(OUTPUT_DIR) / "summary.json").write_text(json.dumps(summary, indent=2))
