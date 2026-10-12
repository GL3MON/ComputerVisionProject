import os
import random
import numpy as np
import torch
import torch.nn as nn
import medmnist
from medmnist import INFO
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

MODEL_BASE = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_MEDICAL = "GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged"
CLASSIFIER_BASE = "./qwen3vl_linear_classifier/linear_classifier_best.pt"
CLASSIFIER_MEDICAL = "./qwen3vl_medical_linear_classifier/linear_classifier_best.pt"
DATASET_NAME = "pneumoniamnist"
IMAGE_SIZE = 224
SEED = 3407

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

class MedMNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = Image.fromarray(np.asarray(image)).convert("RGB")
        label = int(np.asarray(label).squeeze())
        return image, label

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)

dataset_info = INFO[DATASET_NAME]
DatasetClass = getattr(medmnist, dataset_info["python_class"])
val_dataset = MedMNISTWrapper(
    DatasetClass(split="val", size=IMAGE_SIZE, download=True, as_rgb=True)
)

processor = AutoProcessor.from_pretrained(MODEL_BASE)

model_base = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_BASE, dtype=dtype, device_map="auto"
)
model_base.eval()
for p in model_base.parameters():
    p.requires_grad_(False)

model_medical = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_MEDICAL, dtype=dtype, device_map="auto"
)
model_medical.eval()
for p in model_medical.parameters():
    p.requires_grad_(False)

ckpt_base = torch.load(CLASSIFIER_BASE, map_location="cpu")
clf_base = LinearClassifier(ckpt_base["input_dim"], ckpt_base["num_classes"])
clf_base.load_state_dict(ckpt_base["model_state_dict"])
clf_base = clf_base.to(device).eval()
for p in clf_base.parameters():
    p.requires_grad_(False)

ckpt_medical = torch.load(CLASSIFIER_MEDICAL, map_location="cpu")
clf_medical = LinearClassifier(ckpt_medical["input_dim"], ckpt_medical["num_classes"])
clf_medical.load_state_dict(ckpt_medical["model_state_dict"])
clf_medical = clf_medical.to(device).eval()
for p in clf_medical.parameters():
    p.requires_grad_(False)


def get_embedding(model, inputs):
    features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    feature = features[0][0]
    if feature.ndim == 2:
        return feature.mean(dim=0)
    if feature.ndim == 3:
        return feature[0].mean(dim=0)
    raise ValueError(f"Unexpected feature shape: {feature.shape}")


def get_attribution(model, clf, image):
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}

    pixel_values = inputs["pixel_values"].clone().detach()
    pixel_values.requires_grad_(True)
    inputs["pixel_values"] = pixel_values

    embedding = get_embedding(model, inputs).float()
    logits = clf(embedding.unsqueeze(0))

    pred = logits.argmax(dim=-1).item()
    score = logits[0, pred]

    model.zero_grad(set_to_none=True)

    gradient = torch.autograd.grad(
        score, pixel_values, retain_graph=False, create_graph=False
    )[0]

    attribution = (gradient * pixel_values).abs().mean(dim=1).squeeze(0)

    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    temporal, grid_h, grid_w = int(grid[0]), int(grid[1]), int(grid[2])

    attribution = attribution.reshape(temporal, grid_h, grid_w).mean(dim=0)
    attribution = attribution.detach().float().cpu().numpy()

    return attribution, pred, logits[0, pred].item()


def normalize_map(x):
    x = x - x.min()
    max_val = x.max()
    if max_val < 1e-12:
        return x
    return x / max_val


def resize_to_image(attr_map, image):
    return np.array(
        Image.fromarray((attr_map * 255).astype(np.uint8)).resize(
            image.size, Image.Resampling.BILINEAR
        )
    ) / 255.0


OUTPUT_DIR = "./attribution_diff_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

differences = []

for idx in tqdm(range(len(val_dataset)), desc="Computing attributions"):
    image, label = val_dataset[idx]

    attr_base, pred_base, conf_base = get_attribution(model_base, clf_base, image)
    attr_med, pred_med, conf_med = get_attribution(model_medical, clf_medical, image)

    attr_base = normalize_map(attr_base)
    attr_med = normalize_map(attr_med)

    attr_base_img = resize_to_image(attr_base, image)
    attr_med_img = resize_to_image(attr_med, image)

    diff = attr_med_img - attr_base_img
    diff_abs = np.abs(diff)

    diff_mean = diff.mean()
    diff_std = diff.std()
    diff_max = diff.max()
    diff_min = diff.min()

    differences.append({
        "sample": idx,
        "label": label,
        "pred_base": pred_base,
        "pred_med": pred_med,
        "correct_base": int(label == pred_base),
        "correct_med": int(label == pred_med),
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_max": diff_max,
        "diff_min": diff_min,
    })

import pandas as pd
df = pd.DataFrame(differences)
df.to_csv(os.path.join(OUTPUT_DIR, "attribution_differences.csv"), index=False)

med_correct_base_wrong = df[(df["correct_med"] == 1) & (df["correct_base"] == 0)]
base_correct_med_wrong = df[(df["correct_base"] == 1) & (df["correct_med"] == 0)]

print(f"Medical correct, Base wrong: {len(med_correct_base_wrong)}")
print(f"Base correct, Medical wrong: {len(base_correct_med_wrong)}")

for _, row in med_correct_base_wrong.iterrows():
    idx = int(row["sample"])
    image, label = val_dataset[idx]

    attr_base, pred_base, conf_base = get_attribution(model_base, clf_base, image)
    attr_med, pred_med, conf_med = get_attribution(model_medical, clf_medical, image)

    attr_base = normalize_map(attr_base)
    attr_med = normalize_map(attr_med)

    attr_base_img = resize_to_image(attr_base, image)
    attr_med_img = resize_to_image(attr_med, image)
    diff = attr_med_img - attr_base_img

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original Image\nTrue: {label}", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    im1 = axes[0, 1].imshow(attr_base_img, alpha=0.6, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Base Model Attribution\nPred: {pred_base} (conf: {conf_base:.3f})", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(image)
    im2 = axes[0, 2].imshow(attr_med_img, alpha=0.6, cmap="jet", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Medical Model Attribution\nPred: {pred_med} (conf: {conf_med:.3f})", fontsize=11)
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Difference (Medical - Base)\nRed: Medical attends more, Blue: Base attends more", fontsize=11)
    axes[1, 0].axis("off")
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(image)
    diff_pos = np.maximum(diff, 0)
    im3 = axes[1, 1].imshow(diff_pos, alpha=0.6, cmap="Reds", vmin=0, vmax=1)
    axes[1, 1].set_title("Medical > Base (Positive Difference)", fontsize=11)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(image)
    diff_neg = np.maximum(-diff, 0)
    im4 = axes[1, 2].imshow(diff_neg, alpha=0.6, cmap="Blues", vmin=0, vmax=1)
    axes[1, 2].set_title("Base > Medical (Negative Difference)", fontsize=11)
    axes[1, 2].axis("off")
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Sample {idx} | Medical Correct (Pred={pred_med}), Base Wrong (Pred={pred_base}) | "
        f"Diff Mean: {diff.mean():.4f}, Std: {diff.std():.4f}",
        fontsize=13
    )
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"diff_sample_{idx:04d}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

print(f"\nAll difference plots saved to {OUTPUT_DIR}")
print(f"Metrics CSV saved to {OUTPUT_DIR}/attribution_differences.csv")