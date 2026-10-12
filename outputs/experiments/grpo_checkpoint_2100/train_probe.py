import os
import random
import numpy as np
import torch
import torch.nn as nn
import medmnist
from medmnist import INFO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score, f1_score, classification_report


import json
from pathlib import Path
torch.set_num_threads(2)
CHECKPOINT_PATH = '/home/harishr/MedGRPO/qwen3-vl-medical-grpo-vision-lora/checkpoint-2100'

MODEL_NAME = "GL3MON/qwen3-vl-2b-medical-pmcvqa-sft-merged"
DATASET_NAME = "pneumoniamnist"
IMAGE_SIZE = 224

BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 3407

OUTPUT_DIR = '/home/harishr/Med/outputs/experiments/grpo_checkpoint_2100/classifier'
EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

print(f"Device: {device}")
print(f"Dtype: {dtype}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")


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


def build_dataset(split):
    dataset_info = INFO[DATASET_NAME]
    dataset_class = getattr(medmnist, dataset_info["python_class"])

    dataset = dataset_class(
        split=split,
        size=IMAGE_SIZE,
        download=True,
        as_rgb=True,
    )

    return MedMNISTWrapper(dataset)


processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, local_files_only=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
)
from peft import PeftModel
from safetensors.torch import save_file
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=False).merge_and_unload()
model = model.to(device)
print("Loaded and merged checkpoint:", CHECKPOINT_PATH, flush=True)

encoder_dir = Path(OUTPUT_DIR).parent / "encoder"
encoder_dir.mkdir(exist_ok=True)
save_file({k: v.detach().cpu().contiguous() for k, v in model.model.visual.state_dict().items()}, str(encoder_dir / "vision_encoder.safetensors"))
(encoder_dir / "vision_config.json").write_text(json.dumps(model.config.vision_config.to_dict(), indent=2))
processor.save_pretrained(str(encoder_dir / "processor"))
print("Exported merged vision encoder:", encoder_dir, flush=True)


model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long,
    )

    inputs = processor.image_processor(
        images=images,
        return_tensors="pt",
    )

    return inputs, labels


train_dataset = build_dataset("train")
val_dataset = build_dataset("val")
test_dataset = build_dataset("test")

if MAX_TRAIN_SAMPLES is not None:
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        range(min(MAX_TRAIN_SAMPLES, len(train_dataset))),
    )

if MAX_VAL_SAMPLES is not None:
    val_dataset = torch.utils.data.Subset(
        val_dataset,
        range(min(MAX_VAL_SAMPLES, len(val_dataset))),
    )

if MAX_TEST_SAMPLES is not None:
    test_dataset = torch.utils.data.Subset(
        test_dataset,
        range(min(MAX_TEST_SAMPLES, len(test_dataset))),
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)


def extract_features(loader, split_name):
    model.eval()

    all_features = []
    all_labels = []

    for inputs, labels in tqdm(
        loader,
        desc=f"Extracting {split_name}",
    ):
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=torch.cuda.is_available(),
            ):
                features = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                )

        # features is a tuple: (embeddings_tuple, hidden_states_list)
        # embeddings_tuple is a tuple of tensors, one per image in batch
        image_embeds = features.pooler_output

        batch_features = torch.stack([embed.mean(dim=0) for embed in image_embeds])

        all_features.append(batch_features.float().cpu())
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    torch.save(
        features,
        os.path.join(EMBEDDING_DIR, f"{split_name}_features.pt"),
    )

    torch.save(
        labels,
        os.path.join(EMBEDDING_DIR, f"{split_name}_labels.pt"),
    )

    print(
        f"{split_name}: features={features.shape}, labels={labels.shape}"
    )

    return features, labels


train_features, train_labels = extract_features(
    train_loader,
    "train",
)

val_features, val_labels = extract_features(
    val_loader,
    "val",
)

test_features, test_labels = extract_features(
    test_loader,
    "test",
)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, x):
        return self.classifier(x)


input_dim = train_features.shape[1]
num_classes = len(torch.unique(train_labels))

print(f"Embedding dimension: {input_dim}")
print(f"Number of classes: {num_classes}")

train_tensor_dataset = torch.utils.data.TensorDataset(
    train_features,
    train_labels,
)

val_tensor_dataset = torch.utils.data.TensorDataset(
    val_features,
    val_labels,
)

test_tensor_dataset = torch.utils.data.TensorDataset(
    test_features,
    test_labels,
)

classifier_batch_size = 256

train_feature_loader = DataLoader(
    train_tensor_dataset,
    batch_size=classifier_batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

val_feature_loader = DataLoader(
    val_tensor_dataset,
    batch_size=classifier_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

test_feature_loader = DataLoader(
    test_tensor_dataset,
    batch_size=classifier_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)


classifier = LinearClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


def evaluate(loader):
    classifier.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = classifier(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            predictions = logits.argmax(dim=1)

            all_predictions.append(
                predictions.cpu()
            )
            all_targets.append(
                labels.cpu()
            )

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    loss = total_loss / total_samples
    accuracy = accuracy_score(
        targets,
        predictions,
    )
    f1 = f1_score(
        targets,
        predictions,
        average="weighted",
    )

    return loss, accuracy, f1, targets, predictions


best_val_accuracy = -1.0
best_epoch = -1

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
}


for epoch in range(1, EPOCHS + 1):
    classifier.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_feature_loader,
        desc=f"Epoch {epoch}/{EPOCHS}",
    )

    for features, labels in progress:
        features = features.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = classifier(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            predictions == labels
        ).sum().item()
        train_total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{train_correct / train_total:.4f}",
        )

    train_loss /= train_total
    train_accuracy = train_correct / train_total

    val_loss, val_accuracy, val_f1, _, _ = evaluate(
        val_feature_loader
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": classifier.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "model_name": MODEL_NAME,
                "adapter_checkpoint": CHECKPOINT_PATH,
                "dataset_name": DATASET_NAME,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
            },
            os.path.join(
                OUTPUT_DIR,
                "linear_classifier_best.pt",
            ),
        )


checkpoint = torch.load(
    os.path.join(
        OUTPUT_DIR,
        "linear_classifier_best.pt",
    ),
    map_location=device,
)

classifier.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_accuracy, test_f1, test_targets, test_predictions = evaluate(
    test_feature_loader
)

print()
print("Best checkpoint")
print(f"Epoch: {checkpoint['best_epoch']}")
print(f"Validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
print()
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test weighted F1: {test_f1:.4f}")
print()
print(
    classification_report(
        test_targets,
        test_predictions,
        digits=4,
    )
)

torch.save(
    {
        "model_state_dict": classifier.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "model_name": MODEL_NAME,
                "adapter_checkpoint": CHECKPOINT_PATH,
        "dataset_name": DATASET_NAME,
        "image_size": IMAGE_SIZE,
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "best_epoch": checkpoint["best_epoch"],
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "history": history,
    },
    os.path.join(
        OUTPUT_DIR,
        "linear_classifier_final.pt",
    ),
)

print()
print(f"Saved to: {OUTPUT_DIR}")
print(f"Best classifier: {OUTPUT_DIR}/linear_classifier_best.pt")
print(f"Final classifier: {OUTPUT_DIR}/linear_classifier_final.pt")
print(f"Embeddings: {EMBEDDING_DIR}")

from sklearn.metrics import roc_auc_score, confusion_matrix
with torch.inference_mode():
    test_probabilities = torch.softmax(classifier(test_features.to(device)), dim=-1).cpu().numpy()
np.savez(Path(OUTPUT_DIR) / "test_predictions.npz", targets=test_targets, predictions=test_predictions, probabilities=test_probabilities)
summary = {"checkpoint": CHECKPOINT_PATH, "base_model": MODEL_NAME, "dataset": DATASET_NAME, "seed": SEED,
    "image_size": IMAGE_SIZE, "processor": CHECKPOINT_PATH, "pooling": "mean of merged vision tokens",
    "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "classifier_batch_size": classifier_batch_size,
    "split_sizes": {"train": len(train_labels), "val": len(val_labels), "test": len(test_labels)},
    "input_dim": input_dim, "best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy,
    "test_loss": test_loss, "test_accuracy": test_accuracy, "test_weighted_f1": test_f1,
    "test_auroc": roc_auc_score(test_targets, test_probabilities[:, 1]),
    "confusion_matrix": confusion_matrix(test_targets, test_predictions).tolist(),
    "classification_report": classification_report(test_targets, test_predictions, output_dict=True), "history": history}
(Path(OUTPUT_DIR) / "metrics.json").write_text(json.dumps(summary, indent=2))
