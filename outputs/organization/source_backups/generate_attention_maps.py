import os
import logging

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import medmnist

from tqdm.auto import tqdm
from medmnist import INFO
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)


MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

DATASET_NAME = "pneumoniamnist"
IMAGE_SIZE = 224

NUM_SAMPLES = 100
OUTPUT_DIR = "qwen3vl_attention_maps"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type != "cuda":
    raise RuntimeError(
        "A CUDA GPU is required."
    )


if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16


logger.info(
    "Device: %s",
    device,
)

logger.info(
    "GPU: %s",
    torch.cuda.get_device_name(0),
)

logger.info(
    "Dtype: %s",
    dtype,
)


logger.info(
    "Loading processor..."
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
)


logger.info(
    "Loading Qwen3-VL..."
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

logger.info(
    "Model loaded successfully."
)


logger.info(
    "Loading MedMNIST: %s",
    DATASET_NAME,
)

dataset_info = INFO[
    DATASET_NAME
]

DatasetClass = getattr(
    medmnist,
    dataset_info[
        "python_class"
    ],
)

val_dataset = DatasetClass(
    split="val",
    size=IMAGE_SIZE,
    download=True,
    as_rgb=True,
)

logger.info(
    "Validation samples: %d",
    len(val_dataset),
)

logger.info(
    "Number of classes: %d",
    len(dataset_info["label"]),
)

logger.info(
    "Labels: %s",
    dataset_info["label"],
)


num_samples = min(
    NUM_SAMPLES,
    len(val_dataset),
)

logger.info(
    "Generating attention maps for %d samples.",
    num_samples,
)


def attention_rollout(attentions):
    """
    Process attention weights from Qwen3-VL visual encoder.
    The attention weights are of shape [num_heads, seq_len] per layer,
    representing attention from CLS token to all tokens.
    """
    all_layer_attentions = []

    for i, attention in enumerate(attentions):
        attention = attention.float()

        # Average over heads: [num_heads, seq_len] -> [seq_len]
        attention = attention.mean(dim=0)

        all_layer_attentions.append(attention)

    # Average across layers: [num_layers, seq_len] -> [seq_len]
    stacked = torch.stack(all_layer_attentions, dim=0)

    attention_map = stacked.mean(dim=0)

    return attention_map


def generate_attention_map(
    image,
    label,
    index,
):

    inputs = processor.image_processor(
        images=[image],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
        ):

            vision_outputs = model.model.visual(
                inputs["pixel_values"],
                grid_thw=inputs["image_grid_thw"],
                output_attentions=True,
                return_dict=True,
            )

    if isinstance(vision_outputs, tuple):
        attentions = vision_outputs[1] if len(vision_outputs) > 1 else None
    elif hasattr(vision_outputs, "attentions"):
        attentions = vision_outputs.attentions
    else:
        raise RuntimeError(
            f"Expected vision outputs with attentions, "
            f"but got {type(vision_outputs)}"
        )

    if attentions is None:
        raise RuntimeError(
            "Vision encoder returned no attention weights. "
            "Make sure attn_implementation='eager' "
            "and output_attentions=True."
        )


    rollout = attention_rollout(
        attentions
    )


    grid = (
        inputs[
            "image_grid_thw"
        ][0]
        .detach()
        .cpu()
        .tolist()
    )

    temporal, height, width = grid

    num_patches = height * width

    # The attention sequence length (2048) is larger than spatial patches (256).
    # This is likely because the visual encoder computes attention at a higher
    # resolution before pooling. Use the first num_patches tokens.
    if rollout.shape[0] >= num_patches:
        attention_map = rollout[:num_patches]
    else:
        logger.warning(f"Unexpected seq_len {rollout.shape[0]} < num_patches {num_patches}")
        attention_map = rollout

    attention_map = attention_map.reshape(
        height,
        width,
    )


    attention_map = attention_map.float()

    attention_map -= (
        attention_map.min()
    )

    attention_map /= (
        attention_map.max()
        + 1e-8
    )


    attention_map = F.interpolate(
        attention_map[
            None,
            None,
            ...,
        ],
        size=(
            image.height,
            image.width,
        ),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


    attention_map = (
        attention_map
        .detach()
        .cpu()
        .numpy()
    )


    image_array = np.asarray(
        image
    )


    label_name = dataset_info[
        "label"
    ].get(
        str(label),
        str(label),
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )


    axes[0].imshow(
        image_array
    )

    axes[0].set_title(
        f"Original\n"
        f"Label: {label_name}"
    )

    axes[0].axis("off")


    axes[1].imshow(
        attention_map,
        cmap="jet",
    )

    axes[1].set_title(
        "Attention Rollout"
    )

    axes[1].axis("off")


    axes[2].imshow(
        image_array
    )

    axes[2].imshow(
        attention_map,
        cmap="jet",
        alpha=0.5,
    )

    axes[2].set_title(
        "Attention Overlay"
    )

    axes[2].axis("off")


    plt.tight_layout()


    output_path = os.path.join(
        OUTPUT_DIR,
        f"attention_{index:05d}_"
        f"label_{label}.png",
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


    return output_path


successful = 0


progress_bar = tqdm(
    range(num_samples),
    desc="Generating attention maps",
    unit="image",
)


for index in progress_bar:

    image, label = val_dataset[
        index
    ]


    label = int(
        np.asarray(
            label
        ).squeeze()
    )


    try:
        output_path = generate_attention_map(
            image=image,
            label=label,
            index=index,
        )

        successful += 1

        progress_bar.set_postfix(
            saved=successful
        )

        logger.info(
            "Saved sample %d → %s",
            index,
            output_path,
        )

    except Exception as error:

        logger.exception(
            "Failed on sample %d: %s",
            index,
            error,
        )


logger.info(
    "Finished."
)

logger.info(
    "Successfully generated: %d/%d",
    successful,
    num_samples,
)

logger.info(
    "Output directory: %s",
    OUTPUT_DIR,
)