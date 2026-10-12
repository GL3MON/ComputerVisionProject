"""Version-aware pooling of merged vision tokens."""

import torch


def image_embeddings(model, inputs):
    result = model.get_image_features(
        pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
    )
    embeddings = getattr(result, "pooler_output", None)
    if embeddings is None:
        embeddings = result[0]
    if isinstance(embeddings, torch.Tensor):
        embeddings = (embeddings,) if embeddings.ndim == 2 else tuple(embeddings)
    return torch.stack([embedding.mean(dim=0) for embedding in embeddings])


def restore_patch_grid(values, grid, merge_size):
    temporal, height, width = map(int, grid)
    if height % merge_size or width % merge_size or values.numel() != temporal * height * width:
        raise ValueError("Patch/grid dimensions do not match the spatial merge size")
    if not values.is_floating_point():
        values = values.float()
    return (
        values.reshape(temporal, height // merge_size, width // merge_size, merge_size, merge_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(temporal, height, width)
        .mean(dim=0)
    )
