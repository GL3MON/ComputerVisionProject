"""Resolve repository paths without changing the working directory."""

import os
from pathlib import Path


def project_root():
    configured = os.environ.get("MEDICAL_VLM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return Path.cwd().resolve()


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def output_root():
    return project_root() / "outputs"


def configure_wandb():
    directory = output_root() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(directory))
