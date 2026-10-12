"""Configuration loading with isolated defaults and explicit overrides."""

from copy import deepcopy
import inspect
from pathlib import Path


def deep_merge(base, override):
    """Recursively merge override dict into base dict, returning a new dict."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def filter_supported_kwargs(cls, kwargs, warn=True):
    """Drop kwargs that `cls.__init__` doesn't accept, warning about each one.

    Guards against transformers-version drift, e.g. `save_safetensors` not
    existing on some TrainingArguments versions.
    """
    accepted = set(inspect.signature(cls.__init__).parameters)
    supported = {}
    dropped = []
    for key, value in kwargs.items():
        if key in accepted:
            supported[key] = value
        else:
            dropped.append(key)
    if dropped and warn:
        print(
            f"[config] Dropping unsupported {cls.__name__} argument(s) for "
            f"your installed transformers version: {dropped}"
        )
    return supported


def load_config(path=None, defaults=None):
    config = deepcopy(defaults or {})
    if path is not None:
        import yaml

        with Path(path).open(encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        if not isinstance(override, dict):
            raise ValueError("Configuration must be a mapping")
        config = deep_merge(config, override)
    return config
