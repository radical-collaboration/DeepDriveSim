"""Config loading utility for DeepDriveSim workflows."""

import os
from pathlib import Path

import yaml


def load_config(config_file: str) -> dict:
    """Load a YAML config file, expanding environment variables in string values."""
    path = Path(config_file)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return {
            k: os.path.expandvars(v) if isinstance(v, str) else v
            for k, v in raw.items()
        }
    return {}
