"""Shared utilities: deterministic seeding, config loading, repo paths."""
from __future__ import annotations
import os
import random
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CLIPS_DIR = REPO_ROOT / "clips"


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so a run is bit-for-bit reproducible on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(env_key: str) -> dict:
    """Load an environment config plus the shared drift schedule for that env."""
    cfg = load_yaml(CONFIG_DIR / f"{env_key}.yaml")
    drifts = load_yaml(CONFIG_DIR / "drift_schedules.yaml")
    cfg["drift"] = drifts.get(env_key, {})
    return cfg


def ensure_dir(path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
