"""Per-seed JSON logging so every figure and table regenerates from disk."""
from __future__ import annotations
import json
import pathlib
import numpy as np

from ..utils import RESULTS_DIR, ensure_dir


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def result_path(env: str, controller: str, regime: str, seed: int) -> pathlib.Path:
    d = ensure_dir(RESULTS_DIR / env / controller / regime)
    return d / f"seed{seed}.json"


def save_result(env, controller, regime, seed, payload: dict) -> pathlib.Path:
    path = result_path(env, controller, regime, seed)
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    return path


def load_result(env, controller, regime, seed) -> dict:
    with open(result_path(env, controller, regime, seed)) as f:
        return json.load(f)


def load_all(env, controller, regime, seeds) -> list:
    out = []
    for s in seeds:
        p = result_path(env, controller, regime, s)
        if p.exists():
            with open(p) as f:
                out.append(json.load(f))
    return out
