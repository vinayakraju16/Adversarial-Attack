from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CODES_DIR = PROJECT_ROOT / "codes"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def _sanitize(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def make_run_id(dataset: str, model: str, attack: str, n: int, seed: int) -> str:
    dataset_part = _sanitize(dataset)
    model_part = _sanitize(model)
    attack_part = _sanitize(attack)
    return f"{dataset_part}_{model_part}_{attack_part}_n{int(n)}_seed{int(seed)}"
