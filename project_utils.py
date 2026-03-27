from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset

from config import LOGS_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR

TLINK_DATASET_NAME = "mdg-nlp/tlink-extr-classification-sentence"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_LABEL_FIELD = "label"


def ensure_project_dirs() -> None:
    for path in (MODELS_DIR, RESULTS_DIR, LOGS_DIR, RESULTS_DIR / "attacks", RESULTS_DIR / "summary"):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_project_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def load_tlink_dataset() -> DatasetDict:
    return load_dataset(TLINK_DATASET_NAME)


def build_label_mappings(dataset: Dataset, label_field: str = DEFAULT_LABEL_FIELD) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted({str(label) for label in dataset[label_field]})
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(
    dataset: Dataset,
    label2id: dict[str, int],
    label_field: str = DEFAULT_LABEL_FIELD,
    encoded_field: str = "labels",
) -> Dataset:
    def mapper(example: dict[str, Any]) -> dict[str, int]:
        return {encoded_field: label2id[str(example[label_field])]}

    return dataset.map(mapper)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
