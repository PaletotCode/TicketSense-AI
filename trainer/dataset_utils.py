from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def ensure_dataset(gcs_config, training_config, dataset_override: str | None = None) -> Path:
    explicit = dataset_override or os.getenv("LOCAL_DATASET_PATH")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            LOGGER.info("📁 Usando dataset informado em %s", candidate)
            return candidate
        raise FileNotFoundError(f"Dataset indicado não encontrado: {candidate}")

    inferred_name = Path(gcs_config.dataset_path).name or "dataset.jsonl"
    local_dataset = training_config.local_data_dir / inferred_name
    if local_dataset.exists():
        LOGGER.info("📁 Usando dataset local em %s", local_dataset)
        return local_dataset.resolve()

    repo_dataset = Path("dataset.jsonl")
    if repo_dataset.exists():
        local_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_dataset, local_dataset)
        LOGGER.info("📥 Copiado dataset do repositório para %s", local_dataset)
        return local_dataset.resolve()

    if storage is None:
        raise RuntimeError(
            "Dataset não encontrado localmente e google-cloud-storage indisponível. "
            "Defina LOCAL_DATASET_PATH ou configure o GCS."
        )

    credentials_present = bool(
        gcs_config.credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not credentials_present:
        raise RuntimeError(
            "Dataset não encontrado localmente e credenciais GCS ausentes. "
            "Defina LOCAL_DATASET_PATH, ajuste GCS_DATASET_PATH ou configure GOOGLE_APPLICATION_CREDENTIALS."
        )

    client = storage.Client()  # type: ignore
    bucket = client.bucket(gcs_config.bucket_name)
    blob = bucket.blob(gcs_config.dataset_path)

    local_dataset.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_dataset))
    LOGGER.info(
        "📥 Dataset baixado de gs://%s/%s para %s",
        gcs_config.bucket_name,
        gcs_config.dataset_path,
        local_dataset,
    )
    return local_dataset.resolve()


def load_dataset(dataset_path: Path) -> Tuple[List[str], List[List[str]]]:
    texts: List[str] = []
    intents: List[List[str]] = []
    buffer: List[str] = []

    def process_record(record: Dict[str, object]) -> None:
        for message in record.get("messages", []):  # type: ignore[assignment]
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            text = message.get("text")
            label = message.get("intent")
            if not isinstance(text, str):
                continue
            if not isinstance(label, list) or not label:
                continue
            texts.append(text)
            intents.append([str(item) for item in label])

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                buffer.append(line)
                if stripped == "]}":
                    try:
                        record = json.loads("".join(buffer))
                        process_record(record)
                    finally:
                        buffer = []
                continue
            process_record(record)

    if buffer:
        raise ValueError("Arquivo JSONL parece incompleto; dados residuais encontrados.")
    if not texts:
        raise ValueError(f"Dataset em {dataset_path} não contém mensagens válidas.")

    return texts, intents


def build_label_mappings(intent_lists: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_intents = sorted({intent for labels in intent_lists for intent in labels})
    label2id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id2label = {idx: intent for intent, idx in label2id.items()}
    return label2id, id2label


def encode_labels(intent_lists: Sequence[Sequence[str]], label2id: Dict[str, int]) -> List[List[int]]:
    num_labels = len(label2id)
    encoded: List[List[int]] = []
    for intents in intent_lists:
        vector = [0] * num_labels
        for intent in intents:
            if intent in label2id:
                vector[label2id[intent]] = 1
        encoded.append(vector)
    return encoded


def select_indices(
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8:
        raise ValueError("As proporções devem somar 1.0")

    indices = list(range(len(labels)))

    def can_stratify(values: Sequence[int]) -> bool:
        counts = Counter(values)
        return all(count >= 2 for count in counts.values())

    stratify = labels if can_stratify(labels) else None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    temp_labels = [labels[i] for i in temp_idx]
    remainder = val_ratio + test_ratio
    if remainder <= 0:
        return train_idx, temp_idx, []

    stratify_temp = temp_labels if can_stratify(temp_labels) else None

    val_size = test_ratio / remainder
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )
    return train_idx, val_idx, test_idx


def slice_encodings(encodings, labels, indices: Sequence[int]):
    input_ids = [encodings["input_ids"][i] for i in indices]
    attention = [encodings["attention_mask"][i] for i in indices]
    label_slice = [labels[i] for i in indices]

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
    }, label_slice


class IntentDataset(Dataset):
    def __init__(self, encodings, labels) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item
