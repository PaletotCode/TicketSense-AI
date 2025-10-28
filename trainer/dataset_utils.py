"""
Utilitários de dataset para o pipeline de treinamento do PingFy_IA.

Responsável por localizar, validar e preparar o dataset, além de prover
estratégias de split e o wrapper Dataset consumido pelo Trainer.
"""

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
except ImportError:  # pragma: no cover - dependência opcional
    storage = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def ensure_dataset(
    gcs_config,
    training_config,
    dataset_override: str | None = None,
) -> Path:
    """Garante que o dataset esteja disponível localmente.

    Args:
        gcs_config: Configurações de GCS.
        training_config: Configurações de treinamento.
        dataset_override: Caminho absoluto opcional para uso explícito.

    Returns:
        Path: Caminho para o dataset em JSONL.

    Raises:
        FileNotFoundError: Quando o dataset não é encontrado.
        RuntimeError: Quando gcs_config é necessário mas google-cloud-storage não está instalado.
    """
    # 1) Caminho explicitamente informado via parâmetro ou variável de ambiente
    explicit_path = dataset_override or os.getenv("LOCAL_DATASET_PATH")
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists():
            LOGGER.info("📁 Usando dataset informado em %s", candidate)
            return candidate
        raise FileNotFoundError(f"Dataset indicado não encontrado: {candidate}")

    # 2) Diretório local configurado (artifacts/data)
    inferred_name = Path(gcs_config.dataset_path).name or "dataset.jsonl"
    local_dataset = training_config.local_data_dir / inferred_name
    if local_dataset.exists():
        LOGGER.info("📁 Usando dataset local em %s", local_dataset)
        return local_dataset.resolve()

    # 3) Dataset presente na raiz do repositório
    repo_dataset = Path("dataset.jsonl")
    if repo_dataset.exists():
        local_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_dataset, local_dataset)
        LOGGER.info("📥 Copiado dataset do repositório para %s", local_dataset)
        return local_dataset.resolve()

    # 4) Download do GCS
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage não instalado e dataset não encontrado localmente. "
            "Instale a dependência ou disponibilize dataset.jsonl."
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
    """Lê o dataset JSONL e retorna textos e intents.

    Args:
        dataset_path: Caminho para o arquivo JSONL.

    Returns:
        Tuple contendo lista de textos e lista de listas de intents.
    """
    texts: List[str] = []
    intents: List[List[str]] = []
    buffer: List[str] = []

    def process_record(record: Dict) -> None:
        for message in record.get("messages", []):
            if message.get("role") != "user":
                continue
            text = message.get("text")
            label = message.get("intent") or []
            if not text or not label:
                continue
            texts.append(text)
            intents.append(label)

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
                    record = json.loads("".join(buffer))
                    process_record(record)
                    buffer = []
                continue
            else:
                process_record(record)

    if buffer:
        raise ValueError("Arquivo JSONL parece incompleto; sobras não processadas encontradas.")

    if not texts:
        raise ValueError(f"Dataset em {dataset_path} não contém mensagens de usuário válidas.")

    return texts, intents


def build_label_mappings(intent_lists: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Gera os dicionários intent->id e id->intent."""
    unique_intents = sorted({intent for labels in intent_lists for intent in labels})
    label2id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id2label = {idx: intent for intent, idx in label2id.items()}
    return label2id, id2label


def encode_labels(
    intent_lists: Sequence[Sequence[str]],
    label2id: Dict[str, int],
) -> List[List[int]]:
    """Transforma lista de intents em vetores multi-hot."""
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
    """Realiza split reprodutível em train/val/test com fallback quando estratificação falha."""
    indices = list(range(len(labels)))
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8:
        raise ValueError("As proporções de split devem somar 1.0.")

    def can_stratify(values: Sequence[int]) -> bool:
        counts = Counter(values)
        return all(count >= 2 for count in counts.values())

    stratify = labels if can_stratify(labels) else None

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=1 - train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    temp_labels = [labels[i] for i in temp_indices]
    remainder = val_ratio + test_ratio
    if remainder <= 0:
        return train_indices, temp_indices, []

    val_fraction = val_ratio / remainder
    stratify_temp = temp_labels if can_stratify(temp_labels) else None

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1 - val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )

    return train_indices, val_indices, test_indices


def slice_encodings(
    encodings: Dict[str, Sequence[Sequence[int]]],
    labels: Sequence[int],
    indices: Sequence[int],
) -> Tuple[Dict[str, List[Sequence[int]]], List[int]]:
    """Seleciona subconjuntos dos encodings e rótulos a partir dos índices informados."""
    subset_encodings = {key: [val[i] for i in indices] for key, val in encodings.items()}
    subset_labels = [labels[i] for i in indices]
    return subset_encodings, subset_labels


class IntentDataset(Dataset):
    """Wrapper Dataset compatível com o HuggingFace Trainer."""

    def __init__(self, encodings: Dict[str, Sequence[Sequence[int]]], labels: Sequence[Sequence[int]]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item
