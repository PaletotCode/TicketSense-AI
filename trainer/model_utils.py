"""
Utilitários relacionados a modelo e tokenizer do PingFy_IA.

Agrupa funções para criação/carregamento do DistilBERT, gerenciamento de
label maps e upload opcional dos artefatos para o GCS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def create_tokenizer(model_config):
    """Retorna o tokenizer configurado para o projeto."""
    return AutoTokenizer.from_pretrained(model_config.model_name)


def create_model(
    model_config,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> torch.nn.Module:
    """Instancia o AutoModel para classificação de intents."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        dtype=torch.float32,
    )


def save_label_map(
    output_dir: Path,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Path:
    """Salva o mapeamento de intents em JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = output_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "intent_to_id": label2id,
                "id_to_intent": {str(k): v for k, v in id2label.items()},
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    LOGGER.info("💾 label_map.json salvo em %s", label_map_path)
    return label_map_path


def maybe_upload_to_gcs(model_dir: Path, gcs_config) -> None:
    """Realiza upload opcional dos artefatos treinados para o GCS."""
    if storage is None:
        LOGGER.info("☁️ google-cloud-storage não disponível; pulando upload.")
        return

    credentials_present = bool(gcs_config.credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    if not credentials_present:
        LOGGER.info("☁️ Credenciais do GCS não configuradas; pulando upload.")
        return

    client = storage.Client()  # type: ignore
    bucket = client.bucket(gcs_config.bucket_name)
    destination_prefix = gcs_config.model_path.rstrip("/")

    LOGGER.info(
        "☁️ Iniciando upload do modelo para gs://%s/%s",
        gcs_config.bucket_name,
        destination_prefix,
    )

    for local_file in model_dir.glob("**/*"):
        if local_file.is_dir():
            continue
        remote_path = f"{destination_prefix}/{local_file.relative_to(model_dir)}"
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(str(local_file))
        LOGGER.debug("  ↗︎ %s", remote_path)

    LOGGER.info("☁️ Upload concluído com sucesso.")
