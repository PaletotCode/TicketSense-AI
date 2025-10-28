"""
Utilitário CLI para preparar o dataset local do PingFy_IA.

Responsável por validar o formato JSONL, gerar estatísticas básicas e,
opcionalmente, enviar o arquivo para o GCS quando credenciais estiverem
configuradas.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import gcs_config, training_config  # noqa: E402
from trainer.dataset_utils import ensure_dataset, load_dataset  # noqa: E402

LOGGER = logging.getLogger("dataset-prep")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # type: ignore


def upload_dataset_to_gcs(dataset_path: Path) -> None:
    """Realiza upload do dataset validado para o GCS."""
    if storage is None:
        raise RuntimeError("google-cloud-storage não está instalado.")

    client = storage.Client()  # type: ignore
    bucket = client.bucket(gcs_config.bucket_name)
    blob = bucket.blob(gcs_config.dataset_path)
    blob.upload_from_filename(str(dataset_path))

    LOGGER.info(
        "☁️ Dataset enviado para gs://%s/%s",
        gcs_config.bucket_name,
        gcs_config.dataset_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara e valida o dataset do PingFy_IA.")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Envia o dataset validado para o GCS (requer credenciais).",
    )
    args = parser.parse_args()

    dataset_path = ensure_dataset(gcs_config=gcs_config, training_config=training_config)
    texts, intents = load_dataset(dataset_path)

    intent_counts = Counter(label for labels in intents for label in labels)
    LOGGER.info("📄 Dataset validado em %s", dataset_path)
    LOGGER.info("📝 Mensagens de usuário: %s", len(texts))
    LOGGER.info("🏷️  Intents únicas: %s", len(intent_counts))
    LOGGER.info("🏷️  Distribuição das intents: %s", dict(intent_counts))

    if args.upload:
        upload_dataset_to_gcs(dataset_path)


if __name__ == "__main__":
    main()
