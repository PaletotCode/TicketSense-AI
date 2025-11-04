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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

try:
    from google.cloud import storage  # type: ignore
except ImportError:  # pragma: no cover
    storage = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara e valida o dataset.")
    parser.add_argument("--dataset", type=Path, help="Caminho explícito para o dataset JSONL.")
    parser.add_argument("--upload", action="store_true", help="Envia o dataset para o GCS (se configurado).")
    parser.add_argument("--analysis", action="store_true", help="Exibe métricas adicionais de variabilidade.")
    return parser.parse_args()


def upload_dataset_to_gcs(dataset_path: Path) -> None:
    if storage is None:
        raise RuntimeError("google-cloud-storage não está instalado.")
    client = storage.Client()  # type: ignore
    bucket = client.bucket(gcs_config.bucket_name)
    blob = bucket.blob(gcs_config.dataset_path)
    blob.upload_from_filename(str(dataset_path))
    LOGGER.info("☁️ Dataset enviado para gs://%s/%s", gcs_config.bucket_name, gcs_config.dataset_path)


def main() -> None:
    args = parse_args()
    dataset_path = ensure_dataset(
        gcs_config=gcs_config,
        training_config=training_config,
        dataset_override=str(args.dataset) if args.dataset else None,
    )
    texts, intents = load_dataset(dataset_path)

    intent_counts = Counter(label for labels in intents for label in labels)
    LOGGER.info("📄 Dataset validado em %s", dataset_path)
    LOGGER.info("📝 Mensagens de usuário: %s", len(texts))
    LOGGER.info("🏷️  Intents únicas: %s", len(intent_counts))
    LOGGER.info("🏷️  Distribuição das intents: %s", dict(intent_counts))

    if args.analysis:
        lengths = [len(text.split()) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        diversity = {
            intent: len({texts[idx] for idx, labels in enumerate(intents) if intent in labels})
            for intent in intent_counts
        }
        dup_ratio = 1 - (len(set(texts)) / len(texts))
        LOGGER.info("📏 Comprimento médio (tokens): %.2f", avg_len)
        LOGGER.info("📏 Comprimento mínimo/máximo (tokens): %d / %d", min_len, max_len)
        LOGGER.info("🎯 Diversidade (textos únicos por intent): %s", diversity)
        LOGGER.info("🔁 Relação de duplicatas aproximada: %.2f%%", dup_ratio * 100)

    if args.upload:
        upload_dataset_to_gcs(dataset_path)


if __name__ == "__main__":
    main()
