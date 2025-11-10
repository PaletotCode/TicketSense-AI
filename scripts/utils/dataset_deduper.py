from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.llm_clients import LLMClient, get_client  # noqa: E402
from scripts.utils.generation_utils import iter_json_objects  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger("ai-dedupe")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def load_dataset(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_dataset(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_text(record: dict) -> str:
    messages = record.get("messages") or []
    if not messages:
        raise ValueError("Registro sem campo messages.")
    text = messages[0].get("text")
    if not isinstance(text, str):
        raise ValueError("Campo text inválido no registro.")
    return text


def set_text(record: dict, value: str) -> None:
    record["messages"][0]["text"] = value


def extract_intents(record: dict) -> list[str]:
    messages = record.get("messages") or []
    if not messages:
        return []
    intents = messages[0].get("intent") or []
    return [str(intent) for intent in intents]


def group_duplicates(records: list[dict]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        key = normalize_text(extract_text(record))
        buckets.setdefault(key, []).append(idx)
    return {key: idxs for key, idxs in buckets.items() if len(idxs) > 1}


@dataclass
class RewriteTask:
    sample_idx: int
    intents: list[str]
    original_text: str
    attempts: int = 0

    def as_prompt_payload(self) -> dict:
        return {
            "id": self.sample_idx,
            "intents": self.intents,
            "text": self.original_text,
        }


def chunked(items: list[RewriteTask], size: int) -> Iterator[list[RewriteTask]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def build_prompt(tasks: list[RewriteTask]) -> str:
    instructions = (
        "Você é um assistente de curadoria de dados. Receberá uma lista JSON contendo entradas duplicadas de um "
        "dataset de classificação de intents. Para cada item, gere uma NOVA variação da frase do usuário em "
        "português, mantendo a mesma intenção e contexto, evitando repetir exatamente a mesma construção. "
        "Retorne um objeto JSON por linha com os campos obrigatórios: id (inteiro) e text (string reescrita). "
        "Não inclua explicações adicionais ou texto fora dos objetos JSON."
    )
    payload = json.dumps([task.as_prompt_payload() for task in tasks], ensure_ascii=False)
    return f"{instructions}\n\nItens:\n{payload}"


def parse_rewrites(raw: str) -> dict[int, str]:
    outputs: dict[int, str] = {}
    for obj in iter_json_objects(raw):
        idx = obj.get("id")
        text = obj.get("text")
        if idx is None or text is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        text_str = str(text).strip()
        if text_str:
            outputs[idx_int] = text_str
    return outputs


def rewrite_batch(
    client: LLMClient,
    tasks: list[RewriteTask],
    batch_retries: int,
) -> dict[int, str]:
    attempt = 0
    while attempt < batch_retries:
        attempt += 1
        try:
            prompt = build_prompt(tasks)
            raw = client.generate(prompt)
            if not raw:
                raise ValueError("Resposta vazia do LLM")
            rewrites = parse_rewrites(raw)
            if not rewrites:
                raise ValueError("Resposta inválida: nenhum JSON retornado.")
            return rewrites
        except Exception as exc:
            LOGGER.warning("Falha ao reescrever lote (%d/%d): %s", attempt, batch_retries, exc)
            if attempt == batch_retries:
                raise
    return {}


def process_duplicates(
    client: LLMClient,
    records: list[dict],
    duplicates: dict[str, list[int]],
    batch_size: int,
    batch_retries: int,
    max_sample_attempts: int,
) -> int:
    targets: list[RewriteTask] = []
    for idxs in duplicates.values():
        for sample_idx in idxs[1:]:
            targets.append(
                RewriteTask(
                    sample_idx=sample_idx,
                    intents=extract_intents(records[sample_idx]),
                    original_text=extract_text(records[sample_idx]),
                )
            )

    LOGGER.info("✍️  Reescrevendo %d amostras duplicadas nesta iteração.", len(targets))

    seen_texts = {normalize_text(extract_text(record)) for record in records}
    resolved = 0

    pending = {task.sample_idx: task for task in targets}

    while pending:
        current_batch = list(pending.values())[:batch_size]
        try:
            rewrites = rewrite_batch(client, current_batch, batch_retries=batch_retries)
        except Exception as exc:
            LOGGER.error("Interrompendo deduplicação: %s", exc)
            raise

        for task in current_batch:
            new_text = rewrites.get(task.sample_idx)
            if not new_text:
                task.attempts += 1
                if task.attempts >= max_sample_attempts:
                    raise RuntimeError(
                        f"Não foi possível reescrever a amostra #{task.sample_idx} após {max_sample_attempts} tentativas."
                    )
                continue

            normalized = normalize_text(new_text)
            if normalized in seen_texts:
                task.attempts += 1
                if task.attempts >= max_sample_attempts:
                    raise RuntimeError(
                        f"Reescrita ainda duplicada para amostra #{task.sample_idx} após {max_sample_attempts} tentativas."
                    )
                continue

            seen_texts.add(normalized)
            set_text(records[task.sample_idx], new_text)
            resolved += 1
            pending.pop(task.sample_idx, None)

    return resolved


def parse_args() -> argparse.Namespace:
    default_dataset = PROJECT_ROOT / "data" / "synthetic_dataset_v2.jsonl"
    parser = argparse.ArgumentParser(
        description="Remove duplicatas reescrevendo frases via Gemini (ou outro LLM compatível)."
    )
    parser.add_argument("--input", type=Path, default=default_dataset, help="Caminho do dataset de entrada.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dataset,
        help="Destino do dataset deduplicado (por padrão, sobrescreve o original).",
    )
    parser.add_argument("--client", choices=["gemini", "openai", "mock"], default="gemini")
    parser.add_argument("--model", type=str, default=None, help="Modelo a ser usado pelo cliente escolhido.")
    parser.add_argument("--batch-size", type=int, default=10, help="Quantas frases reescrever por chamada ao LLM.")
    parser.add_argument("--batch-retries", type=int, default=3, help="Número de tentativas por lote na API.")
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=6,
        help="Quantidade máxima de reescritas para uma mesma frase antes de abortar.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=0,
        help="Limite de iterações completas. Use 0 para repetir até zerar duplicatas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size deve ser > 0")

    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model

    client = get_client(args.client, **client_kwargs)

    records = load_dataset(args.input)
    LOGGER.info("📚 Dataset carregado (%d linhas)", len(records))

    passes = 0
    while True:
        duplicates = group_duplicates(records)
        duplicate_pairs = sum(len(indices) - 1 for indices in duplicates.values())
        ratio = (duplicate_pairs / len(records)) * 100 if records else 0.0

        LOGGER.info("🔍 Iteração %d — duplicatas restantes: %d (%.2f%%)", passes + 1, duplicate_pairs, ratio)
        if duplicate_pairs == 0:
            LOGGER.info("✅ Nenhuma duplicata restante. Processo concluído.")
            break

        process_duplicates(
            client=client,
            records=records,
            duplicates=duplicates,
            batch_size=args.batch_size,
            batch_retries=args.batch_retries,
            max_sample_attempts=args.max_sample_attempts,
        )
        passes += 1

        if args.max_passes and passes >= args.max_passes:
            raise RuntimeError(
                f"Limite de {args.max_passes} iterações alcançado com duplicatas ainda presentes."
            )

    save_dataset(args.output, records)
    LOGGER.info("💾 Dataset deduplicado salvo em %s", args.output)


if __name__ == "__main__":
    main()
