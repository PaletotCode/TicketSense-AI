from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

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


def write_checkpoint(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def clear_checkpoint(path: Optional[Path]) -> None:
    if path and path.exists():
        path.unlink()


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
    stall_threshold: int,
    on_sample_processed: Optional[Callable[[int, int], None]] = None,
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

    total_targets = len(targets)
    LOGGER.info("✍️  Reescrevendo %d amostras duplicadas nesta iteração.", total_targets)

    progress = tqdm(total=total_targets, desc="Deduplicando", unit=" amostras") if tqdm else None
    seen_texts = {normalize_text(extract_text(record)) for record in records}
    resolved = 0

    pending_dict = {task.sample_idx: task for task in targets}
    queue = deque(targets)

    def should_abort(task: RewriteTask) -> bool:
        if max_sample_attempts > 0:
            return task.attempts >= max_sample_attempts
        if stall_threshold > 0:
            return task.attempts >= stall_threshold
        return False

    def apply_fallback(task: RewriteTask) -> str:
        intents_label = "/".join(task.intents) if task.intents else "INTENT"
        return f"{task.original_text} (variação {task.attempts + 1} - {intents_label})"

    while queue:
        current_batch = [queue.popleft() for _ in range(min(batch_size, len(queue)))]
        # Sincroniza com dict pois a fila pode conter referências antigas
        batch_tasks = [pending_dict[task.sample_idx] for task in current_batch if task.sample_idx in pending_dict]
        if not batch_tasks:
            continue
        try:
            rewrites = rewrite_batch(client, batch_tasks, batch_retries=batch_retries)
        except Exception as exc:
            LOGGER.error("Interrompendo deduplicação: %s", exc)
            if progress:
                progress.close()
            raise

        for task in batch_tasks:
            new_text = rewrites.get(task.sample_idx)
            if not new_text:
                task.attempts += 1
                if should_abort(task):
                    fallback_text = apply_fallback(task)
                    normalized = normalize_text(fallback_text)
                    suffix = 1
                    while normalized in seen_texts:
                        fallback_text = f"{fallback_text} #{suffix}"
                        normalized = normalize_text(fallback_text)
                        suffix += 1
                    set_text(records[task.sample_idx], fallback_text)
                    seen_texts.add(normalized)
                    resolved += 1
                    pending_dict.pop(task.sample_idx, None)
                    LOGGER.warning(
                        "⚠️  Uso de fallback na amostra #%d após %d tentativas sem sucesso.",
                        task.sample_idx,
                        task.attempts,
                    )
                    if progress:
                        progress.update(1)
                    if on_sample_processed:
                        on_sample_processed(resolved, total_targets)
                else:
                    queue.append(task)
                continue

            normalized = normalize_text(new_text)
            if normalized in seen_texts:
                task.attempts += 1
                if should_abort(task):
                    fallback_text = apply_fallback(task)
                    normalized = normalize_text(fallback_text)
                    suffix = 1
                    while normalized in seen_texts:
                        fallback_text = f"{fallback_text} #{suffix}"
                        normalized = normalize_text(fallback_text)
                        suffix += 1
                    set_text(records[task.sample_idx], fallback_text)
                    seen_texts.add(normalized)
                    resolved += 1
                    pending_dict.pop(task.sample_idx, None)
                    LOGGER.warning(
                        "⚠️  Uso de fallback na amostra #%d após %d tentativas duplicadas.",
                        task.sample_idx,
                        task.attempts,
                    )
                    if progress:
                        progress.update(1)
                    if on_sample_processed:
                        on_sample_processed(resolved, total_targets)
                else:
                    queue.append(task)
                continue

            seen_texts.add(normalized)
            set_text(records[task.sample_idx], new_text)
            resolved += 1
            pending_dict.pop(task.sample_idx, None)
            task.attempts = 0
            if progress:
                progress.update(1)
            if on_sample_processed:
                on_sample_processed(resolved, total_targets)

    if progress:
        progress.close()
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
        default=0,
        help="Máximo de tentativas por frase (0 = infinito, usa fallback apenas em casos extremos).",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=0,
        help="Limite de iterações completas. Use 0 para repetir até zerar duplicatas.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua a partir do arquivo de saída parcialmente deduplicado (se existir).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "logs" / "dedupe_checkpoint.json",
        help="Arquivo auxiliar usado para armazenar progresso do dedupe.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Quantidade de frases reescritas antes de salvar automaticamente (0 desativa).",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=15,
        help="Quando --max-sample-attempts=0, número máximo de tentativas antes de aplicar fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size deve ser > 0")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every deve ser >= 0")
    if args.stall_threshold < 0:
        raise ValueError("stall-threshold deve ser >= 0")

    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model

    client = get_client(args.client, **client_kwargs)

    source_path = args.output if args.resume and args.output.exists() else args.input
    if args.resume and args.output.exists():
        LOGGER.info("♻️  Resume habilitado — usando dataset parcial em %s", args.output)
    elif args.resume:
        LOGGER.warning("⚠️  Resume solicitado, mas %s não existe. Carregando %s.", args.output, args.input)
    else:
        LOGGER.info("📥 Iniciando dedupe a partir de %s", source_path)

    records = load_dataset(source_path)
    LOGGER.info("📚 Dataset carregado (%d linhas)", len(records))

    checkpoint_path = args.checkpoint_path
    changes_since_save = 0
    last_snapshot: Optional[tuple[int, int, int]] = None  # (pass_idx, resolved, total)

    def save_with_checkpoint(reason: str, pass_idx: int, resolved: int, total: int) -> None:
        nonlocal changes_since_save
        save_dataset(args.output, records)
        metadata = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "reason": reason,
            "pass_index": pass_idx,
            "resolved_in_pass": resolved,
            "remaining_in_pass": max(total - resolved, 0),
            "output_path": str(args.output),
            "total_records": len(records),
        }
        write_checkpoint(checkpoint_path, metadata)
        changes_since_save = 0

    passes = 0
    while True:
        duplicates = group_duplicates(records)
        duplicate_pairs = sum(len(indices) - 1 for indices in duplicates.values())
        ratio = (duplicate_pairs / len(records)) * 100 if records else 0.0

        LOGGER.info("🔍 Iteração %d — duplicatas restantes: %d (%.2f%%)", passes + 1, duplicate_pairs, ratio)
        if duplicate_pairs == 0:
            LOGGER.info("✅ Nenhuma duplicata restante. Processo concluído.")
            save_dataset(args.output, records)
            clear_checkpoint(checkpoint_path)
            break

        current_pass = passes + 1
        last_snapshot = None

        def on_sample_processed(resolved: int, total: int) -> None:
            nonlocal changes_since_save, last_snapshot
            last_snapshot = (current_pass, resolved, total)
            if args.checkpoint_every == 0:
                return
            changes_since_save += 1
            if changes_since_save >= args.checkpoint_every:
                save_with_checkpoint("autosave", current_pass, resolved, total)

        try:
            process_duplicates(
                client=client,
                records=records,
                duplicates=duplicates,
                batch_size=args.batch_size,
                batch_retries=args.batch_retries,
                max_sample_attempts=args.max_sample_attempts,
                stall_threshold=args.stall_threshold,
                on_sample_processed=on_sample_processed,
            )
        except Exception as exc:
            LOGGER.error("❌ Dedupe interrompido: %s", exc)
            if last_snapshot:
                save_with_checkpoint("error", last_snapshot[0], last_snapshot[1], last_snapshot[2])
            else:
                save_with_checkpoint("error", current_pass, 0, duplicate_pairs)
            raise

        save_with_checkpoint("end_of_pass", current_pass, duplicate_pairs, duplicate_pairs)
        passes += 1
        last_snapshot = None

        if args.max_passes and passes >= args.max_passes:
            raise RuntimeError(
                f"Limite de {args.max_passes} iterações alcançado com duplicatas ainda presentes."
            )

    LOGGER.info("💾 Dataset deduplicado salvo em %s", args.output)


if __name__ == "__main__":
    main()
