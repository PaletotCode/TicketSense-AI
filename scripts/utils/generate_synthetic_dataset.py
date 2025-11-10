from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

# `__file__` está em scripts/utils/, então o raiz fica dois níveis acima.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError as exc:
    print("❌ Dependências faltando. Execute: pip install python-dotenv tqdm", file=sys.stderr)
    raise

load_dotenv(PROJECT_ROOT / ".env")

try:
    from scripts.utils.dataset_recipe import DATASET_RECIPE, TOTAL_SAMPLES, GenerationTask
    from scripts.utils.llm_clients import get_client, LLMClient
    from scripts.utils.generation_utils import build_minimalist_prompt, iter_json_objects, validate_sample
except ImportError:
    # fallback para execução dentro de scripts/
    from utils.dataset_recipe import DATASET_RECIPE, TOTAL_SAMPLES, GenerationTask  # type: ignore
    from utils.llm_clients import get_client, LLMClient  # type: ignore
    from utils.generation_utils import build_minimalist_prompt, iter_json_objects, validate_sample  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("dataset_generator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerador de Dataset Sintético para Intent Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "synthetic_dataset.jsonl")
    parser.add_argument("--client", type=str, choices=["mock", "openai", "gemini"], default="gemini")
    parser.add_argument("--model", type=str, default=None, help="Override de modelo para o cliente escolhido")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continua a geração aproveitando arquivo existente")
    return parser.parse_args()


def load_existing_counts(path: Path) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            intents = obj.get("messages", [{}])[0].get("intent")
            if isinstance(intents, list):
                key = tuple(sorted(map(str, intents)))
                counts[key] = counts.get(key, 0) + 1
    return counts


def compute_recipe_to_run(resume: bool, output_path: Path, recipe: list[GenerationTask]) -> list[GenerationTask]:
    if not resume:
        return recipe
    existing = load_existing_counts(output_path)
    remaining: list[GenerationTask] = []
    for task in recipe:
        key = tuple(sorted(task.intents))
        generated = existing.get(key, 0)
        left = max(0, task.count - generated)
        if left > 0:
            remaining.append(GenerationTask(intents=task.intents, count=left))
    LOGGER.info(
        "Resume ativado: %d tarefas restantes (arquivo possuía %d linhas)",
        len(remaining),
        sum(existing.values()),
    )
    return remaining


def run_generation_pipeline(
    client: LLMClient,
    recipe: list[GenerationTask],
    output_path: Path,
    batch_size: int,
    max_retries: int,
    validate: bool,
    append_mode: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_generated": 0,
        "total_invalid": 0,
        "total_api_calls": 0,
        "total_retries": 0,
    }

    total_to_generate = sum(task.count for task in recipe)
    if total_to_generate == 0:
        LOGGER.warning("Nada para gerar. Verifique se --resume rescindiu toda a receita.")
        return stats

    mode = "a" if append_mode else "w"

    LOGGER.info("=" * 100)
    LOGGER.info("🚀 Iniciando geração de %s amostras", total_to_generate)
    LOGGER.info("📁 Arquivo de saída: %s", output_path)
    LOGGER.info("🤖 Cliente: %s", client.__class__.__name__)
    LOGGER.info("📦 Batch size: %s", batch_size)
    LOGGER.info("=" * 100)

    with output_path.open(mode, encoding="utf-8") as fout:
        with tqdm(total=total_to_generate, unit=" amostras", desc="Progresso Total") as pbar:
            for task_idx, task in enumerate(recipe, start=1):
                LOGGER.info(
                    "\n[Tarefa %d/%d] Intents %s: %d amostras",
                    task_idx,
                    len(recipe),
                    json.dumps(task.intents, ensure_ascii=False),
                    task.count,
                )
                generated_for_task = 0
                total_batches = math.ceil(task.count / batch_size)  # usado apenas para logging
                batch_number = 0

                while generated_for_task < task.count:
                    batch_number += 1
                    remaining = task.count - generated_for_task
                    current_batch = min(batch_size, remaining)
                    LOGGER.debug(
                        "  Lote %d/%d: solicitando %d amostras (faltam %d)",
                        batch_number,
                        total_batches,
                        current_batch,
                        remaining,
                    )

                    attempt = 0
                    success = False
                    while attempt < max_retries:
                        attempt += 1
                        try:
                            prompt = build_minimalist_prompt(task.intents, current_batch)
                            raw = client.generate(prompt)
                            stats["total_api_calls"] += 1
                            if not raw:
                                raise ValueError("Resposta vazia do LLM")

                            batch_valid = 0
                            invalid = 0

                            for obj in iter_json_objects(raw):
                                if validate and not validate_sample(obj, task.intents):
                                    invalid += 1
                                    LOGGER.debug("    Amostra inválida: %s", str(obj)[:120])
                                    continue
                                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                batch_valid += 1

                            stats["total_invalid"] += invalid
                            if batch_valid == 0:
                                raise ValueError("Nenhuma amostra válida no lote.")

                            generated_for_task += batch_valid
                            stats["total_generated"] += batch_valid
                            pbar.update(batch_valid)
                            LOGGER.debug(
                                "  Lote %d concluído: %d válidas, %d inválidas (acumulado tarefa: %d/%d)",
                                batch_number,
                                batch_valid,
                                invalid,
                                generated_for_task,
                                task.count,
                            )
                            success = True
                            break
                        except Exception as exc:
                            stats["total_retries"] += 1
                            LOGGER.warning(
                                "  Tentativa %d/%d falhou: %s",
                                attempt,
                                max_retries,
                                exc,
                            )
                            if attempt == max_retries:
                                LOGGER.error(
                                    "  ❌ Lote abortado após %d tentativas; %d amostras ainda faltam para esta tarefa.",
                                    max_retries,
                                    task.count - generated_for_task,
                                )
                    if not success:
                        break

                LOGGER.info("[Tarefa %d/%d] finalizada: %d/%d amostras.", task_idx, len(recipe), generated_for_task, task.count)

    LOGGER.info("\n" + "=" * 100)
    LOGGER.info("✅ Geração concluída.")
    LOGGER.info("📊 Total gerado: %s", stats["total_generated"])
    LOGGER.info("📊 Inválidas: %s", stats["total_invalid"])
    LOGGER.info("📊 Chamadas de API: %s", stats["total_api_calls"])
    LOGGER.info("📊 Tentativas extras: %s", stats["total_retries"])
    LOGGER.info("💾 Arquivo salvo em: %s", output_path)
    LOGGER.info("=" * 100)
    return stats


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)

    client_kwargs = {}
    if args.model:
        client_kwargs["model"] = args.model
    if args.temperature:
        client_kwargs["temperature"] = args.temperature

    client = get_client(args.client, **client_kwargs)
    recipe = compute_recipe_to_run(args.resume, args.output, DATASET_RECIPE)
    append = args.resume and args.output.exists()
    run_generation_pipeline(
        client=client,
        recipe=recipe,
        output_path=args.output,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        validate=args.validate,
        append_mode=append,
    )


if __name__ == "__main__":
    main()
