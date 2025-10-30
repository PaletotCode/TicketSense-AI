""""""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError:
    print(
        "❌ Dependências faltando. Execute: pip install python-dotenv tqdm",
        file=sys.stderr
    )
    sys.exit(1)
load_dotenv(PROJECT_ROOT / ".env")
try:
    from scripts.utils.dataset_recipe import DATASET_RECIPE, TOTAL_SAMPLES, GenerationTask
    from scripts.utils.llm_clients import CLIENT_MAP, LLMClient, get_client
    from scripts.utils.generation_utils import (
        build_minimalist_prompt, 
        iter_json_objects,
        validate_sample
    )
except ImportError as e:
    print(f"❌ Erro ao importar módulos locais: {e}", file=sys.stderr)
    print("Certifique-se de que todos os arquivos estão no diretório scripts/", file=sys.stderr)
    sys.exit(1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("dataset_generator")


def parse_args() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(
        description="Gerador de Dataset Sintético para TicketSense-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_dataset.jsonl",
        help="Caminho do arquivo JSONL de saída (default: data/synthetic_dataset.jsonl)",
    )
    
    parser.add_argument(
        "--client",
        type=str,
        choices=CLIENT_MAP.keys(),
        default="mock",
        help="Cliente de LLM a ser usado (default: mock)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Número de amostras a pedir ao LLM por vez (default: 50)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo específico do LLM (ex: gpt-4, gemini-pro)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura para geração (0.0-2.0, default: 0.7)",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Número máximo de tentativas em caso de falha (default: 3)",
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Ativa validação rigorosa dos dados gerados",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logging detalhado (nível DEBUG)",
    )
    
    return parser.parse_args()


def run_generation_pipeline(
    client: LLMClient,
    recipe: list[GenerationTask],
    output_path: Path,
    batch_size: int,
    max_retries: int = 3,
    validate: bool = False,
) -> dict:
    """"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_generated": 0,
        "total_invalid": 0,
        "total_api_calls": 0,
        "total_retries": 0,
    }
    
    LOGGER.info("=" * 80)
    LOGGER.info("🚀 Iniciando geração de %s amostras", TOTAL_SAMPLES)
    LOGGER.info("📁 Arquivo de saída: %s", output_path)
    LOGGER.info("🤖 Cliente: %s", client.__class__.__name__)
    LOGGER.info("📦 Tamanho do lote: %s", batch_size)
    LOGGER.info("=" * 80)
    
    with output_path.open("w", encoding="utf-8") as fout:
        with tqdm(total=TOTAL_SAMPLES, unit=" amostras", desc="Progresso Total") as pbar:
            for task_idx, task in enumerate(recipe, 1):
                task_name = json.dumps(task.intents, ensure_ascii=False)
                LOGGER.info(
                    "\n[Tarefa %d/%d] Iniciando %s: %s amostras...",
                    task_idx,
                    len(recipe),
                    task_name,
                    task.count
                )
                
                num_batches = math.ceil(task.count / batch_size)
                generated_for_task = 0
                for batch_idx in range(num_batches):
                    remaining = task.count - generated_for_task
                    current_batch_size = min(batch_size, remaining)
                    
                    if current_batch_size <= 0:
                        continue
                    
                    LOGGER.debug(
                        "  Lote %d/%d: Solicitando %d amostras...",
                        batch_idx + 1,
                        num_batches,
                        current_batch_size
                    )
                    success = False
                    for retry in range(max_retries):
                        try:
                            prompt = build_minimalist_prompt(task.intents, current_batch_size)
                            raw_response = client.generate(prompt)
                            stats["total_api_calls"] += 1
                            
                            if not raw_response:
                                raise ValueError("Resposta vazia do LLM")
                            batch_count = 0
                            invalid_count = 0
                            
                            for valid_json in iter_json_objects(raw_response):
                                if validate:
                                    if not validate_sample(valid_json, task.intents):
                                        invalid_count += 1
                                        LOGGER.warning(
                                            "Amostra inválida (intenções não correspondem): %s",
                                            str(valid_json)[:100]
                                        )
                                        continue
                                if "messages" in valid_json and isinstance(valid_json["messages"], list):
                                    if len(valid_json["messages"]) > 0:
                                        fout.write(json.dumps(valid_json, ensure_ascii=False) + "\n")
                                        batch_count += 1
                                else:
                                    invalid_count += 1
                                    LOGGER.warning(
                                        "Formato JSON inválido: %s",
                                        str(valid_json)[:100]
                                    )
                            
                            stats["total_invalid"] += invalid_count
                            
                            if batch_count == 0:
                                raise ValueError("Nenhuma amostra válida gerada")
                            
                            LOGGER.debug(
                                "  Lote %d/%d: ✓ %d válidas, %d inválidas",
                                batch_idx + 1,
                                num_batches,
                                batch_count,
                                invalid_count
                            )
                            stats["total_generated"] += batch_count
                            generated_for_task += batch_count
                            pbar.update(batch_count)
                            
                            success = True
                            break  # Sai do loop de retry
                            
                        except Exception as e:
                            stats["total_retries"] += 1
                            LOGGER.warning(
                                "  Tentativa %d/%d falhou: %s",
                                retry + 1,
                                max_retries,
                                str(e)
                            )
                            
                            if retry == max_retries - 1:
                                LOGGER.error(
                                    "  ❌ Lote falhou após %d tentativas. Pulando...",
                                    max_retries
                                )
                    
                    if not success:
                        LOGGER.warning("  ⚠️  Lote não gerado. Continuando...")
                
                LOGGER.info(
                    "[Tarefa %d/%d] Concluída: %d/%d amostras geradas",
                    task_idx,
                    len(recipe),
                    generated_for_task,
                    task.count
                )

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("✅ Geração concluída!")
    LOGGER.info("📊 Estatísticas:")
    LOGGER.info("   • Amostras geradas: %d", stats["total_generated"])
    LOGGER.info("   • Amostras inválidas: %d", stats["total_invalid"])
    LOGGER.info("   • Chamadas à API: %d", stats["total_api_calls"])
    LOGGER.info("   • Tentativas de retry: %d", stats["total_retries"])
    LOGGER.info("💾 Arquivo salvo em: %s", output_path)
    LOGGER.info("=" * 80)
    
    return stats


def main():
    """"""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)
    
    try:
        client_kwargs = {}
        if args.model:
            client_kwargs["model"] = args.model
        if args.temperature:
            client_kwargs["temperature"] = args.temperature
        LOGGER.info("Inicializando cliente %s...", args.client)
        client = get_client(args.client, **client_kwargs)
        stats = run_generation_pipeline(
            client=client,
            recipe=DATASET_RECIPE,
            output_path=args.output,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            validate=args.validate,
        )
        success_rate = (stats["total_generated"] / TOTAL_SAMPLES) * 100
        if success_rate < 90:
            LOGGER.warning(
                "⚠️  Taxa de sucesso baixa: %.1f%% (%d/%d)",
                success_rate,
                stats["total_generated"],
                TOTAL_SAMPLES
            )
            sys.exit(1)
        
        LOGGER.info("🎉 Pipeline executado com sucesso!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        LOGGER.warning("\n⚠️  Geração interrompida pelo usuário")
        sys.exit(130)
    except Exception as e:
        LOGGER.error("❌ Erro fatal: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()