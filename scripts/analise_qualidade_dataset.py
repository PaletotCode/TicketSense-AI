from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def carregar_registros(caminho: Path) -> List[Dict[str, Any]]:
    registros: List[Dict[str, Any]] = []
    with caminho.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Linha inválida em {caminho}: {exc}") from exc
    if not registros:
        raise ValueError(f"Nenhum registro válido encontrado em {caminho}")
    return registros


def extrair_mensagens(registros: Iterable[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    textos: List[str] = []
    intents: List[List[str]] = []
    for registro in registros:
        for mensagem in registro.get("messages", []):
            if mensagem.get("role") != "user":
                continue
            texto = (mensagem.get("text") or "").strip()
            rotulos = mensagem.get("intent") or []
            if not texto or not rotulos:
                continue
            textos.append(texto)
            intents.append([str(r) for r in rotulos])
    if not textos:
        raise ValueError("Nenhuma mensagem de usuário válida encontrada.")
    return textos, intents


def analisar_variabilidade(intents: List[List[str]]) -> Dict[str, Any]:
    contador = Counter(rotulo for grupo in intents for rotulo in grupo)
    total = sum(contador.values())
    proporcoes = {rotulo: contador[rotulo] / total for rotulo in contador}
    multi_intent_ratio = sum(1 for grupo in intents if len(grupo) > 1) / len(intents)
    return {
        "quantidade_por_intent": dict(contador),
        "proporcao_por_intent": proporcoes,
        "multi_intent_ratio": multi_intent_ratio,
    }


def detectar_duplicatas(textos: List[str], intents: List[List[str]]) -> Dict[str, Any]:
    pares = Counter(zip(textos, map(tuple, intents)))
    duplicadas = {texto[:60] + "…": cont for (texto, _), cont in pares.items() if cont > 1}
    impacto: Dict[str, int] = defaultdict(int)
    for (texto, rotulos), cont in pares.items():
        if cont > 1:
            for rotulo in rotulos:
                impacto[rotulo] += cont - 1
    return {
        "total_duplicatas": sum(cont - 1 for cont in pares.values() if cont > 1),
        "taxa_duplicadas": 1 - (len(pares) / len(textos)),
        "exemplos": duplicadas,
        "impacto_por_intent": dict(impacto),
    }


def analisar_comprimento(textos: List[str]) -> Dict[str, Any]:
    comprimentos = [len(texto.split()) for texto in textos]
    comprimentos.sort()
    return {
        "media_tokens": statistics.mean(comprimentos),
        "mediana_tokens": statistics.median(comprimentos),
        "desvio_padrao_tokens": statistics.pstdev(comprimentos),
        "percentil_10": comprimentos[int(len(comprimentos) * 0.10)],
        "percentil_90": comprimentos[int(len(comprimentos) * 0.90) - 1],
        "min_tokens": comprimentos[0],
        "max_tokens": comprimentos[-1],
    }


def identificar_baixa_qualidade(textos: List[str], intents: List[List[str]], min_tokens: int) -> Dict[str, Any]:
    baixa = []
    for texto, rotulos in zip(textos, intents):
        tokens = len(texto.split())
        if tokens < min_tokens or texto.isupper():
            baixa.append({"texto": texto, "tokens": tokens, "intents": rotulos})
    return {
        "total": len(baixa),
        "percentual": (len(baixa) / len(textos)) * 100,
        "amostras": baixa[:20],
    }


def exportar_csv(caminho: Path, variabilidade: Dict[str, Any], comprimento: Dict[str, Any]) -> None:
    import csv

    with caminho.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metrica", "valor"])
        for intent, qtd in variabilidade["quantidade_por_intent"].items():
            writer.writerow([f"intent_count::{intent}", qtd])
        for intent, share in variabilidade["proporcao_por_intent"].items():
            writer.writerow([f"intent_share::{intent}", share])
        for chave, valor in comprimento.items():
            writer.writerow([f"length::{chave}", valor])


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de qualidade de dataset JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--saida-relatorio", type=Path)
    parser.add_argument("--exportar-csv", type=Path)
    parser.add_argument("--min-tokens", type=int, default=3)
    args = parser.parse_args()

    registros = carregar_registros(args.input)
    textos, intents = extrair_mensagens(registros)

    variabilidade = analisar_variabilidade(intents)
    duplicatas = detectar_duplicatas(textos, intents)
    comprimento = analisar_comprimento(textos)
    baixa_qualidade = identificar_baixa_qualidade(textos, intents, args.min_tokens)

    relatorio = {
        "arquivo": str(args.input),
        "total_mensagens": len(textos),
        "variabilidade": variabilidade,
        "duplicatas": duplicatas,
        "comprimento": comprimento,
        "baixa_qualidade": baixa_qualidade,
    }

    print("=== Relatório de Qualidade ===")
    print(f"Arquivo: {relatorio['arquivo']}")
    print(f"Total de mensagens válidas: {relatorio['total_mensagens']}")
    print("\nDistribuição por intent:")
    for intent, qtd in variabilidade["quantidade_por_intent"].items():
        share = variabilidade["proporcao_por_intent"][intent] * 100
        print(f"  - {intent}: {qtd} ({share:.2f}%)")
    print(f"\nRatio multi-intent: {variabilidade['multi_intent_ratio']:.2%}")
    print(f"Duplicatas totais: {duplicatas['total_duplicatas']} (taxa {duplicatas['taxa_duplicadas']:.2%})")
    print(f"Comprimento médio (tokens): {comprimento['media_tokens']:.2f}")
    print(f"Comprimento mediano: {comprimento['mediana_tokens']}")
    print(f"Amostras consideradas baixa qualidade: {baixa_qualidade['total']} ({baixa_qualidade['percentual']:.2f}%)")

    if args.saida_relatorio:
        args.saida_relatorio.parent.mkdir(parents=True, exist_ok=True)
        args.saida_relatorio.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRelatório salvo em: {args.saida_relatorio}")

    if args.exportar_csv:
        args.exportar_csv.parent.mkdir(parents=True, exist_ok=True)
        exportar_csv(args.exportar_csv, variabilidade, comprimento)
        print(f"CSV salvo em: {args.exportar_csv}")


if __name__ == "__main__":
    main()
