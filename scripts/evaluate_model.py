#!/usr/bin/env python3
"""
Avaliador das intenções usando a API local.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import httpx


INTENTS = [
    "LEAD_INTENT",
    "PAYMENT",
    "SUPPORT",
    "GREETING",
    "TECHNICAL_ISSUE",
    "BILLING",
    "CANCELLATION",
    "UPGRADE",
    "INFORMATION",
    "COMPLAINT",
    "OTHER",
]


def _pairs(texts: Iterable[str], intent: str) -> List[Tuple[str, List[str]]]:
    return [(text, [intent]) for text in texts]


def build_eval_set() -> List[Tuple[str, List[str]]]:
    data: List[Tuple[str, List[str]]] = []

    data += _pairs(
        [
            "Quero falar com o time comercial para contratar",
            "Quanto custa o plano empresarial se eu fechar hoje?",
            "Podemos agendar uma demonstração ainda essa semana?",
            "Me liga, quero assinar o premium agora",
            "Tenho budget aprovado e preciso da proposta final",
        ],
        "LEAD_INTENT",
    )
    data += _pairs(
        [
            "Quero pagar meu boleto atrasado hoje",
            "Consegue parcelar essa fatura?",
            "Paguei mas ainda aparece em aberto",
            "Link de pagamento não funciona",
            "Qual a taxa para antecipar?",
        ],
        "PAYMENT",
    )
    data += _pairs(
        [
            "Preciso falar com alguém do suporte",
            "Me ajuda a configurar minha conta",
            "Onde troco a senha?",
            "Quais canais de atendimento vocês têm?",
            "Não encontrei a opção no app, podem orientar?",
        ],
        "SUPPORT",
    )
    data += _pairs(
        [
            "Olá, bom dia!",
            "Oi, tudo bem?",
            "Boa noite, equipe",
            "E aí pessoal",
            "Oi, alguém online?",
        ],
        "GREETING",
    )
    data += _pairs(
        [
            "Minha internet caiu de novo",
            "O app fecha sozinho ao abrir",
            "Erro 500 ao acessar minha conta",
            "Não estou recebendo o código de verificação",
            "O site está muito lento hoje",
        ],
        "TECHNICAL_ISSUE",
    )
    data += _pairs(
        [
            "Veio uma cobrança duplicada",
            "Quero nota fiscal da última fatura",
            "Qual o vencimento da minha conta?",
            "Tem tarifa extra nesse mês?",
            "Quero falar com o setor de cobrança",
        ],
        "BILLING",
    )
    data += _pairs(
        [
            "Quero cancelar minha assinatura",
            "Como encerrar o plano atual?",
            "Preciso cancelar hoje, por favor",
            "Encerrar contrato sem multa é possível?",
            "Não quero continuar, cancelem",
        ],
        "CANCELLATION",
    )
    data += _pairs(
        [
            "Quero migrar para o plano premium",
            "Como faço upgrade para empresarial?",
            "Tem plano com mais recursos?",
            "Quero aumentar meu pacote",
            "Existe opção de plano anual?",
        ],
        "UPGRADE",
    )
    data += _pairs(
        [
            "Vocês atendem no sábado?",
            "Qual o prazo de confirmação?",
            "Onde vejo meu histórico?",
            "Tem integração com ERP?",
            "Qual a política de privacidade?",
        ],
        "INFORMATION",
    )
    data += _pairs(
        [
            "Estou insatisfeito com o atendimento",
            "Fiz reclamação e ninguém respondeu",
            "Isso é um absurdo",
            "Quero registrar uma queixa formal",
            "Muito decepcionado com o serviço",
        ],
        "COMPLAINT",
    )
    data += _pairs(
        [
            "Obrigado!",
            "Beleza, valeu",
            "Sem intenção específica",
            "Tô só testando aqui",
            "Não sei explicar direito",
        ],
        "OTHER",
    )

    combos = [
        ("Tenho interesse em comprar o plano premium, qual o preço?", ["LEAD_INTENT", "UPGRADE", "INFORMATION"]),
        ("Quero assinar hoje, enviem o link de pagamento", ["LEAD_INTENT", "PAYMENT"]),
        ("Boa tarde, poderiam me explicar os planos premium?", ["GREETING", "INFORMATION", "UPGRADE"]),
        ("Fatura duplicada e quero reembolso", ["BILLING", "PAYMENT", "COMPLAINT"]),
        ("Já paguei e consta em aberto, preciso de ajuda", ["PAYMENT", "SUPPORT", "BILLING"]),
        ("Internet caiu, estou irritado", ["TECHNICAL_ISSUE", "COMPLAINT"]),
        ("Quero cancelar e migrar para outro plano", ["CANCELLATION", "UPGRADE"]),
        ("Quero negociar boleto em atraso com o financeiro", ["PAYMENT", "BILLING"]),
        ("O app não abre e preciso falar com alguém", ["TECHNICAL_ISSUE", "SUPPORT"]),
        ("Quero cancelar, estou insatisfeito", ["CANCELLATION", "COMPLAINT"]),
        ("Quero saber taxas e link para pagar", ["INFORMATION", "PAYMENT"]),
        ("Recebi atendimento ruim e duas cobranças", ["COMPLAINT", "BILLING"]),
    ]
    data += combos
    return data


def topk_hits(probs: Dict[str, float], gold: Sequence[str], k: int) -> bool:
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top = [lbl for lbl, _ in ordered[:k]]
    return any(lbl in gold for lbl in top)


def threshold_labels(probs: Dict[str, float], thresh: float) -> List[str]:
    return [lbl for lbl, score in probs.items() if score >= thresh]


@dataclass
class Scores:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-9)

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-9)

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-9)


def evaluate(endpoint: str, threshold: float, topk: int) -> None:
    samples = build_eval_set()
    client = httpx.Client(timeout=20.0)

    total = len(samples)
    subset_ok = 0
    any_ok = 0
    topk_ok = 0

    per_label: Dict[str, Scores] = {lbl: Scores() for lbl in INTENTS}
    confusions = collections.Counter()
    hard_examples: List[Tuple[str, List[str], Dict[str, float]]] = []

    for text, gold in samples:
        resp = client.post(
            f"{endpoint.rstrip('/')}/predict_intent",
            json={"text": text, "return_probabilities": True},
        )
        resp.raise_for_status()
        payload = resp.json()
        probs = payload.get("all_probabilities") or {}
        if not probs and payload.get("intent"):
            probs = {payload["intent"]: payload.get("confidence", 0.0)}

        predicted = threshold_labels(probs, threshold)
        if not predicted and probs:
            predicted = [max(probs.items(), key=lambda x: x[1])[0]]

        if set(predicted) == set(gold):
            subset_ok += 1
        if any(lbl in gold for lbl in predicted):
            any_ok += 1
        if probs and topk_hits(probs, gold, topk):
            topk_ok += 1

        if probs:
            top1 = max(probs.items(), key=lambda x: x[1])[0]
            if top1 not in gold:
                confusions[(tuple(sorted(gold)), top1)] += 1
                hard_examples.append((text, gold, probs))

        for label in INTENTS:
            true = int(label in gold)
            pred = int(label in predicted)
            if true and pred:
                per_label[label].tp += 1
            elif not true and pred:
                per_label[label].fp += 1
            elif true and not pred:
                per_label[label].fn += 1

    print("\n=== Resultados Gerais ===")
    print(f"Amostras: {total}")
    print(f"Subset accuracy (exato): {subset_ok / total:.3f}")
    print(f"Top-1 acerta alguma gold: {any_ok / total:.3f}")
    print(f"Hit@{topk}: {topk_ok / total:.3f}")

    micro = Scores()
    for score in per_label.values():
        micro.tp += score.tp
        micro.fp += score.fp
        micro.fn += score.fn
    print("\nMicro-métricas (multi-label via threshold):")
    print(f"Precision: {micro.precision():.3f} | Recall: {micro.recall():.3f} | F1: {micro.f1():.3f}")

    print("\nPor intenção (P/R/F1):")
    for label in INTENTS:
        score = per_label[label]
        print(
            f"- {label:<16} P {score.precision():.3f}  R {score.recall():.3f}  F1 {score.f1():.3f} "
            f"(tp={score.tp}, fp={score.fp}, fn={score.fn})"
        )

    print("\nPrincipais confusões (gold -> predito):")
    for (gold_set, pred), count in confusions.most_common(10):
        print(f"- {list(gold_set)} → {pred}: {count}")

    if hard_examples:
        print("\nExemplos difíceis (top-5):")
        for text, gold, probs in hard_examples[:5]:
            ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            pretty = [(lbl, round(score, 3)) for lbl, score in ordered]
            print("*", text)
            print("  gold:", gold)
            print("  top:", pretty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o classificador via API.")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.endpoint, args.threshold, args.topk)


if __name__ == "__main__":
    main()
