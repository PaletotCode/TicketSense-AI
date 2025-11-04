from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GenerationTask:
    intents: List[str]
    count: int

    def __post_init__(self) -> None:
        if not self.intents:
            raise ValueError("A lista de intenções não pode estar vazia")
        if self.count <= 0:
            raise ValueError(f"Count deve ser positivo, recebeu {self.count}")
        if not all(isinstance(intent, str) for intent in self.intents):
            raise ValueError("Todas as intenções devem ser strings")


DATASET_RECIPE: List[GenerationTask] = [
    GenerationTask(["LEAD_INTENT"], 1310),
    GenerationTask(["PAYMENT"], 610),
    GenerationTask(["SUPPORT"], 610),
    GenerationTask(["GREETING"], 200),
    GenerationTask(["TECHNICAL_ISSUE"], 810),
    GenerationTask(["BILLING"], 610),
    GenerationTask(["CANCELLATION"], 500),
    GenerationTask(["UPGRADE"], 600),
    GenerationTask(["INFORMATION"], 710),
    GenerationTask(["COMPLAINT"], 810),
    GenerationTask(["OTHER"], 130),
    GenerationTask(["LEAD_INTENT", "UPGRADE"], 560),
    GenerationTask(["LEAD_INTENT", "INFORMATION"], 520),
    GenerationTask(["LEAD_INTENT", "PAYMENT"], 430),
    GenerationTask(["LEAD_INTENT", "SUPPORT"], 320),
    GenerationTask(["LEAD_INTENT", "UPGRADE", "INFORMATION"], 320),
    GenerationTask(["LEAD_INTENT", "PAYMENT", "INFORMATION"], 280),
    GenerationTask(["LEAD_INTENT", "UPGRADE", "PAYMENT"], 240),
    GenerationTask(["LEAD_INTENT", "SUPPORT", "INFORMATION"], 200),
    GenerationTask(["LEAD_INTENT", "COMPLAINT"], 150),
    GenerationTask(["LEAD_INTENT", "PAYMENT", "SUPPORT"], 190),
    GenerationTask(["PAYMENT", "BILLING"], 420),
    GenerationTask(["PAYMENT", "SUPPORT"], 320),
    GenerationTask(["BILLING", "COMPLAINT"], 380),
    GenerationTask(["INFORMATION", "SUPPORT"], 380),
    GenerationTask(["INFORMATION", "OTHER"], 140),
    GenerationTask(["TECHNICAL_ISSUE", "COMPLAINT"], 520),
    GenerationTask(["TECHNICAL_ISSUE", "SUPPORT"], 530),
    GenerationTask(["TECHNICAL_ISSUE", "SUPPORT", "COMPLAINT"], 420),
    GenerationTask(["TECHNICAL_ISSUE", "SUPPORT", "PAYMENT"], 240),
    GenerationTask(["CANCELLATION", "COMPLAINT"], 320),
    GenerationTask(["CANCELLATION", "UPGRADE"], 200),
    GenerationTask(["PAYMENT", "BILLING", "COMPLAINT"], 260),
    GenerationTask(["PAYMENT", "INFORMATION"], 280),
    GenerationTask(["SUPPORT", "COMPLAINT"], 220),
    GenerationTask(["UPGRADE", "INFORMATION"], 260),
]

TOTAL_SAMPLES = sum(task.count for task in DATASET_RECIPE)


def validate_recipe() -> None:
    if not DATASET_RECIPE:
        raise ValueError("DATASET_RECIPE não pode estar vazia")
    seen = set()
    for task in DATASET_RECIPE:
        key = tuple(sorted(task.intents))
        if key in seen:
            raise ValueError(f"Combinação duplicada encontrada: {task.intents}")
        seen.add(key)
    print(f"✓ Receita validada: {len(DATASET_RECIPE)} tarefas, {TOTAL_SAMPLES} amostras totais")


validate_recipe()
