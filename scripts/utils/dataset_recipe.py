""""""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class GenerationTask:
    """"""
    intents: List[str]  # A lista de intenções (ex: ["PAYMENT"])
    count: int          # Quantas amostras gerar (ex: 500)
    
    def __post_init__(self):
        """"""
        if not self.intents:
            raise ValueError("A lista de intenções não pode estar vazia")
        if self.count <= 0:
            raise ValueError(f"Count deve ser positivo, recebeu {self.count}")
        if not all(isinstance(intent, str) for intent in self.intents):
            raise ValueError("Todas as intenções devem ser strings")
DATASET_RECIPE: List[GenerationTask] = [
    GenerationTask(intents=["PAYMENT"], count=500),
    GenerationTask(intents=["SUPPORT"], count=500),
    GenerationTask(intents=["GREETING"], count=200),
    GenerationTask(intents=["TECHNICAL_ISSUE"], count=500),
    GenerationTask(intents=["BILLING"], count=500),
    GenerationTask(intents=["CANCELLATION"], count=400),
    GenerationTask(intents=["UPGRADE"], count=300),
    GenerationTask(intents=["INFORMATION"], count=400),
    GenerationTask(intents=["COMPLAINT"], count=400),
    GenerationTask(intents=["OTHER"], count=100),
    GenerationTask(intents=["BILLING", "COMPLAINT"], count=300),
    GenerationTask(intents=["TECHNICAL_ISSUE", "SUPPORT"], count=300),
    GenerationTask(intents=["PAYMENT", "TECHNICAL_ISSUE"], count=200),
    GenerationTask(intents=["CANCELLATION", "COMPLAINT"], count=200),
    GenerationTask(intents=["UPGRADE", "INFORMATION"], count=150),
    GenerationTask(intents=["PAYMENT", "BILLING"], count=150),
    GenerationTask(intents=["PAYMENT", "BILLING", "COMPLAINT"], count=100),
    GenerationTask(intents=["TECHNICAL_ISSUE", "SUPPORT", "COMPLAINT"], count=100),
]
TOTAL_SAMPLES = sum(task.count for task in DATASET_RECIPE)
def validate_recipe():
    """"""
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