"""
Define a "Receita" para a geração do dataset sintético.
Basta editar a lista DATASET_RECIPE para controlar o que será gerado.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class GenerationTask:
    """Define um lote de geração para uma combinação de intenções."""
    intents: List[str]  # A lista de intenções (ex: ["PAYMENT"])
    count: int          # Quantas amostras gerar (ex: 500)
    
    def __post_init__(self):
        """Valida os parâmetros da tarefa."""
        if not self.intents:
            raise ValueError("A lista de intenções não pode estar vazia")
        if self.count <= 0:
            raise ValueError(f"Count deve ser positivo, recebeu {self.count}")
        if not all(isinstance(intent, str) for intent in self.intents):
            raise ValueError("Todas as intenções devem ser strings")


# --- A RECEITA DO DATASET ---
# O desenvolvedor edita esta lista para definir o dataset de 5k+ amostras.
DATASET_RECIPE: List[GenerationTask] = [
    # Bloco 1: Fundações (Intenções Únicas)
    # Estas são as intenções principais que o modelo precisa reconhecer
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

    # Bloco 2: Combinações Comuns (Pares)
    # Casos onde usuários expressam duas intenções simultaneamente
    GenerationTask(intents=["BILLING", "COMPLAINT"], count=300),
    GenerationTask(intents=["TECHNICAL_ISSUE", "SUPPORT"], count=300),
    GenerationTask(intents=["PAYMENT", "TECHNICAL_ISSUE"], count=200),
    GenerationTask(intents=["CANCELLATION", "COMPLAINT"], count=200),
    GenerationTask(intents=["UPGRADE", "INFORMATION"], count=150),
    GenerationTask(intents=["PAYMENT", "BILLING"], count=150),

    # Bloco 3: Combinações Complexas (Robustez)
    # Casos raros mas importantes para a robustez do modelo
    GenerationTask(intents=["PAYMENT", "BILLING", "COMPLAINT"], count=100),
    GenerationTask(intents=["TECHNICAL_ISSUE", "SUPPORT", "COMPLAINT"], count=100),
]

# Calcula o total de amostras automaticamente
TOTAL_SAMPLES = sum(task.count for task in DATASET_RECIPE)

# Validação da receita ao importar o módulo
def validate_recipe():
    """Valida a integridade da receita de dataset."""
    if not DATASET_RECIPE:
        raise ValueError("DATASET_RECIPE não pode estar vazia")
    
    # Verifica se há duplicatas
    seen = set()
    for task in DATASET_RECIPE:
        key = tuple(sorted(task.intents))
        if key in seen:
            raise ValueError(f"Combinação duplicada encontrada: {task.intents}")
        seen.add(key)
    
    print(f"✓ Receita validada: {len(DATASET_RECIPE)} tarefas, {TOTAL_SAMPLES} amostras totais")

# Executa a validação ao importar
validate_recipe()