"""
Esquemas Pydantic utilizados pela API de inferência do PingFy_IA.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Payload para predição unitária."""

    text: str = Field(..., min_length=1, max_length=500, description="Texto da mensagem do usuário")
    return_probabilities: bool = Field(
        default=False,
        description="Retornar probabilidades de todas as classes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Bom dia, o boleto deste mês não chegou no meu email. Poderia reenviar?",
                "return_probabilities": False,
            }
        }
    }


class IntentPrediction(BaseModel):
    """Resposta com a intenção classificada."""

    intent: str = Field(..., description="Intenção predita")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança (0-1)")
    all_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probabilidades de todas as classes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intent": "PAYMENT",
                "confidence": 0.9823,
                "all_probabilities": None,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Payload para predições em lote."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de textos para classificação",
    )


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str
    model_loaded: bool
    device: str


class ModelInfo(BaseModel):
    """Informações sobre o modelo carregado."""

    model_name: str
    num_labels: int
    intent_classes: List[str]
    max_seq_length: int
    device: str
