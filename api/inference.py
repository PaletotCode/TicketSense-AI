"""
API FastAPI para inferência de intenções do PingFy_IA.

Endpoints:
- POST /predict_intent: Classifica a intenção de uma mensagem
- POST /predict_batch: Classifica intenções em lote
- GET /health: Health check
- GET /model_info: Informações do modelo
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any

import torch
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    IntentPrediction,
    ModelInfo,
    PredictionRequest,
)
from config.config import DEVICE, api_config, gcs_config, model_config, training_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelInference:
    """Classe responsável por carregar o modelo e realizar inferência."""

    def __init__(self) -> None:
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer = None
        self.id_to_intent: Dict[int, str] | None = None
        self.intent_to_id: Dict[str, int] | None = None
        self.model_loaded = False

        self._load_model()

    def _load_model(self) -> None:
        """Carrega modelo, tokenizer e label map."""
        try:
            model_path = training_config.best_model_dir

            if not model_path.exists():
                logger.warning("⚠️  Model not found at %s", model_path)
                logger.info("📥 Downloading model from GCS...")
                self._download_model_from_gcs()

            logger.info("📂 Loading model from %s", model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            label_map_path = model_path / "label_map.json"
            if not label_map_path.exists():
                raise FileNotFoundError(f"Label map not found at {label_map_path}")

            with label_map_path.open("r", encoding="utf-8") as file:
                label_map = json.load(file)
                self.intent_to_id = label_map["intent_to_id"]
                self.id_to_intent = {int(k): v for k, v in label_map["id_to_intent"].items()}

            num_labels = len(self.id_to_intent)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                id2label=self.id_to_intent,
                label2id=self.intent_to_id,
                dtype=torch.float32,
            ).to(DEVICE)

            self.model.eval()
            self.model_loaded = True

            logger.info("✅ Model loaded successfully!")
            logger.info("📊 Device: %s", DEVICE)
            logger.info("📊 Number of intents: %s", num_labels)

        except Exception as exc:
            logger.error("❌ Error loading model: %s", exc)
            raise

    def _download_model_from_gcs(self) -> None:
        """Realiza download do modelo armazenado no GCS."""
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(gcs_config.bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_config.model_path)

        model_path = training_config.best_model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            relative_path = blob.name.replace(f"{gcs_config.model_path}/", "")
            local_file_path = model_path / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file_path))
            logger.info("  ✓ Downloaded %s", relative_path)

        logger.info("✅ Model downloaded from GCS")

    @torch.no_grad()
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """Classifica a intenção de um único texto."""
        if not self.model_loaded or not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=model_config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(outputs.logits).squeeze(0).detach().cpu()

        predicted_class_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class_id].item())

        result: Dict[str, Any] = {
            "intent": self.id_to_intent[predicted_class_id],  # type: ignore[index]
            "confidence": confidence,
        }

        if return_probabilities:
            all_probs = {
                self.id_to_intent[i]: float(probabilities[i].item())  # type: ignore[index]
                for i in range(len(probabilities))
            }
            result["all_probabilities"] = dict(
                sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            )

        return result


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="PingFy_IA - Intent Classification API",
    description="API para classificação de intenções em mensagens do WhatsApp/Instagram",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_inference: ModelInference | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Inicializa o modelo durante o boot da API."""
    global model_inference

    logger.info("=" * 60)
    logger.info("🚀 PINGFY_IA - API STARTING")
    logger.info("=" * 60)

    try:
        model_inference = ModelInference()
        logger.info("✅ Model loaded and ready for inference")
    except Exception as exc:
        logger.error("❌ Failed to load model: %s", exc)
        logger.warning("⚠️  API will start but predictions will fail")

    logger.info("=" * 60)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """Endpoint de health check."""
    return HealthResponse(
        status="healthy" if model_inference and model_inference.model_loaded else "unhealthy",
        model_loaded=model_inference.model_loaded if model_inference else False,
        device=str(DEVICE),
    )


@app.get("/model_info", response_model=ModelInfo, summary="Model Information")
async def get_model_info() -> ModelInfo:
    """Retorna informações sobre o modelo carregado."""
    if not model_inference or not model_inference.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    return ModelInfo(
        model_name=model_config.model_name,
        num_labels=len(model_inference.id_to_intent),  # type: ignore[arg-type]
        intent_classes=list(model_inference.intent_to_id.keys()),  # type: ignore[arg-type]
        max_seq_length=model_config.max_seq_length,
        device=str(DEVICE),
    )


@app.post(
    "/predict_intent",
    response_model=IntentPrediction,
    summary="Predict Intent",
    status_code=status.HTTP_200_OK,
)
async def predict_intent(request: PredictionRequest) -> IntentPrediction:
    """Classifica a intenção de uma mensagem."""
    if not model_inference or not model_inference.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check /health endpoint.",
        )

    try:
        result = model_inference.predict(
            text=request.text,
            return_probabilities=request.return_probabilities,
        )
        logger.info(
            "✅ Prediction: '%s...' -> %s (%.4f)",
            request.text[:50],
            result["intent"],
            result["confidence"],
        )
        return IntentPrediction(**result)
    except Exception as exc:
        logger.error("❌ Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post(
    "/predict_batch",
    response_model=list[IntentPrediction],
    summary="Predict Intent (Batch)",
)
async def predict_batch(request: BatchPredictionRequest = Body(...)) -> list[IntentPrediction]:
    """Classifica intenções em lote."""
    if not model_inference or not model_inference.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        results = [
            IntentPrediction(**model_inference.predict(text, return_probabilities=False))
            for text in request.texts
        ]
        logger.info("✅ Batch prediction completed: %s messages", len(request.texts))
        return results
    except Exception as exc:
        logger.error("❌ Batch prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.inference:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.reload,
        log_level=api_config.log_level,
        workers=1,
    )
