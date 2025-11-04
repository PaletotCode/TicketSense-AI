from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import torch
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.admin import router as admin_router
from api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    IntentPrediction,
    ModelInfo,
    PredictionRequest,
)
from config.config import DEVICE, api_config, gcs_config, model_config, training_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self) -> None:
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer = None
        self.id_to_intent: Dict[int, str] | None = None
        self.intent_to_id: Dict[str, int] | None = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            model_path = training_config.best_model_dir
            if not model_path.exists():
                logger.warning("⚠️  Modelo não encontrado em %s", model_path)
                self._download_model_from_gcs()

            logger.info("📂 Carregando modelo de %s", model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            label_map_path = model_path / "label_map.json"
            if not label_map_path.exists():
                raise FileNotFoundError(f"label_map.json não encontrado em {label_map_path}")
            with label_map_path.open("r", encoding="utf-8") as f:
                label_map = json.load(f)
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

            logger.info("✅ Modelo carregado! total de intents: %s | device: %s", num_labels, DEVICE)
        except Exception as exc:
            logger.error("❌ Erro ao carregar modelo: %s", exc, exc_info=True)
            raise

    def _download_model_from_gcs(self) -> None:
        if not gcs_config.credentials_path and not gcs_config.model_path:
            raise RuntimeError("Sem modelo local e sem configuração GCS válida.")
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(gcs_config.bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_config.model_path)
        target = training_config.best_model_dir
        target.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = blob.name.replace(f"{gcs_config.model_path}/", "")
            local = target / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local))
            logger.info("  ↘︎ %s", rel)

    @torch.no_grad()
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        if not self.model_loaded or not self.model or not self.tokenizer:
            raise RuntimeError("Modelo não carregado.")

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

        ordered: List[tuple[str, float]] = sorted(
            ((self.id_to_intent[i], float(probabilities[i].item())) for i in range(len(probabilities))),
            key=lambda item: item[1],
            reverse=True,
        )

        threshold = 0.35
        selected: List[tuple[str, float]] = []
        for idx, (label, score) in enumerate(ordered[:3]):
            if idx == 0 or score >= threshold:
                selected.append((label, score))

        if not selected and ordered:
            selected.append(ordered[0])

        top_label, top_score = selected[0]
        result: Dict[str, Any] = {
            "intent": top_label,
            "confidence": top_score,
            "intents": [{"label": label, "score": score} for label, score in selected],
        }

        if return_probabilities:
            result["all_probabilities"] = {label: score for label, score in ordered}

        return result


app = FastAPI(
    title="TicketSense-AI",
    description="API de classificação multi-intenção",
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
app.include_router(admin_router)


@app.on_event("startup")
async def startup_event() -> None:
    global model_inference
    logger.info("=" * 60)
    logger.info("🚀 TicketSense-AI iniciando...")
    logger.info("=" * 60)
    try:
        model_inference = ModelInference()
        logger.info("✅ modelo pronto para inferência")
    except Exception as exc:
        logger.error("❌ Falha ao carregar modelo: %s", exc)
        model_inference = None
    logger.info("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model_inference and model_inference.model_loaded else "unhealthy",
        device=str(DEVICE),
    )


@app.get("/model_info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    if not model_inference or not model_inference.model_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelo não carregado")
    return ModelInfo(
        model_name=model_config.model_name,
        intents=model_inference.intent_to_id,  # type: ignore[arg-type]
        max_length=model_config.max_seq_length,
        device=str(DEVICE),
    )


@app.post("/predict_intent", response_model=IntentPrediction)
async def predict_intent(payload: PredictionRequest = Body(...)) -> IntentPrediction:
    if not model_inference:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelo não carregado")
    result = model_inference.predict(payload.text, return_probabilities=payload.return_probabilities)
    return IntentPrediction(**result)


@app.post("/predict_batch")
async def predict_batch(payload: BatchPredictionRequest = Body(...)) -> List[IntentPrediction]:
    if not model_inference:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelo não carregado")
    predictions = []
    for item in payload.items:
        result = model_inference.predict(item.text, return_probabilities=item.return_probabilities)
        predictions.append(IntentPrediction(**result))
    return predictions
