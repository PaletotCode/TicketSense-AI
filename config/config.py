"""
Configurações centralizadas do projeto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GCSConfig:
    bucket_name: str = os.getenv("GCS_BUCKET_NAME", "pingfy-dataset")
    dataset_path: str = os.getenv("GCS_DATASET_PATH", "data/synthetic_dataset.jsonl")
    model_path: str = os.getenv("GCS_MODEL_PATH", "models/checkpoints/distilbert-intents")
    credentials_path: str | None = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


@dataclass
class ModelConfig:
    model_name: str = "microsoft/mdeberta-v3-base"
    max_seq_length: int = 128
    dropout: float = 0.1
    intent_classes: List[str] | None = None

    def __post_init__(self) -> None:
        if self.intent_classes is None:
            self.intent_classes = [
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


@dataclass
class TrainingConfig:
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "10"))
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", "500"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.01"))

    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    fp16: bool = False

    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    logging_steps: int = 50

    artifacts_dir: Path = field(default_factory=lambda: Path("./artifacts"))
    local_data_dir: Path = field(default_factory=lambda: Path("./data"))
    checkpoints_dir: Path = field(init=False)
    best_model_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.best_model_dir = self.artifacts_dir / "best_model"
        self.logs_dir = self.artifacts_dir / "logs"
        for directory in [
            self.artifacts_dir,
            self.local_data_dir,
            self.checkpoints_dir,
            self.best_model_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")


gcs_config = GCSConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
api_config = APIConfig()

import torch  # noqa: E402


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
print(f"🚀 Using device: {DEVICE}")
