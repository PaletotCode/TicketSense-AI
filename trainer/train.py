"""
Pipeline de treinamento do modelo de intenções.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from config.config import DEVICE, gcs_config, model_config, training_config
from trainer.dataset_utils import (
    IntentDataset,
    build_label_mappings,
    encode_labels,
    ensure_dataset,
    load_dataset,
    select_indices,
    slice_encodings,
)
from trainer.model_utils import create_model, create_tokenizer, maybe_upload_to_gcs, save_label_map

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOG_FILE = training_config.logs_dir / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("trainer")


def compute_metrics(eval_preds) -> Dict[str, float]:
    logits, labels = eval_preds
    probabilities = torch.sigmoid(torch.from_numpy(logits))
    preds = (probabilities >= 0.5).int()
    for i in range(preds.size(0)):
        if preds[i].sum() == 0:
            top_idx = torch.argmax(probabilities[i])
            preds[i, top_idx] = 1
    preds_np = preds.numpy()
    labels_np = torch.from_numpy(labels).int().numpy()
    subset_accuracy = (preds_np == labels_np).all(axis=1).mean()
    precision = precision_score(labels_np, preds_np, average="micro", zero_division=0)
    recall = recall_score(labels_np, preds_np, average="micro", zero_division=0)
    f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
    return {
        "accuracy": float(subset_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_training() -> Dict[str, float]:
    dataset_path = ensure_dataset(gcs_config=gcs_config, training_config=training_config)
    texts, intent_lists = load_dataset(dataset_path)
    label2id, id2label = build_label_mappings(intent_lists)
    primary_label_ids = [label2id[intents[0]] for intents in intent_lists]
    encoded_labels = encode_labels(intent_lists, label2id)

    tokenizer = create_tokenizer(model_config)
    encodings = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=model_config.max_seq_length,
    )

    train_idx, val_idx, _ = select_indices(
        labels=primary_label_ids,
        train_ratio=training_config.train_split,
        val_ratio=training_config.val_split,
        test_ratio=training_config.test_split,
        seed=training_config.random_seed,
    )

    train_encodings, train_labels = slice_encodings(encodings, encoded_labels, train_idx)
    val_encodings, val_labels = slice_encodings(encodings, encoded_labels, val_idx)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    model = create_model(model_config, label2id=label2id, id2label=id2label)

    output_dir = training_config.checkpoints_dir
    best_model_dir = training_config.best_model_dir
    logging_dir = training_config.logs_dir / "trainer"
    logging_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=training_config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(logging_dir),
        logging_strategy="steps",
        logging_steps=training_config.logging_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps * 4,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],
        seed=training_config.random_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    LOGGER.info("🚀 Iniciando treinamento no device %s", DEVICE)
    train_output = trainer.train()
    LOGGER.info("✅ Treinamento concluído: %s", train_output.metrics)

    eval_metrics = trainer.evaluate()
    LOGGER.info("📊 Métricas de validação: %s", eval_metrics)

    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)
    save_label_map(best_model_dir, label2id=label2id, id2label=id2label)

    maybe_upload_to_gcs(best_model_dir, gcs_config=gcs_config)
    return eval_metrics


def main() -> Dict[str, float]:
    metrics = run_training()
    LOGGER.info("🏁 Pipeline finalizado com métricas: %s", metrics)
    return metrics


if __name__ == "__main__":
    main()
