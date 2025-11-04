SHELL := /bin/bash
PYTHON := source venv/bin/activate && python

DATASET := data/synthetic_dataset_v2.jsonl
MODEL := gemini-2.0-flash-lite
BATCH := 50

.PHONY: env dataset resume validate train eval api quality clean

env:
	python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

dataset:
	$(PYTHON) scripts/utils/generate_synthetic_dataset.py \
		--client gemini \
		--model $(MODEL) \
		--batch-size $(BATCH) \
		--output $(DATASET) \
		--validate \
		--verbose

resume:
	$(PYTHON) scripts/utils/generate_synthetic_dataset.py \
		--client gemini \
		--model $(MODEL) \
		--batch-size $(BATCH) \
		--output $(DATASET) \
		--validate \
		--verbose \
		--resume

validate:
	$(PYTHON) scripts/prepare_dataset.py --dataset $(DATASET) --analysis

quality:
	$(PYTHON) scripts/analise_qualidade_dataset.py \
		--input $(DATASET) \
		--saida-relatorio artifacts/logs/dataset_quality.json \
		--exportar-csv artifacts/logs/dataset_quality.csv

train:
	LOCAL_DATASET_PATH=$(DATASET) $(PYTHON) -m trainer.train

eval:
	$(PYTHON) scripts/evaluate_model.py --threshold 0.4 --topk 3

api:
	$(PYTHON) -m uvicorn api.inference:app --reload --port 8000

clean:
	rm -rf venv __pycache__ artifacts/checkpoints/*
