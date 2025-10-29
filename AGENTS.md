# Repository Guidelines

## Project Structure & Module Organization
`api/` delivers the FastAPI inference server; `trainer/` holds dataset tooling and the DistilBERT fine-tuning flow. Shared dataclasses and defaults live in `config/`, and one-off utilities in `scripts/`. Store validated JSONL data in `data/`, training outputs in `artifacts/`, and optional exports in `models/`. Environment settings come from a root `.env` loaded by `config/config.py`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: create and enter the virtual environment.
- `pip install -r requirements.txt`: install the FastAPI, Transformers, and training dependencies.
- `python scripts/prepare_dataset.py [--upload]`: validate JSONL data, print label stats, optionally push to GCS.
- `python -m trainer.train`: run the full training loop and write artifacts under `artifacts/best_model/`.
- `uvicorn api.inference:app --reload --port 8000`: launch the inference API for local QA.
- `curl -X POST http://localhost:8000/predict_intent -d '{"text": "..."}' -H "Content-Type: application/json"`: smoke-test the prediction endpoint.

## Coding Style & Naming Conventions
Stick to 4-space indentation, type hints, and descriptive English identifiers, mirroring `trainer/` and `config/`. Maintain module docstrings, rely on `logging` instead of prints, and keep UTF-8 files. Extend configuration by editing the appropriate dataclass in `config/config.py` rather than introducing loose constants.

## Testing Guidelines
There is no automated test suite yet. Validate data changes with `python scripts/prepare_dataset.py`, then run a shortened training pass (`NUM_EPOCHS=2 python -m trainer.train`) to ensure the pipeline still converges. For API updates, exercise `/health` and `/predict_intent` with curl or the interactive docs and review `artifacts/logs/training.log` for regressions.

## Commit & Pull Request Guidelines
Adopt the Conventional Commit style already in Git history (`type: summary`, e.g. `docs: atualizar guia de treino`). Keep changes scoped per commit. Pull requests must outline context, solution, and dataset/model effects, link tracking tickets, and attach metrics or response samples. Call out any updates under `artifacts/` or `data/` so reviewers can reproduce.

## Dataset & Model Handling
Keep datasets as JSONL with one conversation and labeled intents per line; run `scripts/format_dataset.py` to normalize new files. Treat `artifacts/best_model/` as the deployable bundle and only trigger the GCS upload in `trainer/model_utils.py` after local validation. Update `.env` credential paths that feed `config/GCSConfig` whenever keys rotate.
