from __future__ import annotations

import asyncio
import ast
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from sse_starlette.sse import EventSourceResponse  # type: ignore
except ModuleNotFoundError:

    class EventSourceResponse(StreamingResponse):
        def __init__(self, generator: AsyncGenerator[str, None]):
            super().__init__(generator, media_type="text/event-stream")

from config.config import training_config

router = APIRouter(prefix="/admin", tags=["admin"])

TRAINING_LOG_PATH = training_config.logs_dir / "training.log"
DATA_DIR = Path("data")
ARTIFACTS_DIR = training_config.artifacts_dir
BEST_MODEL_DIR = training_config.best_model_dir


class TrainingRequest(BaseModel):
    datasetPath: str
    epochs: int = Field(ge=1, le=50)
    batchSize: int = Field(ge=1, le=128)
    learningRate: float = Field(gt=0)
    notes: str | None = None


class CommandRequest(BaseModel):
    command: str
    args: List[str] | None = None


class TrainingEvent(BaseModel):
    type: str
    message: str | None = None
    metric: Dict[str, Any] | None = None
    status: str | None = None


def _parse_training_history(limit: int | None = None) -> List[Dict[str, Any]]:
    if not TRAINING_LOG_PATH.exists():
        return []
    pattern = re.compile(r"Pipeline finalizado com métricas: (\{.*\})")
    records: List[Dict[str, Any]] = []
    with TRAINING_LOG_PATH.open("r", encoding="utf-8") as log:
        for line in log:
            match = pattern.search(line)
            if not match:
                continue
            try:
                payload = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError):
                continue
            timestamp = line.split(" - ")[0]
            records.append(
                {
                    "timestamp": timestamp,
                    "accuracy": float(payload.get("eval_accuracy", 0.0)),
                    "precision": float(payload.get("eval_precision", 0.0)),
                    "recall": float(payload.get("eval_recall", 0.0)),
                    "f1": float(payload.get("eval_f1", 0.0)),
                    "loss": float(payload.get("eval_loss", 0.0)),
                    "duration": float(payload.get("eval_runtime", 0.0)),
                    "datasetSize": float(payload.get("eval_samples_per_second", 0.0))
                    * float(payload.get("eval_runtime", 0.0)),
                }
            )
    records.sort(key=lambda item: item["timestamp"])
    if limit:
        return records[-limit:]
    return records


def _current_model_name() -> str:
    return BEST_MODEL_DIR.name


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _list_datasets() -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    if DATA_DIR.exists():
        for file in sorted(DATA_DIR.glob("*.jsonl")):
            datasets.append({"name": file.name, "samples": _count_lines(file)})
    return datasets


def _list_models() -> List[Dict[str, Any]]:
    discovered: set[Path] = set()
    if not ARTIFACTS_DIR.exists():
        return []
    for config_file in ARTIFACTS_DIR.rglob("config.json"):
        candidate = config_file.parent
        if (candidate / "pytorch_model.bin").exists():
            discovered.add(candidate)

    history = _parse_training_history()
    latest = history[-1] if history else None

    models: List[Dict[str, Any]] = []
    for directory in sorted(discovered):
        rel = directory.relative_to(ARTIFACTS_DIR).as_posix()
        models.append(
            {
                "name": rel,
                "version": datetime.fromtimestamp(directory.stat().st_mtime).isoformat(),
                "tags": ["checkpoint" if "checkpoint" in rel else "model"],
                "description": str(directory.resolve()),
                "isActive": directory.resolve() == BEST_MODEL_DIR.resolve(),
                "metrics": latest,
            }
        )
    return models


@router.get("/dashboard")
async def dashboard_snapshot() -> Dict[str, Any]:
    history = _parse_training_history(limit=1)
    return {
        "activeModel": _current_model_name(),
        "queueSize": training_manager.queue_size,
        "pendingJobs": training_manager.pending_jobs,
        "lastTraining": history[0] if history else None,
        "datasets": _list_datasets(),
    }


@router.get("/training/history")
async def training_history() -> List[Dict[str, Any]]:
    return _parse_training_history()


@router.get("/models")
async def models() -> List[Dict[str, Any]]:
    return _list_models()


@router.post("/models/activate", status_code=status.HTTP_202_ACCEPTED)
async def activate_model(payload: Dict[str, str]) -> Dict[str, str]:
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Nome do modelo é obrigatório.")
    target = ARTIFACTS_DIR / name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Modelo não encontrado.")
    if target.resolve() == BEST_MODEL_DIR.resolve():
        return {"status": "already_active"}

    if BEST_MODEL_DIR.exists():
        shutil.rmtree(BEST_MODEL_DIR)
    shutil.copytree(target, BEST_MODEL_DIR)
    return {"status": "activated"}


class TrainingManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: set[asyncio.Queue[TrainingEvent]] = set()
        self._task: asyncio.Task | None = None
        self.pending_jobs = 0

    @property
    def queue_size(self) -> int:
        return 1 if self._task and not self._task.done() else 0

    def register(self) -> asyncio.Queue[TrainingEvent]:
        queue: asyncio.Queue[TrainingEvent] = asyncio.Queue()
        self._clients.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue[TrainingEvent]) -> None:
        self._clients.discard(queue)

    async def broadcast(self, event: TrainingEvent) -> None:
        for queue in list(self._clients):
            await queue.put(event)

    async def start(self, request: TrainingRequest) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                raise HTTPException(status_code=409, detail="Treinamento já em andamento.")
            self.pending_jobs += 1
            self._task = asyncio.create_task(self._run_training(request))

    async def _run_training(self, request: TrainingRequest) -> None:
        await self.broadcast(TrainingEvent(type="status", status="iniciando"))
        env = os.environ.copy()
        env.update(
            {
                "LOCAL_DATASET_PATH": request.datasetPath,
                "NUM_EPOCHS": str(request.epochs),
                "BATCH_SIZE": str(request.batchSize),
                "LEARNING_RATE": str(request.learningRate),
            }
        )

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "trainer.train",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        await self.broadcast(TrainingEvent(type="status", status="treinando"))

        try:
            assert process.stdout
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="ignore").strip()
                if not text:
                    continue
                await self.broadcast(TrainingEvent(type="log", message=text))
                if "eval_accuracy" in text and "eval_loss" in text:
                    metric = _try_parse_metric_line(text)
                    if metric:
                        await self.broadcast(TrainingEvent(type="metric", metric=metric))

            await process.wait()
            status_text = "concluído" if process.returncode == 0 else f"erro ({process.returncode})"
            await self.broadcast(TrainingEvent(type="status", status=status_text))
        finally:
            self.pending_jobs = max(0, self.pending_jobs - 1)


def _try_parse_metric_line(line: str) -> Dict[str, Any] | None:
    try:
        payload = ast.literal_eval(line.split("INFO - ")[-1])
    except Exception:
        return None
    required = {"eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_loss"}
    if not required.issubset(payload.keys()):
        return None
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy": float(payload.get("eval_accuracy", 0.0)),
        "precision": float(payload.get("eval_precision", 0.0)),
        "recall": float(payload.get("eval_recall", 0.0)),
        "f1": float(payload.get("eval_f1", 0.0)),
        "loss": float(payload.get("eval_loss", 0.0)),
        "duration": float(payload.get("eval_runtime", 0.0)),
        "datasetSize": float(payload.get("eval_samples_per_second", 0.0))
        * float(payload.get("eval_runtime", 0.0)),
    }


training_manager = TrainingManager()


@router.post("/training/start", status_code=status.HTTP_202_ACCEPTED)
async def start_training(request: TrainingRequest) -> Dict[str, str]:
    await training_manager.start(request)
    return {"status": "scheduled"}


@router.get("/training/stream")
async def training_stream() -> EventSourceResponse:
    queue = training_manager.register()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                event = await queue.get()
                yield f"data: {event.json()}\n\n"
        finally:
            training_manager.unregister(queue)

    return EventSourceResponse(event_generator())


@router.post("/commands/run")
async def run_command(payload: CommandRequest) -> Dict[str, Any]:
    if not payload.command:
        raise HTTPException(status_code=400, detail="Comando é obrigatório.")

    cmd = [payload.command, *payload.args] if payload.args else [payload.command]
    if cmd and cmd[0] == "python":
        cmd[0] = sys.executable
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = (result.stdout or "") + (result.stderr or "")
        return {"output": output.strip(), "returncode": result.returncode}
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return {"output": output.strip(), "returncode": exc.returncode}
