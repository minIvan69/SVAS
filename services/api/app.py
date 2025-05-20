# services/api/app.py
from __future__ import annotations

import os, uuid
from typing import Annotated
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header,status
from pydantic import BaseModel,Field
from celery.result import AsyncResult
from services.worker.tasks import celery_app

# ─────────── настройки ──────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads")).absolute()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(title="VoiceID API")


class _FileBody(BaseModel):
    filename: str = Field(..., examples=["sample.wav"])
# ─────────── схемы запросов/ответов ─────────────────────────────────────────────
class EnrollIn(BaseModel):
    user: str
    filename: str               #- имя файла, который уже лежит в UPLOAD_DIR

class EnrollOut(BaseModel):
    task_id: str
    speaker_id: str
    file: str

class VerifyIn(BaseModel):
    speaker_id: str
    filename: str  # имя файла для проверки в папке uploads/

class VerifyOut(BaseModel):
    task_id: str
    speaker_id: str
    file: str

# ─────────── auth-header helper ─────────────────────────────────────────────────
def check_key(x_api_key: str = Header(...)):
    if x_api_key != "changeme":
        raise HTTPException(403, "invalid key")
    
# ──────────────────────────  Helpers  ───────────────────────────────────────────
def _assert_wav(path: Path) -> None:
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"{path.name!r} not found in {UPLOAD_DIR}")
    if path.suffix.lower() != ".wav":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="only .wav files accepted")

# ─────────── ping ───────────────────────────────────────────────────────────────
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# ─────────── enroll (JSON) ──────────────────────────────────────────────────────
@app.post("/enroll", response_model=EnrollOut, status_code=202)
async def enroll(body: EnrollIn, _=Depends(check_key)):
    """
    Создать голосовой профиль для `body.user` по файлу `body.filename`
    (файл должен находиться в каталоге uploads).
    """
    file_path = UPLOAD_DIR / body.filename
    if not file_path.exists():
        raise HTTPException(404, f"{body.filename} not found in uploads/")
    if file_path.suffix.lower() != ".wav":
        raise HTTPException(415, "only .wav files accepted")

    speaker_id = str(uuid.uuid4())
    task = celery_app.send_task(
        "services.worker.tasks.enroll_task",          # имя Celery-задачи в worker’e
        args=[speaker_id, body.user, str(file_path)],
    )
    return EnrollOut(task_id=task.id, speaker_id=speaker_id, file=body.filename)

@app.post(
    "/verify",
    response_model=VerifyOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Поставить задачу на верификацию",
)
async def verify(body: VerifyIn, _: Annotated[None, Depends(check_key)]):
    """
    Проверить, принадлежит ли WAV-файл **body.filename**
    выше зарегистрированному **body.speaker_id**.
    """
    file_path = UPLOAD_DIR / body.filename
    if not file_path.exists():
        raise HTTPException(404, f"{body.filename} not found in uploads/")
    if file_path.suffix.lower() != ".wav":
        raise HTTPException(415, "only .wav files accepted")
    # _assert_wav(file_path)

    task = celery_app.send_task(
        "services.worker.tasks.verify_task",
        args=[body.speaker_id, str(file_path)],
    )

    return VerifyOut(task_id=task.id, speaker_id=body.speaker_id, file=body.filename)

# ─────────── посмотреть статус/результат ────────────────────────────────────────
@app.get("/result/{task_id}")
async def result(task_id: str, _=Depends(check_key)):
    res = AsyncResult(task_id, app=celery_app)
    return {"status": res.status} if not res.ready() else res.result
