# services/api/app.py
import os, tempfile, shutil, uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTasks
from celery.result import AsyncResult
from services.worker.tasks import celery_app
from services.worker.core.db import async_session_maker
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# from celery_app import celery_app
from services.api.deps import get_async_session
from services.api.crud import get_by_user
import shutil, uuid, pathlib, aiofiles
from celery import Celery, states
from pathlib import Path

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads")).absolute()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # на всякий случай
API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(title="VoiceID API")


UPLOAD_DIR = pathlib.Path("/tmp/voiceid")
UPLOAD_DIR.mkdir(exist_ok=True)
# 2) «Фабрика» сессий --------------------------------------------------

# -------------------------------------------------------------------------
#  Хелпер: сохранить файл на диск и вернуть путь
# -------------------------------------------------------------------------
async def _store_file(upload: UploadFile) -> Path:
    if upload.content_type not in ("audio/wav", "audio/x-wav"):
        raise HTTPException(415, detail="only audio/wav accepted")

    # уникальное имя → <uuid>.wav
    fname = f"{uuid.uuid4()}.wav"
    dst = UPLOAD_DIR / fname

    # Write chunk-by-chunk чтобы не съесть всю память
    with dst.open("wb") as f:
        while chunk := await upload.read(8192):
            f.write(chunk)

    return dst


@asynccontextmanager
async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with async_session_maker() as session:
        yield session

def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="invalid key")
    
@app.get("/ping")
async def ping():
    return {"status": "ok"}


# @app.post("/enroll/")
# async def enroll(
#     username: str,
#     speaker_id: str,
#     file: UploadFile = File(...),
#     db = Depends(get_async_session),
# ):
#     if await get_by_user(db, username):
#         raise HTTPException(400, "user already enrolled")
    
#     dest = pathlib.Path("/app/uploads") / audio.filename
#     with dest.open("wb") as f:
#         f.write(await audio.read())

#     # # сохраняем во временный файл
#     # tmp = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
#     # async with aiofiles.open(tmp, "wb") as out:
#     #     while chunk := await file.read(4096):
#     #         await out.write(chunk)

#     # пуляем задачу
#     celery_app.send_task(
#         "voiceid.extract",
#         args=[username, speaker_id, str(tmp)]
#     )
#     return {"status": "queued"}

@app.post("/enroll")
async def enroll(user: str, audio: UploadFile = File(...)) -> JSONResponse:
    """
    Сохранить голос `audio` и создать голосовой профиль для `user`.
    """
    file_path = await _store_file(audio)
    speaker_id = str(uuid.uuid4())

    task = celery_app.send_task(
        "tasks.enroll",  # имя задачи в worker'е
        args=[speaker_id, user, str(file_path)],
    )

    return JSONResponse(
        {
            "task_id": task.id,
            "speaker_id": speaker_id,
            "file": file_path.name,
        },
        status_code=202,
    )



@app.post("/verify")
async def verify(
    background: BackgroundTasks,
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
    _=Depends(check_key),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    shutil.copyfileobj(audio.file, tmp)
    tmp.close()

    task = celery_app.send_task("verify_voice", args=[speaker_id, tmp.name])
    background.add_task(lambda: os.unlink(tmp.name))
    return {"task_id": task.id}


@app.get("/result/{task_id}")
async def result(task_id: str, _=Depends(check_key)):
    res = AsyncResult(task_id, app=celery_app)
    if not res.ready():
        return {"status": res.status}
    return res.result

