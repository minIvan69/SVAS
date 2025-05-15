# services/api/app.py
import os, tempfile, shutil, uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from starlette.background import BackgroundTasks
from celery.result import AsyncResult
from worker.tasks import celery_app
from core.db import get_async_session
from core.models import Base
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from celery_app import celery_app
from deps import get_async_session
from crud import get_by_user
import shutil, uuid, pathlib, aiofiles

API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(title="VoiceID API")


UPLOAD_DIR = pathlib.Path("/tmp/voiceid")
UPLOAD_DIR.mkdir(exist_ok=True)

def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="invalid key")
    
@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/enroll/")
async def enroll(
    username: str,
    speaker_id: str,
    file: UploadFile = File(...),
    db = Depends(get_async_session),
):
    if await get_by_user(db, username):
        raise HTTPException(400, "user already enrolled")

    # сохраняем во временный файл
    tmp = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    async with aiofiles.open(tmp, "wb") as out:
        while chunk := await file.read(4096):
            await out.write(chunk)

    # пуляем задачу
    celery_app.send_task(
        "voiceid.extract",
        args=[username, speaker_id, str(tmp)]
    )
    return {"status": "queued"}


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
