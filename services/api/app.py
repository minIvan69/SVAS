# services/api/app.py
import os, tempfile, shutil, uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from starlette.background import BackgroundTasks
from celery.result import AsyncResult
from worker.tasks import celery_app
from core.db import get_async_session
from core.models import Base

API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(title="SVAS Voice API")


def check_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="invalid key")


@app.post("/enroll")
async def enroll(
    background: BackgroundTasks,
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
    _=Depends(check_key),
):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    shutil.copyfileobj(audio.file, tmp)
    tmp.close()

    task = celery_app.send_task("enroll_voice", args=[speaker_id, tmp.name])
    background.add_task(lambda: os.unlink(tmp.name))
    return {"task_id": task.id}


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
