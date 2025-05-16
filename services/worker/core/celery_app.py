# services/worker/celery_app.py
from celery import Celery
from services.worker.core.config import settings   # ваш pydantic‑Settings

celery_app = Celery(
    "voiceid",
    broker=settings.RABBIT_URL,
    backend=settings.RABBIT_URL,   # или Redis, если настроите
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]