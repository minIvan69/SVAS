# services/worker/celery_app.py
from celery import Celery
from services.worker.core.config import settings   # ваш pydantic‑Settings

celery_app = Celery(
    "voiceid",
    broker=settings.RABBIT_URL,
    backend='rpc://',                  # <-- заменили
)
celery_app.autodiscover_tasks(['services.worker'])
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]