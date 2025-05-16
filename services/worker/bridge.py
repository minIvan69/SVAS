# services/worker/tasks.py
from celery import Celery
from core.model import verify  # heavy ML-код

celery_app = Celery(
    __name__,
    broker="amqp://guest:guest@rabbit//",
)

@celery_app.task(name="verify_speaker")
def verify_speaker(user: str, audio_url: str) -> bool:
    return verify(user, audio_url)
