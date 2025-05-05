from celery import Celery
celery_app = Celery(__name__, broker=settings.RABBIT_URL)