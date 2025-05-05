from celery import Celery
from .config import RABBIT_URL
celery_app = Celery('worker', broker=RABBIT_URL)