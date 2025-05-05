FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1-dev build-essential git && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY services/worker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY services/worker/ .

ENV PYTHONUNBUFFERED=1
CMD ["celery", "-A", "core.celery_app:celery_app", "worker", "--loglevel=info"]
