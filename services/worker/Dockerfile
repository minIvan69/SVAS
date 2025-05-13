# FROM python:3.11-slim

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libsndfile1-dev build-essential git && \
#     rm -rf /var/lib/apt/lists/*

# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# WORKDIR /app

# # 1) requirements — указываем правильный относительный путь
# COPY services/worker/requirements.txt /tmp/requirements.txt
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r /tmp/requirements.txt

# # 2) код worker
# COPY services/worker/ .

# # 3) Alembic конфиг и папка миграций
# COPY alembic.ini  /app/alembic.ini
# COPY alembic      /app/alembic

# ENV PYTHONUNBUFFERED=1
# CMD ["celery", "-A", "core.celery_app:celery_app", "worker", "--loglevel=info"]

FROM python:3.11-slim

# system libs for psycopg binary + soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc libsndfile1-dev git && \
    rm -rf /var/lib/apt/lists/*

# faster wheels for torch
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# === install deps ===
COPY services/worker/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        alembic psycopg[binary] asyncpg sqlalchemy[asyncio] pgvector sqlalchemy-pgvector

# === copy code ===
COPY services/worker/ .

ENV PYTHONPATH=/app
CMD ["celery", "-A", "core.celery_app:celery_app", "worker", "--loglevel=info"]
