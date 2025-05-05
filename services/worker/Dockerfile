# services/api/Dockerfile  (worker такой же)
FROM mambaorg/micromamba:1.5.3

# 1)  Python и бинарные зависимости
COPY environment.yml /tmp/
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# 2)  Код приложения
WORKDIR /app
COPY . /app

# 3)  Скачиваем модель во время build
RUN python - <<'PY'
import torch, os, requests, zipfile, io
os.makedirs('/models', exist_ok=True)
torch.hub.load('speechbrain/speechbrain', 'spkrec-ecapa-voxceleb',
               savedir='/models')
PY

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
