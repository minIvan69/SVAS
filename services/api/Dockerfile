FROM python:3.11-slim

# 1.  system deps -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc && \
    rm -rf /var/lib/apt/lists/*

# 2.  рабочая папка – корень проекта внутри образа
WORKDIR /app

# 3.  зависимости -----------------------------------------------------
COPY services/api/requirements.txt ./requirements.txt
#  (если у api и worker одинаковый requirements.txt – скопируйте один)
RUN pip install --no-cache-dir -r requirements.txt

# 4.  копируем весь каталог services/ !!! -----------------------------
COPY services/ /app/services/

# 5.  PYTHONPATH, чтобы /app попадал в поиск модулей
ENV PYTHONPATH=/app

# 6.  команда запуска -------------------------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]
