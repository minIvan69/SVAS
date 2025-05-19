## 🚀 Сборка

make up-build

## 🔁Запуск(если уже всё собрано)

make up

## 🛑 Остановка:

make down

## 🧼 Почистить dangling-образы:

make prune-images

## ☠️ Полная зачистка проекта:

make nuke

## 👀 Логи:

make logs

# Рестарт без пересборки

make restart-one name=flask_app

# Рестарт с пересборкой

make rebuild-one name=flask_app

source .venv/bin/activate

ython services/worker/train_thresholds.py \
 --dataset-root ~/Datasets/voxceleb1/wav \
 --sample-fraction 1.0

поднять db
docker-compose -f docker-compose.dev.yml pull db # подтянуть новый тег
docker-compose -f docker-compose.dev.yml up -d db # запустить только БД
проверить
docker-compose -f docker-compose.dev.yml ps

curl -X POST "http://localhost:8000/enroll/?user=sweet" \
 -F "file=@uploads/sweet1.wav"

curl -X POST "http://localhost:8000/verify/?user=sweet" \
 -F "file=@uploads/sweet1.wav"

docker compose -f docker-compose.dev.yml up -d --build api (контейнер) запуск с подтягиванием
