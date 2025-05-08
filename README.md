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
