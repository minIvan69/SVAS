# Файл переменных окружения
ENV_FILE = .env

# Команда запуска docker compose
DC = docker compose --env-file $(ENV_FILE)

# 🛠️ Построить образы (с кэшем)
build:
	$(DC) build

# 🧼 Пересобрать образы (без кэша)
rebuild:
	$(DC) build --no-cache

# 🚀 Запустить контейнеры
up:
	$(DC) up

# 🚀 Пересобрать и запустить контейнеры
up-build:
	$(DC) up --build

# 🛑 Остановить контейнеры
down:
	$(DC) down

# 🧹 Удалить все dangling образы
prune-images:
	docker image prune -f

# 🧹 Удалить всё, что не используется (остановленные контейнеры, тома, сети и образы)
prune-all:
	docker system prune -a --volumes -f

# 💣 Полная зачистка: остановка, удаление контейнеров, удаление всех образов
nuke:
	$(DC) down -v --rmi all --remove-orphans
	docker system prune -a --volumes -f

# 👀 Посмотреть логи
logs:
	$(DC) logs -f

# 🐳 Посмотреть, что запущено
ps:
	$(DC) ps

# 🔁 Перезапуск конкретного контейнера
restart:
	@read -p "Enter container name to restart: " name; \
	docker compose restart $$name

# 🔁 Перезапуск определенного контейнера через аргумент (make restart-one name=flask_app):
restart-one:
	@if [ -z "$(name)" ]; then \
		echo "❌ Usage: make restart-one name=container_name"; \
	else \
		docker compose restart $(name); \
	fi

# 🔁 Пересборка и рестарт конкретного контейнера (make rebuild-one name=flask_app)
rebuild-one:
	@if [ -z "$(name)" ]; then \
		echo "❌ Usage: make rebuild-one name=container_name"; \
	else \
		docker compose build $(name); \
		docker compose stop $(name); \
		docker compose up -d $(name); \
	fi

# 🐚 Зайти внутрь контейнера (bash)
shell:
	@if [ -z "$(name)" ]; then \
		echo "❌ Usage: make shell name=container_name"; \
	else \
		docker exec -it $(name) /bin/bash || docker exec -it $(name) /bin/sh; \
	fi

# 📜 Показать логи конкретного контейнера
logs-one:
	@if [ -z "$(name)" ]; then \
		echo "❌ Usage: make logs-one name=container_name"; \
	else \
		docker compose logs -f $(name); \
	fi