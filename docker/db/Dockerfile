# Dockerfile.db
FROM postgres:15

# подключаем pgdg и вытягиваем pgvector, собранный под arm64
RUN echo "deb [arch=arm64] http://apt.postgresql.org/pub/repos/apt/ bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list  && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor \
        -o /usr/share/keyrings/postgresql.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/postgresql.gpg] \
         http://apt.postgresql.org/pub/repos/apt/ bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends postgresql-15-pgvector && \
    rm -rf /var/lib/apt/lists/*