from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Centralised configuration for the SVAS worker.

    Values may be overridden via environment variables or a `.env` file that
    lives at the project root.  Keeping everything in one place avoids the
    classic "settings‑creep" that plagues large Celery/FastAPI stacks.
    """

    # ── Infrastructure ────────────────────────────────────────────────────────
    RABBIT_URL: str = "amqp://guest:guest@rabbitmq:5672//"
    POSTGRES_DSN: str = "postgresql+psycopg://svas_worker:password@postgres:5432/svas"

    # ── Models & audio ────────────────────────────────────────────────────────
    MODEL_CACHE_DIR: Path = Path("/app/.model_cache")  # mapped volume in Docker
    SAMPLE_RATE: int = 16_000
    SEGMENT_LENGTH_SEC: float = 2.0  # ECAPA was trained on 2‑s chunks
    SEGMENT_HOP_SEC: float = 1.0     # 50 % overlap → robust mean‑vec

    # VAD / denoise tweaks
    FRAME_LENGTH_MS: int = 25        # for RNNoise + VAD
    VAD_THRESHOLD: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton‑style access, importable from anywhere
settings = Settings()
