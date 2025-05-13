from pathlib import Path
from pydantic_settings import BaseSettings  # Pydantic v2
import yaml


class Settings(BaseSettings):
    """Project‑wide settings.

    Any env var with prefix `SVAS_` overrides defaults, e.g. `SVAS_RABBIT_URL=`.
    """
    DB_URL_ASYNC: str = "postgresql+asyncpg://postgres:postgres@db/voiceid"
    DB_URL_SYNC:  str = "postgresql+psycopg://postgres:postgres@localhost:5432/voiceid"
    # ── infra
    RABBIT_URL: str = "amqp://guest:guest@rabbitmq:5672//"
    POSTGRES_DSN: str = "postgresql+psycopg://svas_worker:password@postgres:5432/svas"

    # ── paths / models
    # MODEL_CACHE_DIR: Path = Path("/app/.model_cache")
    MODEL_CACHE_DIR: Path = Path.home() / ".cache/svas_models"
    THRESHOLDS_PATH: Path = Path(__file__).with_name("thresholds.yml")

    # ── audio params
    SAMPLE_RATE: int = 16_000
    SEGMENT_LENGTH_SEC: float = 2.0
    SEGMENT_HOP_SEC: float = 1.0

    class Config:
        env_prefix = "SVAS_"


settings = Settings()


# Helper – lazy load thresholds once per worker
_thresholds_cache = None


def load_thresholds():
    global _thresholds_cache
    if _thresholds_cache is None:
        with open(settings.THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            _thresholds_cache = yaml.safe_load(f)
    return _thresholds_cache
