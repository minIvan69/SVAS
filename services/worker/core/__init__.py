"""Public shortcuts so that `import core` feels intuitive during dev."""

from .config import settings  # noqa: F401 – re‑export
from .model import extract_embedding, get_speaker_model  # noqa: F401 – re‑export
from .preprocessing import preprocess  # noqa: F401 – re‑export

__all__ = [
    "settings",
    "extract_embedding",
    "get_speaker_model",
    "preprocess",
]