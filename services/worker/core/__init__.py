"""Public shortcuts so that `import core` feels intuitive during dev."""

from .config import settings  # noqa: F401 – re‑export
from .model import extract_embedding, get_speaker_model  # noqa: F401 – re‑export
from .preprocessing import preprocess  # noqa: F401 – re‑export
from types import ModuleType as _ModuleType  # noqa: E402, PEP 563
import sys as _sys  # noqa

__all__ = [
    "settings",
    "extract_embedding",
    "get_speaker_model",
    "preprocess",
]

_mod: _ModuleType = _sys.modules[__name__]
if not hasattr(_mod, "verify"):
    from .scoring import verify  # type: ignore  # noqa: F401

__all__.append("verify")