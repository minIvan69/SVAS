"""Lazy‑loaded singleton around SpeechBrain ECAPA‑TDNN.

`get_speaker_model()` ensures we download weights only once and reuse the
instance across Celery workers (each worker process will hold its own copy).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

import torch
from speechbrain.pretrained import SpeakerRecognition

from .config import settings

__all__ = ["get_speaker_model", "extract_embedding"]

class Base(DeclarativeBase):
    pass


class Embedding(Base):
    __tablename__ = "embeddings"

    id:          Mapped[int]  = mapped_column(primary_key=True)
    speaker_id:  Mapped[str]  = mapped_column(index=True)
    vec:         Mapped[list] = mapped_column(Vector(192))
    created_at:  Mapped[datetime] = mapped_column(
        default=datetime.utcnow, nullable=False
    )


def _device() -> str:
    """Pick the best available device automatically (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_speaker_model(device: Optional[str] = None) -> SpeakerRecognition:
    device = device or _device()
    run_opts = {"device": device}
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path(settings.MODEL_CACHE_DIR)),
        run_opts=run_opts,
    )
    return model


def extract_embedding(
    wav: torch.Tensor, *, sample_rate: int = settings.SAMPLE_RATE, device: Optional[str] = None
) -> torch.Tensor:
    """Compute a 192‑D speaker embedding from a mono waveform tensor.

    Args:
        wav: Tensor shape (1, n_samples) in the range [-1, 1].
        sample_rate: Sampling rate of `wav`.
        device: Override target device.

    Returns:
        1‑D tensor of length 192 on CPU memory (ready for DB serialisation).
    """
    model = get_speaker_model(device)

    if sample_rate != settings.SAMPLE_RATE:
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0),
            scale_factor=settings.SAMPLE_RATE / sample_rate,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    with torch.inference_mode():
        emb = model.encode_batch(wav, torch.tensor([settings.SAMPLE_RATE]))
    return emb.squeeze().cpu()