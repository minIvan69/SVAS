"""Lazy‑loaded singleton around SpeechBrain ECAPA‑TDNN.

`get_speaker_model()` ensures we download weights only once and reuse the
instance across Celery workers (each worker process will hold its own copy).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union, Sequence
import torch
import torchaudio
import numpy as np
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, VARCHAR
# from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector

import torch
from speechbrain.pretrained import SpeakerRecognition,EncoderClassifier

from .config import settings

SR = 16_000
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
CACHE_DIR    = "/root/.cache/speechbrain"   

__all__ = ["get_speaker_model", "extract_embedding"]

class Base(DeclarativeBase):
    pass

_model: SpeakerRecognition | None = None

@lru_cache
def _sb_model() -> SpeakerRecognition:
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/app/.cache/speechbrain"
    )

def _to_tensor(wav: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(wav, str):                       # путь к файлу
        sig, sr = torchaudio.load(wav)
    elif isinstance(wav, np.ndarray):
        sig = torch.from_numpy(wav).unsqueeze(0)
        sr = SR
    else:
        sig, sr = wav.unsqueeze(0), SR

    if sr != SR:
        sig = torchaudio.functional.resample(sig, sr, SR)
    return sig

class _Wrapper:
    """Добавляем метод `encode`, чтобы старая логика не падала."""
    def __init__(self, m: SpeakerRecognition):
        self._m = m

    def encode(self, wav: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            emb = self._m.encode_batch(_to_tensor(wav)).squeeze(0)
        return emb.cpu().numpy()

@lru_cache(maxsize=1)
def get_model():
    """
    Скачивает (один раз) и возвращает
    SpeechBrain EncoderClassifier с методом encode_batch().
    """
    return EncoderClassifier.from_hparams(
        source  = MODEL_SOURCE,
        savedir = CACHE_DIR,
        run_opts={"device": "cpu"},        # GPU не предполагается в Dev-контейнере
    )


# def get_model() -> SpeakerRecognition:
#     global _model
#     if _model is None:
#         _model = SpeakerRecognition.from_hparams(
#             source="speechbrain/spkrec-ecapa-voxceleb",
#             savedir="/root/.cache/speechbrain/ecapa"   # чтобы качалось один раз
#         )
#     return _model

class Embedding(Base):
    __tablename__ = "embeddings"

    id:   Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(VARCHAR(64), unique=True, index=True)
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