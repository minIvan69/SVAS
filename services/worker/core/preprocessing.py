"""Audio preprocessing pipeline for SVAS worker.

1. **Load & resample** to 16 kHz mono
2. **RNNoise** denoising (optional ‑ falls back to identity if wheel missing)
3. **Silero VAD** to remove non‑speech
4. **Segmentation** into overlapping 2‑s windows ready for ECAPA
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from pathlib import Path
import numpy as np
import torch
import torchaudio, torchaudio.transforms as T, tempfile, pathlib
from .config import settings

# ── RNNoise (optional on arm64) ───────────────────────────────────────────────
try:
    from rnnoise import RNNoise  # type: ignore

    _rn = RNNoise()

    def _denoise(w: torch.Tensor) -> torch.Tensor:
        w16 = (w * 32768.0).short().squeeze(0).numpy()
        w16 = _rn.filter(w16)
        return torch.from_numpy(w16).unsqueeze(0).float() / 32768.0

except Exception:  # pragma: no cover – pristine arm Macs often lack rnnoise

    def _denoise(w: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple alias
        return w

# ── Silero VAD (optional) ────────────────────────────────────────────────────
try:
    from silero.vad import VoiceActivityDetector  # type: ignore

    _vad = VoiceActivityDetector(torch.device("cpu"))

    def _apply_vad(w: torch.Tensor) -> torch.Tensor:
        ts = _vad(w.squeeze(0), settings.SAMPLE_RATE, threshold=settings.VAD_THRESHOLD)
        if not ts:
            return torch.zeros_like(w)
        voiced = torch.cat(
            [w[:, int(t["start"] * settings.SAMPLE_RATE) : int(t["end"] * settings.SAMPLE_RATE)] for t in ts],
            dim=1,
        )
        return voiced

except Exception:  # pragma: no cover – allow CI without silero

    def _apply_vad(w: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple alias
        return w

# ── Core helpers ─────────────────────────────────────────────────────────────

def load_audio(path: str | Path) -> torch.Tensor:
    w, sr = torchaudio.load(str(path))
    if sr != settings.SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, settings.SAMPLE_RATE)
    if w.shape[0] > 1:  # stereo → mono
        w = w.mean(0, keepdim=True)
    return w


def segment(w: torch.Tensor) -> List[torch.Tensor]:
    seg_len = int(settings.SEGMENT_LENGTH_SEC * settings.SAMPLE_RATE)
    hop = int(settings.SEGMENT_HOP_SEC * settings.SAMPLE_RATE)
    # Pad short utterances so we always have ≥ 1 segment
    if w.shape[1] < seg_len:
        w = torch.nn.functional.pad(w, (0, seg_len - w.shape[1]))
    return [w[:, i : i + seg_len] for i in range(0, w.shape[1] - seg_len + 1, hop)]


# ── Public API ───────────────────────────────────────────────────────────────

def preprocess(path: str | Path, *, denoise: bool = True) -> List[torch.Tensor]:
    """Full pipeline: load → (optional) denoise → VAD → segment."""
    w = load_audio(path)
    if denoise:
        w = _denoise(w)
    w = _apply_vad(w)
    return segment(w)

def denoise_and_split(
        wav_path: Path | str,
        *_,                          #  ← примет все позиционные «лишние»
        **__,                        #  ← и любые именованные аргументы
) -> list[np.ndarray]:
    """
    ВРЕМЕННАЯ ЗАГЛУШКА ⚠️
    Принимает любые дополнительные аргументы и просто отдаёт
    «как есть» один-единственный фрагмент аудио (без денойза
    и без разбивки), чтобы Celery-таски не падали.
    """

    # ── минимальная реализация ───────────────────────────────
    import soundfile as sf
    # import librosa

    y, sr = sf.read(wav_path)          # load wav
    # if y.ndim > 1:                     # стерео → моно
    #     y = librosa.to_mono(y.T)

    return [y.astype(np.float32)]      # один «сегмент» в списке

# def denoise_and_split(
#         wav_path: Path | str,
#         *,
#         sr: int = 16_000,          #  ➜  добавили аргумент со значением по-умолчанию
#         chunk_sec: float = 1.5,
#         silence_sec: float = 0.3,
# ) -> list[np.ndarray]:
    
#     y, file_sr = librosa.load(wav_path, sr=None, mono=True)
#     if file_sr != sr:
#         y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
#     sig, sr = torchaudio.load(wav_path)

#     # if sr != sr_out:
#     #     sig = torchaudio.functional.resample(sig, sr, sr_out)
#     # очень простой VAD: режем на окна фикс. длины
#     n, step = sig.shape[-1], int(win_sec * sr_out)
#     parts = []
#     for i in range(0, n, step):
#         chunk = sig[:, i:i+step]
#         if chunk.shape[-1] < step: break
#         tmp = tempfile.NamedTemporaryFile(
#             suffix=".wav", delete=False, dir="/tmp").name
#         torchaudio.save(tmp, chunk, sr_out)
#         parts.append(tmp)
#     return parts