"""Audio preprocessing pipeline for SVAS worker.

1. **Load & resample** to 16 kHz mono
2. **RNNoise** denoising (optional ‑ falls back to identity if wheel missing)
3. **Silero VAD** to remove non‑speech
4. **Segmentation** into overlapping 2‑s windows ready for ECAPA
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torchaudio

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
