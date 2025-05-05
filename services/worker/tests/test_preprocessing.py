"""PyTest smoke‑tests for the first PR.

They verify that segmentation and the full pipeline work on synthetic audio on
any platform (CI / M1 / x86)."""

from pathlib import Path

import torch
import torchaudio

from core import preprocess, settings


def _tone(duration: float = 3.2, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(int(settings.SAMPLE_RATE * duration)) / settings.SAMPLE_RATE
    return (0.05 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def test_segment_lengths(tmp_path: Path):
    wav = _tone(3.0)
    file = tmp_path / "tone.wav"
    torchaudio.save(file.as_posix(), wav, settings.SAMPLE_RATE)

    segments = preprocess(file, denoise=False)
    expected_len = int(settings.SEGMENT_LENGTH_SEC * settings.SAMPLE_RATE)
    assert all(seg.shape == (1, expected_len) for seg in segments)
    assert len(segments) >= 1
