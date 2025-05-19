import numpy as np
from requests import Session
import os, asyncio, uuid, torch
from celery import Celery
from services.worker.core.db import SessionLocal, get_sync_session
from services.worker.core.model import Embedding, Base
from services.worker.core import preprocess, extract_embedding, verify
from services.worker.core.celery_app import celery_app
from services.worker.core.preprocessing import denoise_and_split
from services.worker.core.model import get_speaker_model, get_model
from services.worker.core.scoring import verify
from services.worker.core.model   import extract_embedding
from services.worker.core.db import save_profile, load_profile
from services.worker.core.celery_app import celery_app
from services.worker.core.db import async_session_maker     # ваш create_async_engine(...)
from .crud import add_embedding
from sklearn.metrics.pairwise import cosine_similarity
import torch

celery_app = Celery(
    "svas",
    broker=os.getenv("RABBIT_URL"),         # amqp://guest:guest@rabbit:5672//
    backend="rpc://",
)

# ---- helpers ---------------------------------------------------------------
def _store_vec(speaker_id: str, vec: torch.Tensor):
    from pgvector.sqlalchemy import Vector
    with get_sync_session() as db:
        db.add(Embedding(speaker_id=speaker_id, vec=vec.tolist()))
        db.commit()

def _fetch_profile(speaker_id: str):
    with get_sync_session() as db:
        rows = db.query(Embedding).filter_by(speaker_id=speaker_id).all()
        return torch.tensor([r.vec for r in rows]) if rows else None
# ----------------------------------------------------------------------------

@celery_app.task(name="enroll_voice")
def enroll_voice(speaker_id: str, wav_path: str):
    segs = preprocess(wav_path, denoise=False)
    vec   = extract_embedding(segs[0])
    _store_vec(speaker_id, vec)
    return {"status": "ok"}

@celery_app.task(name="verify_voice")
def verify_voice(speaker_id: str, wav_path: str):
    segs = preprocess(wav_path, denoise=False)
    probe = extract_embedding(segs[0])

    profile = _fetch_profile(speaker_id)
    if profile is None:
        return {"match": False, "reason": "no-profile"}

    result = verify(probe, profile)
    return result

@celery_app.task(name="services.worker.tasks.enroll_task")
def enroll_task(speaker_id: str, user: str, wav_path: str) -> None:
    """
    • denoise + ресемплинг 16 kHz  
    • берём ≤5 кусков по 2–3 сек, считаем эмбеддинги, усредняем  
    • сохраняем в pgvector
    """
    model = get_model()

    # ① speechbrain ожидает 16 kHz float32 tensor [-1,1]
    segments = denoise_and_split(wav_path, sr=16_000)

    if not segments:
        raise RuntimeError("no speech found")

    # в батч → [B, T]
    batch = torch.stack([torch.tensor(seg) for seg in segments[:5]])
    batch_tensor = torch.tensor(batch, dtype=torch.float32)
    # ② получаем [B, 192]
    embeds = model.encode_batch(batch_tensor).squeeze(1)          # -> Tensor[B,192]
    vec = embeds.mean(dim=0).cpu().numpy()               # -> np.ndarray[192]

    # ③ INSERT INTO embeddings…
    with Session() as db:
        db.add(Embedding(user=user,
                         speaker_id=speaker_id,
                         vec=vec.tolist()))
        db.commit()

@celery_app.task(name="services.worker.tasks.verify_task")   # !!! имя должно совпадать
def verify_task(speaker_id: str, wav_path: str) -> dict:
    """
    • вытаскивает эмбеддинг тестового файла
    • сравнивает с эталонными векторами speaker_id
    • возвращает {'score': float, 'success': bool}
    """
    model = get_model()

    # ---------- 1. эмбеддинг проверочного файла ----------
    # encode_file принимает путь к 16-kHz WAV и сам делает VAD/нормализацию
    probe_vec = (
        model.encode_file(wav_path)     # Tensor[1,192]
        .squeeze(0)                     # -> [192]
        .cpu()
        .numpy()
    )

    # ---------- 2. достаём все reference-векторы ----------
    with SessionLocal() as db:
        refs = (
            db.query(Embedding.vec)
              .filter(Embedding.speaker_id == speaker_id)
              .all()
        )

    if not refs:
        raise RuntimeError(f"No embeddings for speaker_id={speaker_id}")

    ref_vecs = np.vstack([row[0] for row in refs])        # shape [N,192]

    # ---------- 3. cosine score ----------
    # возвращается [[score]], берём [0][0]
    score: float = cosine_similarity([probe_vec], ref_vecs).max()

    # порог можно вынести в settings
    matched = score > 0.75

    return {"score": score, "success": matched}

@celery_app.task(name="voiceid.extract")
def extract_task(user: str, speaker_id: str, wav_path: str):
    """Сохраняет вектор в БД."""
    # 1) вычисляем эмбеддинг
    vec = extract_embedding(wav_path)               # → numpy.ndarray(shape=(192,))
    # 2) сохраняем
    import asyncio, numpy as np
    async def _save():
        async with async_session_maker() as db:
            await add_embedding(db,
                                user=user,
                                speaker_id=speaker_id,
                                vec=vec.astype(np.float32).tolist())
    asyncio.run(_save())

@celery_app.task
def verify_speaker(user: str, audio_url: str) -> bool:
    return verify(user, audio_url)