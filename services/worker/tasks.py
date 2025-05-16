import os, asyncio, uuid, torch
from celery import Celery
from services.worker.core.db import get_sync_session
from services.worker.core.model import Embedding, Base
from services.worker.core import preprocess, extract_embedding, verify
from services.worker.core.celery_app import celery_app
from services.worker.core.preprocessing import denoise_and_split
from services.worker.core.model import get_speaker_model as get_model
from services.worker.core.scoring import score_embedding
from services.worker.core.db import save_profile, load_profile
from celery_app import celery_app
from services.worker.core.model import extractor            # SpeakerRecognition wrapper
from services.worker.core.db import async_session_maker     # ваш create_async_engine(...)
from crud import add_embedding

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

@celery_app.task
def enroll_task(user_id: str, wav_path: str, tier: str):
    segments = denoise_and_split(wav_path)
    model = get_model()
    vecs = [ model.encode(seg) for seg in segments[:5] ]
    save_profile(user_id, tier, vecs)
    return True

@celery_app.task
def verify_task(user_id: str, wav_path: str, tier: str):
    segments = denoise_and_split(wav_path)
    model = get_model()
    emb = model.encode(segments)
    profile = load_profile(user_id, tier)
    score, match = score_embedding(emb, profile, tier)
    return {"score": score, "match": match}

@celery_app.task(name="voiceid.extract")
def extract_task(user: str, speaker_id: str, wav_path: str):
    """Сохраняет вектор в БД."""
    # 1) вычисляем эмбеддинг
    vec = extractor(wav_path)               # → numpy.ndarray(shape=(192,))
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