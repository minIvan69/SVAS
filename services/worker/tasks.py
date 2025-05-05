from core.celery_app import celery_app
from core.preprocessing import denoise_and_split
from core.model import get_model
from core.scoring import score_embedding
from core.db import save_profile, load_profile

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
