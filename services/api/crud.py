# services/worker/crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select
from services.worker.core.models import Embedding  # ORM‑модель

async def add_embedding(db: AsyncSession, *, user: str, speaker_id: str, vec):
    stmt = insert(Embedding).values(
        user=user,
        speaker_id=speaker_id,
        vec=vec,          # pgvector.VECTOR
    )
    await db.execute(stmt)
    await db.commit()

async def get_by_user(db: AsyncSession, user: str):
    q = select(Embedding).where(Embedding.user == user)
    res = await db.execute(q)
    return res.scalar_one_or_none()
