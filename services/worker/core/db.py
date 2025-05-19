import psycopg
from .config import settings

import os, asyncio
from contextlib import asynccontextmanager, contextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine  
import numpy as np, json, pathlib, os

BASE = pathlib.Path("/app/profiles"); BASE.mkdir(exist_ok=True)


DB_URL_ASYNC = os.getenv("DB_URL_ASYNC")  #  postgres+asyncpg://user:pwd@db:5432/voiceid
DB_URL_SYNC  = os.getenv("DB_URL_SYNC")   #  postgres+psycopg://user:pwd@db:5432/voiceid

async_engine = create_async_engine(settings.DB_URL_ASYNC, echo=False)
sync_engine  = create_engine(settings.DB_URL_SYNC, echo=False)

AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
SessionLocal      = sessionmaker(bind=sync_engine, expire_on_commit=False)

async_session_maker = async_sessionmaker(
    async_engine, expire_on_commit=False
)


@asynccontextmanager
async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@contextmanager
def get_sync_session() -> Session:
    with SessionLocal() as session:
        yield session

def save_profile(user_id, tier, vecs):
    embedding = np.stack(vecs).mean(axis=0).tolist()
    with psycopg.connect(DB_URL) as conn:
        conn.execute(
          "INSERT INTO voices(user_id, tier, embedding) VALUES(%s,%s,%s)",
          (user_id, tier, embedding)
        )
def load_profile(user_id, tier):
    with psycopg.connect(DB_URL) as conn:
        row = conn.execute(
          "SELECT embedding FROM voices WHERE user_id=%s AND tier=%s",
          (user_id, tier)
        ).fetchone()
    return np.array(row[0])

def _path(uid: str, tier: str):
    return BASE / f"{uid}_{tier}.json"

def save_profile(uid: str, tier: str, vecs):
    with _path(uid, tier).open("w") as f:
        json.dump([v.tolist() for v in vecs], f)


def load_profile(uid: str, tier: str):
    p = _path(uid, tier)
    if not p.exists(): return None
    return np.array(json.load(p), dtype=np.float32)