from fastapi import Depends
from services.worker.core.db import async_session_maker

async def get_async_session():
    async with async_session_maker() as s:
        yield s