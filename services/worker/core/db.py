import psycopg
from .config import DB_URL
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
