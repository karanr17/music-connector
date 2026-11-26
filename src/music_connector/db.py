from contextlib import contextmanager

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL


def as_array(v) -> np.ndarray:
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


@contextmanager
def connect():
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        yield conn


def upsert_song(conn, *, title, artist=None, youtube_id=None, local_path=None,
                release_year=None, pools=(), source, duration_s=None) -> int:
    """Insert a song or return the existing row's id (matched on youtube_id or local_path)."""
    key_col, key_val = ("youtube_id", youtube_id) if youtube_id else ("local_path", local_path)
    if key_val is None:
        raise ValueError("a song needs a youtube_id or a local_path")
    row = conn.execute(f"SELECT id FROM songs WHERE {key_col} = %s", (key_val,)).fetchone()
    if row:
        return row[0]
    return conn.execute(
        """INSERT INTO songs (title, artist, youtube_id, local_path, release_year, pools, source, duration_s)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (title, artist, youtube_id, local_path, release_year, list(pools), source, duration_s),
    ).fetchone()[0]


def store_embedding(conn, song_id: int, *, song_vec: np.ndarray, spread: float, coverage: float,
                    window_starts: list[float], window_vecs: np.ndarray, model: str,
                    duration_s: float | None = None) -> None:
    conn.execute(
        """UPDATE songs SET embedding = %s, embedding_spread = %s, coverage = %s,
                  embedding_model = %s, duration_s = COALESCE(%s, duration_s)
           WHERE id = %s""",
        (song_vec, spread, coverage, model, duration_s, song_id),
    )
    conn.execute("DELETE FROM song_windows WHERE song_id = %s", (song_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO song_windows (song_id, start_s, embedding) VALUES (%s, %s, %s)",
            [(song_id, s, v) for s, v in zip(window_starts, window_vecs, strict=True)],
        )


def add_event(conn, song_id: int, kind: str, *, at=None, source: str = "manual") -> None:
    conn.execute(
        "INSERT INTO events (song_id, kind, at, source) VALUES (%s, %s, COALESCE(%s, now()), %s)",
        (song_id, kind, at, source),
    )
