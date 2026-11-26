import numpy as np
import pytest


class FakeConn:
    """Answers the handful of read queries taste/recommend issue, keyed on SQL fragments."""

    def __init__(self, events=(), songs=None):
        # events: (song_id, kind, age_days); songs: {id: (title, unit vector)}
        self.events = list(events)
        self.songs = songs or {}

    def execute(self, sql, params=()):
        if "FROM events e JOIN songs" in sql:
            return _Result([(self.songs[i][1], k, age) for i, k, age in self.events if i in self.songs])
        if sql.startswith("SELECT embedding FROM songs WHERE embedding IS NOT NULL"):
            return _Result([(v,) for _, v in self.songs.values()])
        if "FROM events" in sql and "kind" in sql:
            return _Result(self.events)
        if "FROM songs WHERE id = ANY" in sql:
            ids = params[0]
            return _Result([(i, self.songs[i][0], self.songs[i][1]) for i in ids if i in self.songs])
        raise AssertionError(f"unexpected query: {sql}")


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


def unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_conn():
    return FakeConn


class FakeEmbedder:
    """Stands in for MuQ-MuLan: a fixed random projection of each window's magnitude spectrum,
    so windows that sound alike get similar vectors."""

    model_id = "fake/mulan"

    def __init__(self, dim: int = 32, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.normal(size=(64, dim))
        self.window_calls = 0

    def embed_windows(self, wins: np.ndarray) -> np.ndarray:
        self.window_calls += 1
        spec = np.abs(np.fft.rfft(wins, n=126, axis=1))[:, :64]
        v = np.log1p(spec) @ self.proj
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        v = self.rng.normal(size=(len(texts), self.proj.shape[1]))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def click_track(bpm: float = 120.0, seconds: float = 40.0, sr: int = 24_000) -> np.ndarray:
    """Clicks on every beat, louder on each bar's first beat, over a quiet tone that changes
    pitch halfway (two 'sections')."""
    t = np.arange(int(seconds * sr)) / sr
    tone = np.where(t < seconds / 2, np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)) * 0.05
    out = tone.astype(np.float32)
    period = 60.0 / bpm
    click = (np.hanning(240) * np.sin(2 * np.pi * 2000 * np.arange(240) / sr)).astype(np.float32)
    for i, start in enumerate(np.arange(0.0, seconds - 0.02, period)):
        a = int(start * sr)
        out[a : a + 240] += click * (0.9 if i % 4 == 0 else 0.5)
    return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
