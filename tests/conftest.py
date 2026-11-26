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
