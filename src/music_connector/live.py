"""Turn a live tab-audio stream plus player events into per-track embeddings and listen events.

The browser sends raw PCM at its native rate and player events on the same ordered channel.
Audio only counts while a track is playing and no ad is showing. A seek drops the partial
window so one window never splices two parts of the song.
"""

from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from . import db
from .audio import MIN_TAIL_S
from .config import SAMPLE_RATE, WINDOW_S
from .embedder import Embedder, pool

# Windows quieter than this are pauses, muted tabs or buffering gaps, not music.
SILENCE_RMS = 1e-3
# Below this YouTube volume the signal is too quiet to rescale reliably.
MIN_VOLUME = 0.05
COMPLETE_FRACTION = 0.8
SKIP_SECONDS = 30.0


@dataclass
class Track:
    video_id: str
    title: str
    artist: str | None = None
    duration_s: float | None = None
    source: str = "ytmusic"
    window_starts: list[float] = field(default_factory=list)
    window_vecs: list[np.ndarray] = field(default_factory=list)
    listened_s: float = 0.0


@dataclass
class TrackResult:
    video_id: str
    title: str
    windows: int
    listened_s: float
    coverage: float | None
    event: str | None
    stored_embedding: bool


class ListenSession:
    def __init__(self, embedder: Embedder, sample_rate: int):
        self.embedder = embedder
        self.sr_in = sample_rate
        g = gcd(SAMPLE_RATE, sample_rate)
        self._up, self._down = SAMPLE_RATE // g, sample_rate // g
        self.window_in = int(WINDOW_S * sample_rate)
        self.track: Track | None = None
        self.playing = False
        self.ad = False
        self.volume = 1.0
        self._buf = np.empty(0, dtype=np.float32)
        self._buf_pos = 0.0
        self._last_window: np.ndarray | None = None

    # -- events ---------------------------------------------------------------------------

    def on_track(self, video_id: str, title: str, artist: str | None, duration_s: float | None,
                 source: str, position: float = 0.0) -> TrackResult | None:
        """A new track started. Returns the result of finalising the previous one, if any."""
        if self.track and self.track.video_id == video_id:
            self.on_seek(position)
            return None
        result = self.finish("next")
        self.track = Track(video_id, title, artist, duration_s, source)
        self._reset_buffer(position)
        return result

    def on_meta(self, video_id: str, title: str | None, artist: str | None) -> bool:
        """Late-arriving title/artist for the current track. Returns whether it applied."""
        if self.track is None or self.track.video_id != video_id or not title:
            return False
        self.track.title = title
        self.track.artist = artist or self.track.artist
        return True

    def on_state(self, playing: bool, position: float, volume: float | None = None,
                 duration_s: float | None = None) -> None:
        if volume is not None:
            self.volume = volume
        # YouTube often reports the duration only after the track has started.
        if duration_s and self.track is not None:
            self.track.duration_s = duration_s
        if playing and not self.playing:
            self._reset_buffer(position)
        self.playing = playing

    def on_seek(self, position: float) -> None:
        self._reset_buffer(position)

    def on_ad(self, active: bool) -> None:
        self.ad = active
        self._reset_buffer(self._buf_pos)

    def on_audio(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Buffer audio; returns any complete windows (already resampled) ready to embed."""
        if not (self.track and self.playing and not self.ad) or self.volume < MIN_VOLUME:
            return []
        # YouTube applies its slider and loudness normalisation as element volume; undoing
        # it keeps live embeddings comparable with full-scale local files.
        self._buf = np.concatenate([self._buf, pcm.astype(np.float32) / self.volume])
        ready = []
        while len(self._buf) >= self.window_in:
            raw, self._buf = self._buf[: self.window_in], self._buf[self.window_in:]
            start = self._buf_pos
            self._buf_pos += WINDOW_S
            self._last_window = raw
            if np.sqrt(np.mean(raw**2)) < SILENCE_RMS:
                continue
            self.track.listened_s += WINDOW_S
            self.track.window_starts.append(start)
            ready.append(self._resample(raw))
        return ready

    def add_vectors(self, vecs: np.ndarray) -> None:
        if self.track is not None:
            self.track.window_vecs.extend(vecs)

    def finish(self, reason: str) -> TrackResult | None:
        """Finalise the current track: embed any tail, pool, store, and log a listen event."""
        track = self.track
        if track is None:
            return None
        tail = self._tail_window(reason)
        if tail is not None:
            track.window_starts.append(max(self._buf_pos + len(self._buf) / self.sr_in - WINDOW_S, 0.0))
            track.listened_s += len(self._buf) / self.sr_in
            track.window_vecs.extend(self.embedder.embed_windows(tail[None, :]))
        self.track = None
        self._reset_buffer(0.0)
        return _store(track, len(track.window_vecs), self.embedder.model_id, reason)

    # -- internals --------------------------------------------------------------------------

    def _reset_buffer(self, position: float) -> None:
        self._buf = np.empty(0, dtype=np.float32)
        self._buf_pos = float(position)
        self._last_window = None

    def _tail_window(self, reason: str) -> np.ndarray | None:
        """On a natural end, the last few seconds become one window overlapping the previous one."""
        if reason != "ended" or self._last_window is None:
            return None
        if len(self._buf) < MIN_TAIL_S * self.sr_in:
            return None
        raw = np.concatenate([self._last_window, self._buf])[-self.window_in:]
        if np.sqrt(np.mean(raw**2)) < SILENCE_RMS:
            return None
        return self._resample(raw)

    def _resample(self, raw: np.ndarray) -> np.ndarray:
        if self._up == self._down:
            return raw.astype(np.float32)
        return resample_poly(raw, self._up, self._down).astype(np.float32)


def _event_for(track: Track, reason: str) -> str | None:
    if track.duration_s:
        if reason == "ended" or track.listened_s >= COMPLETE_FRACTION * track.duration_s:
            return "complete"
    if reason == "next" and track.listened_s < SKIP_SECONDS:
        return "skip"
    return "play" if track.listened_s >= SKIP_SECONDS else None


def _store(track: Track, n_windows: int, model_id: str, reason: str) -> TrackResult:
    event = _event_for(track, reason)
    coverage = (min(track.listened_s / track.duration_s, 1.0) if track.duration_s else None)
    stored = False
    with db.connect() as conn:
        song_id = db.upsert_song(conn, title=track.title, artist=track.artist,
                                 youtube_id=track.video_id, source=track.source,
                                 duration_s=track.duration_s)
        # The row may predate this listen (e.g. created by a like) without a duration, or with
        # a placeholder or stale title from before the player had this track's metadata.
        conn.execute("UPDATE songs SET duration_s = COALESCE(%s, duration_s) WHERE id = %s",
                     (track.duration_s, song_id))
        db.refresh_song_meta(conn, song_id, track.title, track.artist, track.video_id)
        if n_windows:
            existing = conn.execute(
                "SELECT coverage FROM songs WHERE id = %s AND embedding IS NOT NULL", (song_id,)
            ).fetchone()
            # Keep whichever listen heard more of the song.
            if existing is None or (coverage or 0.0) >= (existing[0] or 0.0):
                vecs = np.stack(track.window_vecs[:n_windows])
                song_vec, spread = pool(vecs)
                db.store_embedding(
                    conn, song_id, song_vec=song_vec, spread=spread, coverage=coverage or 0.0,
                    window_starts=track.window_starts[:n_windows], window_vecs=vecs, model=model_id,
                )
                stored = True
        if event == "complete":
            prior = conn.execute(
                "SELECT 1 FROM events WHERE song_id = %s AND kind = 'complete' LIMIT 1", (song_id,)
            ).fetchone()
            if prior:
                db.add_event(conn, song_id, "replay", source="extension")
        if event:
            db.add_event(conn, song_id, event, source="extension")
        conn.commit()
    return TrackResult(track.video_id, track.title, n_windows, track.listened_s, coverage,
                       event, stored)


def record_feedback(video_id: str, kind: str, title: str | None, artist: str | None,
                    source: str) -> None:
    with db.connect() as conn:
        song_id = db.upsert_song(conn, title=title or video_id, artist=artist,
                                 youtube_id=video_id, source=source)
        db.refresh_song_meta(conn, song_id, title, artist, video_id)
        db.add_event(conn, song_id, kind, source="extension")
        conn.commit()
