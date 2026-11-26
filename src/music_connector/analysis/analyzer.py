import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..audio import decode, decode_bytes, hop_windows
from ..config import MODEL_ID, SAMPLE_RATE, RecommendConfig
from . import descriptors as desc
from . import dsp
from .schema import SongAnalysis
from .segments import SegmentConfig, segment

log = logging.getLogger("music_connector.analysis")

# Bump when the same audio would produce different output: it invalidates cached analyses.
ANALYZER_VERSION = "1.0.0"
WINDOW_S = 10.0
HOP_S = 5.0
SONG_TOP_K = 3
SEGMENT_TOP_K = 2
CACHE_DIR = Path(os.environ.get("MC_CACHE_DIR", Path.home() / ".cache" / "music-connector")) / "analysis"


@dataclass(frozen=True)
class AnalyzeOptions:
    windows: bool = False
    nearest: int = 0

    def key(self, calibration: str) -> str:
        blob = json.dumps([ANALYZER_VERSION, calibration, asdict(self)], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _times(x) -> list[float]:
    return [round(float(t), 3) for t in x]


def _series(x) -> list[float]:
    return [round(float(v), 3) for v in x]


NearestFn = Callable[[np.ndarray, int], list[dict]]


def library_nearest(vec: np.ndarray, k: int) -> list[dict]:
    from .. import db

    # Skip re-uploads of this very recording, the same rule the playlist builder uses.
    dup = RecommendConfig().duplicate_cosine
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT id, title, artist, 1 - (embedding <=> %s) AS cos FROM songs
               WHERE embedding IS NOT NULL ORDER BY embedding <=> %s LIMIT %s""",
            (vec, vec, k + 5),
        ).fetchall()
    return [
        {"songId": r[0], "title": r[1], "artist": r[2], "cosine": round(float(r[3]), 4)}
        for r in rows
        if r[3] < dup
    ][:k]


class SongAnalyzer:
    """Turns audio into a SongAnalysis. The model is loaded on first use, or shared with
    `mc serve` by passing its Embedder."""

    def __init__(
        self,
        embedder=None,
        calibration: desc.Calibration | None = None,
        cache_dir: Path | None = CACHE_DIR,
        nearest_fn: NearestFn = library_nearest,
        segment_config: SegmentConfig = SegmentConfig(),
    ):
        self._embedder = embedder
        self.calibration = calibration if calibration is not None else desc.load_calibration()
        self.cache_dir = cache_dir
        self.nearest_fn = nearest_fn
        self.segment_config = segment_config
        self._text: np.ndarray | None = None

    @property
    def embedder(self):
        if self._embedder is None:
            from ..embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

    @property
    def calibration_name(self) -> str:
        return self.calibration.name if self.calibration else "uncalibrated"

    def text_vectors(self) -> np.ndarray:
        if self._text is None:
            self._text = self.embedder.embed_text([p for _, _, p in desc.prompts()])
        return self._text

    def cache_path(self, sha: str, options: AnalyzeOptions) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sha}.{options.key(self.calibration_name)}.json"

    def cached(self, sha: str, options: AnalyzeOptions) -> SongAnalysis | None:
        p = self.cache_path(sha, options)
        if p and p.exists():
            try:
                return SongAnalysis.model_validate_json(p.read_text())
            except ValueError:
                log.warning("ignoring unreadable cache entry %s", p)
        return None

    def _store(self, a: SongAnalysis, options: AnalyzeOptions) -> None:
        p = self.cache_path(a.source.sha256, options)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(a.to_json())
            tmp.replace(p)

    def analyze_file(
        self, path: Path, options: AnalyzeOptions = AnalyzeOptions(), use_cache: bool = True
    ) -> SongAnalysis:
        path = Path(path)
        sha = sha256_file(path)
        if use_cache and (hit := self.cached(sha, options)):
            return hit
        a = self.analyze_wav(decode(path), sha256=sha, file_name=path.name, options=options)
        self._store(a, options)
        return a

    def analyze_bytes(
        self,
        data: bytes,
        file_name: str | None = None,
        options: AnalyzeOptions = AnalyzeOptions(),
        use_cache: bool = True,
    ) -> SongAnalysis:
        sha = hashlib.sha256(data).hexdigest()
        if use_cache and (hit := self.cached(sha, options)):
            return hit
        a = self.analyze_wav(decode_bytes(data), sha256=sha, file_name=file_name, options=options)
        self._store(a, options)
        return a

    def analyze_wav(
        self, wav: np.ndarray, *, sha256: str, file_name: str | None, options: AnalyzeOptions
    ) -> SongAnalysis:
        if len(wav) == 0:
            raise ValueError("audio decoded to zero samples")
        t0 = time.perf_counter()
        sr = SAMPLE_RATE
        duration = len(wav) / sr
        d = dsp.analyse(wav, sr)
        t1 = time.perf_counter()

        wins, starts = hop_windows(wav, sr, WINDOW_S, HOP_S)
        vecs = self.embedder.embed_windows(wins)
        text = self.text_vectors()
        t2 = time.perf_counter()

        segs = segment(
            duration=duration,
            beats=d.beats,
            downbeats=d.downbeats,
            beat_features=d.beat_features,
            window_starts=np.asarray(starts),
            window_vecs=vecs,
            window_s=WINDOW_S,
            rms=d.energy.rms,
            rms_rate=d.energy.frame_rate,
            cfg=self.segment_config,
        )

        cal = self.calibration
        song_vec = vecs.mean(axis=0)
        song_vec /= np.linalg.norm(song_vec)
        song_cos = song_vec @ text.T
        song_z = desc.zscores(song_cos, *((cal.song_mean, cal.song_std) if cal else (None, None)))
        wmean, wstd = (cal.window_mean, cal.window_std) if cal else (None, None)
        seg_cos = np.array([s.vector for s in segs]) @ text.T
        seg_z = desc.zscores(seg_cos, wmean, wstd)

        windows = None
        if options.windows:
            win_cos = vecs @ text.T
            win_z = desc.zscores(win_cos, wmean, wstd)
            windows = {
                "windowS": WINDOW_S,
                "hopS": HOP_S,
                "starts": _times(starts),
                "descriptors": [desc.top_k(c, z, 1) for c, z in zip(win_cos, win_z, strict=True)],
            }

        nearest = None
        if options.nearest > 0:
            try:
                nearest = self.nearest_fn(song_vec.astype(np.float32), options.nearest)
            except Exception as e:  # The library DB is optional; analysis must not fail on it.
                log.warning("nearest songs unavailable: %s", e)
                nearest = []

        e = d.energy
        result = SongAnalysis.model_validate(
            {
                "analyzer": {
                    "version": ANALYZER_VERSION,
                    "model": getattr(self.embedder, "model_id", MODEL_ID),
                    "calibration": self.calibration_name,
                },
                "source": {
                    "sha256": sha256,
                    "fileName": file_name,
                    "durationS": round(duration, 3),
                    "sampleRate": sr,
                },
                "tempo": {
                    "bpm": round(d.bpm, 2),
                    "confidence": round(d.beat_confidence, 2),
                    "beatsPerBar": 4,
                },
                "beats": _times(d.beats),
                "downbeats": _times(d.downbeats),
                "onsets": _times(d.onsets),
                "energy": {
                    "frameRate": e.frame_rate,
                    "loudnessDb": round(e.loudness_db, 2),
                    "rms": _series(e.rms),
                    "low": _series(e.low),
                    "mid": _series(e.mid),
                    "high": _series(e.high),
                },
                "segments": [
                    {
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "label": s.label,
                        "group": s.letter,
                        "variant": s.variant,
                        "similarity": None if s.similarity is None else round(s.similarity, 3),
                        "boundaryStrength": round(s.boundary_strength, 3),
                        "energy": round(s.energy, 3),
                        "descriptors": desc.top_k(c, z, SEGMENT_TOP_K),
                    }
                    for s, c, z in zip(segs, seg_cos, seg_z, strict=True)
                ],
                "descriptors": desc.top_k(song_cos, song_z, SONG_TOP_K),
                "windows": windows,
                "nearest": nearest,
            }
        )
        log.info(
            "analysed %s: %.0f s audio, dsp %.1f s, model %.1f s, total %.1f s",
            file_name or sha256[:12],
            duration,
            t1 - t0,
            t2 - t1,
            time.perf_counter() - t0,
        )
        return result
