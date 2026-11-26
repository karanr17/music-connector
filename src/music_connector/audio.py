import subprocess
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, WINDOW_S

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg", ".opus", ".webm", ".aiff"}
MIN_TAIL_S = 3.0


def decode(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at `sr`."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def windows(wav: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S
            ) -> tuple[np.ndarray, list[float]]:
    """Split into back-to-back windows. Returns (windows [n, samples], start times in seconds).

    A leftover tail of at least MIN_TAIL_S becomes one extra window aligned to the end of the
    song (overlapping the previous one) rather than being zero-padded, since silence padding
    shifts the embedding. Clips shorter than one window are looped to fill it.
    """
    size = int(sr * window_s)
    if len(wav) < size:
        reps = int(np.ceil(size / max(len(wav), 1)))
        return np.tile(wav, reps)[:size][None, :], [0.0]
    n_full = len(wav) // size
    starts = [i * size for i in range(n_full)]
    if len(wav) - n_full * size >= MIN_TAIL_S * sr:
        starts.append(len(wav) - size)
    return np.stack([wav[s:s + size] for s in starts]), [s / sr for s in starts]


def find_audio_files(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.suffix.lower() in AUDIO_EXTS:
            found.append(p)
    return found
