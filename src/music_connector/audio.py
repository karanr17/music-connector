import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, WINDOW_S

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg", ".opus", ".webm", ".aiff"}
MIN_TAIL_S = 3.0


def _ffmpeg_args(src: str, sr: int) -> list[str]:
    return ["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]


def decode(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at `sr`."""
    out = subprocess.run(_ffmpeg_args(str(path), sr), check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def decode_bytes(data: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an in-memory audio file. Tries a pipe first; MP4/M4A files with the index at the
    end can't be read from a pipe, so those go through a private temp file that is deleted
    straight away."""
    try:
        out = subprocess.run(
            _ffmpeg_args("pipe:0", sr),
            input=data,
            check=True,
            capture_output=True,
        ).stdout
        if out:
            return np.frombuffer(out, dtype=np.float32).copy()
    except subprocess.CalledProcessError:
        pass
    fd, tmp = tempfile.mkstemp(prefix="mc-analyze-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return decode(tmp, sr)
    finally:
        os.unlink(tmp)


def windows(
    wav: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S
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
    return np.stack([wav[s : s + size] for s in starts]), [s / sr for s in starts]


def hop_windows(
    wav: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S, hop_s: float = WINDOW_S / 2
) -> tuple[np.ndarray, list[float]]:
    """Overlapping windows every `hop_s`. As in `windows`, the last window is aligned to the end
    of the song instead of padded, and clips shorter than one window are looped."""
    size, hop = int(sr * window_s), int(sr * hop_s)
    if len(wav) <= size:
        return windows(wav, sr, window_s)
    starts = list(range(0, len(wav) - size + 1, hop))
    if len(wav) - (starts[-1] + size) >= hop // 2:
        starts.append(len(wav) - size)
    return np.stack([wav[s : s + size] for s in starts]), [s / sr for s in starts]


def find_audio_files(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.suffix.lower() in AUDIO_EXTS:
            found.append(p)
    return found
