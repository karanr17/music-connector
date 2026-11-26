"""Time MuQ-MuLan on this machine and project the cost of embedding N full songs."""

import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from .audio import decode, windows
from .config import SAMPLE_RATE
from .embedder import Embedder, pool

console = Console()


def _synthetic_song(seconds: float, seed: int = 0) -> np.ndarray:
    # Content doesn't affect speed; this only stands in when no real files are given.
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = sum(np.sin(2 * np.pi * f * t) for f in (110, 220, 330, 440))
    return (0.1 * tone + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def _chip() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return platform.processor()


def run(
    files: list[Path],
    song_seconds: float,
    batch_sizes: list[int],
    repeats: int,
    project_songs: int,
    device: str | None,
) -> None:
    console.print(f"[bold]Machine:[/] {_chip()} · torch {torch.__version__}")

    t0 = time.perf_counter()
    emb = Embedder(device=device)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in emb.model.parameters()) / 1e6
    console.print(
        f"[bold]Model:[/] {emb.model_id} ({n_params:.0f}M params) on "
        f"[cyan]{emb.device}[/], loaded in {load_s:.1f}s"
    )

    decode_times = []
    if files:
        songs = []
        for f in files:
            t0 = time.perf_counter()
            wav = decode(f)
            decode_times.append(time.perf_counter() - t0)
            songs.append((f.name, wav))
        console.print(f"Decoded {len(files)} file(s): mean {np.mean(decode_times):.2f}s per file")
    else:
        songs = [(f"synthetic {song_seconds:.0f}s", _synthetic_song(song_seconds))]

    wins, _ = windows(songs[0][1])
    emb.embed_windows(wins[: max(batch_sizes)])  # warm-up: MPS compiles kernels on first use
    emb.sync()

    table = Table(title="Embedding speed (per song, excludes decode)")
    for col in ("batch", "song", "audio", "windows", "embed time", "× real-time"):
        table.add_column(col, justify="right")
    best = None
    for bs in batch_sizes:
        emb.batch_size = bs
        for name, wav in songs:
            wins, _ = windows(wav)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                vecs = emb.embed_windows(wins)
                emb.sync()
                times.append(time.perf_counter() - t0)
            t = float(np.median(times))
            audio_s = len(wav) / SAMPLE_RATE
            table.add_row(
                str(bs), name[:40], f"{audio_s:.0f}s", str(len(wins)), f"{t:.2f}s", f"{audio_s / t:.0f}×"
            )
            per_audio_second = t / audio_s
            if best is None or per_audio_second < best[1]:
                best = (bs, per_audio_second)
    console.print(table)

    song_vec, spread = pool(vecs)
    console.print(
        f"Sanity: song vector norm {np.linalg.norm(song_vec):.3f}, "
        f"window spread {spread:.3f}, dim {song_vec.shape[0]}"
    )

    if emb.device == "mps":
        console.print(f"MPS memory: {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB allocated")

    bs, per_s = best
    avg_song_s = 210.0
    embed_h = per_s * avg_song_s * project_songs / 3600
    decode_h = (np.mean(decode_times) if decode_times else 0.25) * project_songs / 3600
    console.print(
        f"\n[bold]Projection for {project_songs:,} songs × {avg_song_s / 60:.1f} min[/] "
        f"(best batch size {bs}):\n"
        f"  embedding ≈ [green]{embed_h:.1f} h[/]   decode ≈ {decode_h:.1f} h"
        f"{'' if decode_times else ' (assumed 0.25 s/file)'}\n"
        f"  live capture headroom: model runs {1 / per_s:.0f}× faster than playback"
    )
