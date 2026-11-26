import os
import shutil
import subprocess
import sys
from pathlib import Path

import mutagen
from rich.progress import Progress

from . import db
from .audio import AUDIO_EXTS, decode, find_audio_files
from .embedder import Embedder


def _tags(path: Path) -> dict:
    try:
        f = mutagen.File(path, easy=True)
    except mutagen.MutagenError:
        f = None
    tags = dict(f.tags or {}) if f is not None and f.tags is not None else {}

    def first(key):
        v = tags.get(key)
        return v[0] if v else None

    year = first("date") or first("originaldate")
    return {
        "title": first("title") or path.stem,
        "artist": first("artist"),
        "release_year": int(year[:4]) if year and year[:4].isdigit() else None,
    }


def _prevent_idle_sleep() -> None:
    """macOS suspends the process on idle sleep, which turns minutes of work into hours.
    `caffeinate -w` holds the assertion until this process exits. Closing the lid on
    battery still sleeps."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def ingest_files(paths: list[Path], pools: list[str], force: bool = False) -> int:
    # Every check that can fail runs before the model load, which takes 10-40 s.
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"no such file or folder: {', '.join(missing)}")
    files = find_audio_files(paths)
    if not files:
        raise FileNotFoundError(f"no audio files ({', '.join(sorted(AUDIO_EXTS))}) found")
    with db.connect() as conn:
        _prevent_idle_sleep()
        embedder = Embedder()
        return _embed_all(conn, embedder, files, pools, force)


def _embed_all(conn, embedder: Embedder, files: list[Path], pools: list[str], force: bool) -> int:
    done = 0
    with Progress() as progress:
        task = progress.add_task("Embedding", total=len(files))
        for path in files:
            local_path = str(path.resolve())
            row = conn.execute(
                "SELECT id, embedding_model FROM songs WHERE local_path = %s", (local_path,)
            ).fetchone()
            if row and row[1] == embedder.model_id and not force:
                progress.advance(task)
                continue
            try:
                wav = decode(path)
                if not len(wav):
                    raise ValueError("decoded to no audio")
            except Exception as e:  # noqa: BLE001 - one unreadable file shouldn't stop a batch
                progress.console.print(f"[red]skip[/] {path.name}: {e}")
                progress.advance(task)
                continue
            emb = embedder.embed_wav(wav)
            song_id = row[0] if row else db.upsert_song(
                conn, **_tags(path), local_path=local_path, pools=pools, source="local",
            )
            db.store_embedding(
                conn, song_id, song_vec=emb.song_vec, spread=emb.spread, coverage=1.0,
                window_starts=emb.window_starts, window_vecs=emb.window_vecs,
                model=embedder.model_id, duration_s=emb.duration_s,
            )
            conn.commit()
            done += 1
            progress.advance(task)
    return done
