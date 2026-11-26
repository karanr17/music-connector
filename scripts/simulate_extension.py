"""Drive `mc serve` with the extension's protocol using local audio files, faster than real time.

    uv run python scripts/simulate_extension.py FULL.mp3 SKIPPED.mp3 PARTIAL.mp3

Scenarios: a full listen at 50% YouTube volume, a skip after 20 s, and a partial listen with a
seek, an ad break, a pause and a like, ended by disconnecting. Songs are stored under
youtube_id SIM-1..3; `--cleanup` deletes them afterwards.
"""

import argparse
import asyncio
import json

import numpy as np
import psycopg
from websockets.asyncio.client import connect

from music_connector.audio import decode
from music_connector.config import DATABASE_URL

SR = 48_000
CHUNK = SR // 10


async def stream(ws, wav: np.ndarray, start_s: float, seconds: float, gain: float = 1.0):
    a, b = int(start_s * SR), int(min((start_s + seconds) * SR, len(wav)))
    for i in range(a, b, CHUNK):
        await ws.send((wav[i:min(i + CHUNK, b)] * gain).astype(np.float32).tobytes())


async def send(ws, **msg):
    await ws.send(json.dumps(msg))


async def run(url: str, files: list[str]) -> None:
    full, skipped, partial = (decode(f, SR) for f in files)
    dur = [len(w) / SR for w in (full, skipped, partial)]

    async with connect(url, max_size=2**24) as ws:
        await send(ws, type="hello", sampleRate=SR)

        await send(ws, type="track", videoId="SIM-1", title="sim full listen", artist="sim",
                   duration=None, position=0, source="ytmusic")
        await send(ws, type="state", playing=True, position=0, volume=0.5, duration=dur[0])
        await stream(ws, full, 0, dur[0], gain=0.5)
        await send(ws, type="end", reason="ended")

        await send(ws, type="track", videoId="SIM-2", title="sim skipped", artist="sim",
                   duration=dur[1], position=0, source="ytmusic")
        await send(ws, type="state", playing=True, position=0, volume=1.0, duration=dur[1])
        await stream(ws, skipped, 0, 20)

        await send(ws, type="track", videoId="SIM-3", title="sim partial", artist="sim",
                   duration=dur[2], position=0, source="youtube")
        await send(ws, type="state", playing=True, position=0, volume=1.0, duration=dur[2])
        await stream(ws, partial, 0, 60)
        await send(ws, type="seek", position=120)
        await stream(ws, partial, 120, 40)
        await send(ws, type="ad", active=True)
        await ws.send((np.random.default_rng(0).standard_normal(SR * 15) * 0.3)
                      .astype(np.float32).tobytes())
        await send(ws, type="ad", active=False)
        await send(ws, type="state", playing=False, position=160, volume=1.0)
        await ws.send(np.zeros(SR * 5, dtype=np.float32).tobytes())
        await send(ws, type="state", playing=True, position=160, volume=1.0)
        await stream(ws, partial, 160, 25)
        await send(ws, type="feedback", videoId="SIM-3", kind="like", title="sim partial",
                   artist="sim", source="youtube")
        # Give the server time to embed queued windows before the socket closes.
        await asyncio.sleep(1)


def report(cleanup: bool) -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """SELECT s.youtube_id, s.duration_s, s.coverage,
                      (SELECT count(*) FROM song_windows w WHERE w.song_id = s.id),
                      (SELECT array_agg(w.start_s ORDER BY w.start_s) FROM song_windows w
                        WHERE w.song_id = s.id),
                      (SELECT string_agg(kind, ',' ORDER BY id) FROM events e WHERE e.song_id = s.id)
               FROM songs s WHERE youtube_id LIKE 'SIM-%%' ORDER BY youtube_id"""
        ).fetchall()
        for vid, d, cov, n, starts, events in rows:
            cov_s = f"{cov:.2f}" if cov is not None else "-"
            print(f"{vid}: duration {d:.0f}s coverage {cov_s} windows {n} events [{events}]")
            print(f"    window starts: {starts}")
        if cleanup:
            conn.execute("DELETE FROM songs WHERE youtube_id LIKE 'SIM-%%'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs=3)
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.url, args.files))
    report(args.cleanup)


if __name__ == "__main__":
    main()
