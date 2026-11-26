"""Local WebSocket server the browser extension streams to.

Protocol (one connection per captured tab, messages processed strictly in order):
  text   {"type": "hello", "sampleRate": 48000}
  text   {"type": "track", "videoId", "title", "artist", "duration", "position", "source"}
  text   {"type": "meta", "videoId", "title", "artist"}   (title/artist arrived after "track")
  text   {"type": "state", "playing": bool, "position": float, "volume": float, "duration"}
  text   {"type": "seek", "position": float}
  text   {"type": "ad", "active": bool}
  text   {"type": "feedback", "videoId", "kind": "like"|"dislike", "title", "artist", "source"}
  text   {"type": "end", "reason": "ended"|"closed"}
  binary mono float32 PCM at the hello sample rate
"""

import asyncio
import json
import logging

import numpy as np
from websockets.asyncio.server import serve
from websockets.http11 import Response

from .embedder import Embedder
from .live import ListenSession, TrackResult, record_feedback

log = logging.getLogger("music_connector.server")
FEEDBACK_KINDS = {"like", "dislike"}


def _describe(r: TrackResult) -> str:
    cov = f"{r.coverage:.0%}" if r.coverage is not None else "?"
    return (
        f"{r.title!r}: {r.windows} windows, heard {r.listened_s:.0f}s ({cov}), "
        f"event={r.event or '-'}, embedding {'saved' if r.stored_embedding else 'kept'}"
    )


async def _handle(ws, embedder: Embedder, embed_lock: asyncio.Lock) -> None:
    session: ListenSession | None = None
    log.info("extension connected from %s", ws.remote_address)

    async def run(fn, *args):
        # MuQ-MuLan on MPS isn't safe to call from several threads at once.
        async with embed_lock:
            return await asyncio.to_thread(fn, *args)

    try:
        async for message in ws:
            if isinstance(message, bytes):
                if session is None:
                    continue
                pcm = np.frombuffer(message, dtype=np.float32)
                windows = session.on_audio(pcm)
                if windows:
                    vecs = await run(embedder.embed_windows, np.stack(windows))
                    session.add_vectors(vecs)
                continue

            msg = json.loads(message)
            kind = msg.get("type")
            if kind == "hello":
                session = ListenSession(embedder, int(msg["sampleRate"]))
                log.info("capturing at %s Hz", msg["sampleRate"])
            elif session is None:
                continue
            elif kind == "track":
                if msg.get("title"):
                    log.info("now playing: %s – %s", msg.get("artist"), msg.get("title"))
                result = await run(
                    session.on_track,
                    msg["videoId"],
                    msg.get("title") or msg["videoId"],
                    msg.get("artist"),
                    msg.get("duration"),
                    msg.get("source", "ytmusic"),
                    float(msg.get("position") or 0),
                )
                if result:
                    log.info("finished %s", _describe(result))
            elif kind == "meta":
                if session.on_meta(msg["videoId"], msg.get("title"), msg.get("artist")):
                    log.info("now playing: %s – %s", msg.get("artist"), msg.get("title"))
            elif kind == "state":
                session.on_state(
                    bool(msg["playing"]),
                    float(msg.get("position") or 0),
                    msg.get("volume"),
                    msg.get("duration"),
                )
            elif kind == "seek":
                session.on_seek(float(msg["position"]))
            elif kind == "ad":
                session.on_ad(bool(msg["active"]))
                log.info("ad %s", "started" if msg["active"] else "ended")
            elif kind == "feedback" and msg.get("kind") in FEEDBACK_KINDS:
                await asyncio.to_thread(
                    record_feedback,
                    msg["videoId"],
                    msg["kind"],
                    msg.get("title"),
                    msg.get("artist"),
                    msg.get("source", "ytmusic"),
                )
                log.info("%s: %s", msg["kind"], msg.get("title"))
            elif kind == "end":
                result = await run(session.finish, msg.get("reason", "closed"))
                if result:
                    log.info("finished %s", _describe(result))
    finally:
        if session is not None and session.track is not None:
            result = await run(session.finish, "closed")
            if result:
                log.info("finished on disconnect %s", _describe(result))
        log.info("extension disconnected")


def _only_extensions(connection, request) -> Response | None:
    """Web pages can open WebSockets to localhost too; only the extension (or a local script,
    which sends no Origin) may feed listening data."""
    origin = request.headers.get("Origin")
    if origin is None or origin.startswith("chrome-extension://"):
        return None
    log.warning("rejected connection from origin %s", origin)
    return connection.respond(403, "forbidden origin\n")


async def serve_forever(host: str, port: int) -> None:
    embedder = Embedder()
    embed_lock = asyncio.Lock()
    log.info("model %s ready on %s", embedder.model_id, embedder.device)
    async with serve(
        lambda ws: _handle(ws, embedder, embed_lock),
        host,
        port,
        max_size=2**24,
        process_request=_only_extensions,
    ):
        log.info("listening on ws://%s:%d", host, port)
        await asyncio.Future()
