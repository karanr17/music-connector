"""Local server: the browser extension's WebSocket and the analysis HTTP API, on one port.

WebSocket at / (one connection per captured tab, messages processed strictly in order):
  text   {"type": "hello", "sampleRate": 48000}
  text   {"type": "track", "videoId", "title", "artist", "duration", "position", "source"}
  text   {"type": "meta", "videoId", "title", "artist"}   (title/artist arrived after "track")
  text   {"type": "state", "playing": bool, "position": float, "volume": float, "duration"}
  text   {"type": "seek", "position": float}
  text   {"type": "ad", "active": bool}
  text   {"type": "feedback", "videoId", "kind": "like"|"dislike", "title", "artist", "source"}
  text   {"type": "end", "reason": "ended"|"closed"}
  binary mono float32 PCM at the hello sample rate

HTTP (see README "Analysis sidecar"):
  GET  /health                     {"ok": true, "schemaVersion": "1.0", ...}
  POST /analyze                    body = audio bytes, or JSON {"path": "/abs/file.mp3"}
  GET  /analyze?path=/abs/file.mp3
  Query options for both: windows=1, nearest=N, refresh=1, name=<file name for byte uploads>.
"""

import asyncio
import json
import logging
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .analysis import SCHEMA_VERSION, AnalyzeOptions, SongAnalyzer
from .audio import AUDIO_EXTS
from .live import ListenSession, TrackResult, record_feedback

log = logging.getLogger("music_connector.server")
FEEDBACK_KINDS = {"like", "dislike"}
MAX_UPLOAD_BYTES = 256 * 2**20
# The visual music player's Vite dev server. MC_ALLOWED_ORIGINS adds more (comma-separated).
DEFAULT_ANALYZE_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

EMBEDDER = web.AppKey("embedder", object)
ANALYZER = web.AppKey("analyzer", object)
EMBED_LOCK = web.AppKey("embed_lock", asyncio.Lock)
ANALYZE_LOCK = web.AppKey("analyze_lock", asyncio.Lock)
ORIGINS = web.AppKey("origins", frozenset)


def analyze_origins() -> frozenset[str]:
    extra = [o.strip().rstrip("/") for o in os.environ.get("MC_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return frozenset((*DEFAULT_ANALYZE_ORIGINS, *extra))


def _describe(r: TrackResult) -> str:
    cov = f"{r.coverage:.0%}" if r.coverage is not None else "?"
    return (
        f"{r.title!r}: {r.windows} windows, heard {r.listened_s:.0f}s ({cov}), "
        f"event={r.event or '-'}, embedding {'saved' if r.stored_embedding else 'kept'}"
    )


async def _ws_messages(ws: web.WebSocketResponse) -> AsyncIterator[bytes | str]:
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            yield msg.data
        elif msg.type == WSMsgType.TEXT:
            yield msg.data
        elif msg.type == WSMsgType.ERROR:
            log.warning("websocket error: %s", ws.exception())
            return


async def handle_stream(messages: AsyncIterator[bytes | str], embedder, embed_lock: asyncio.Lock) -> None:
    session: ListenSession | None = None

    async def run(fn, *args):
        async with embed_lock:
            return await asyncio.to_thread(fn, *args)

    try:
        async for message in messages:
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


async def websocket(request: web.Request) -> web.StreamResponse:
    origin = request.headers.get("Origin")
    # Web pages can open WebSockets to localhost too; only the extension (or a local script,
    # which sends no Origin) may feed listening data.
    if origin is not None and not origin.startswith("chrome-extension://"):
        log.warning("rejected websocket from origin %s", origin)
        return web.Response(status=403, text="forbidden origin\n")
    ws = web.WebSocketResponse(max_msg_size=2**24)
    if not ws.can_prepare(request).ok:
        return web.Response(status=426, text="music-connector: WebSocket endpoint; see /health\n")
    await ws.prepare(request)
    log.info("extension connected from %s", request.remote)
    try:
        await handle_stream(_ws_messages(ws), request.app[EMBEDDER], request.app[EMBED_LOCK])
    finally:
        log.info("extension disconnected")
    return ws


def _cors_headers(origin: str | None) -> dict[str, str]:
    if origin is None:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }


@web.middleware
async def guard(request: web.Request, handler):
    # DNS rebinding: a hostile page can resolve its own name to 127.0.0.1 and then make
    # same-origin requests that carry no Origin header. Its Host header still names it.
    host = (
        request.host.rsplit(":", 1)[0]
        if not request.host.startswith("[")
        else request.host.split("]")[0] + "]"
    )
    if host not in LOCAL_HOSTS:
        return web.Response(status=403, text="forbidden host\n")
    if request.path == "/":
        return await handler(request)
    origin = request.headers.get("Origin")
    if origin is not None and origin not in request.app[ORIGINS]:
        log.warning("rejected %s %s from origin %s", request.method, request.path, origin)
        return web.Response(status=403, text="forbidden origin\n")
    if request.method == "OPTIONS":
        headers = _cors_headers(origin)
        if request.headers.get("Access-Control-Request-Private-Network") == "true":
            headers["Access-Control-Allow-Private-Network"] = "true"
        return web.Response(status=204, headers=headers)
    try:
        resp = await handler(request)
    except web.HTTPException as e:
        resp = e
    resp.headers.update(_cors_headers(origin))
    return resp


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _flag(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes"}


async def health(request: web.Request) -> web.Response:
    analyzer = request.app[ANALYZER]
    return web.json_response(
        {"ok": True, "schemaVersion": SCHEMA_VERSION, "calibration": analyzer.calibration_name}
    )


async def analyze(request: web.Request) -> web.Response:
    q = request.query
    body: dict = {}
    data: bytes | None = None
    if request.method == "POST":
        if request.content_type == "application/json":
            try:
                body = await request.json()
            except ValueError:
                return _error(400, "invalid JSON body")
            if not isinstance(body, dict):
                return _error(400, "JSON body must be an object")
        else:
            data = await request.read()
            if not data:
                return _error(400, "empty body: send audio bytes or JSON {'path': ...}")

    def opt(name: str):
        return body.get(name, q.get(name))

    try:
        options = AnalyzeOptions(windows=_flag(str(opt("windows") or "")), nearest=int(opt("nearest") or 0))
    except ValueError:
        return _error(400, "nearest must be an integer")
    use_cache = not _flag(str(opt("refresh") or ""))
    analyzer: SongAnalyzer = request.app[ANALYZER]

    if data is None:
        raw = opt("path")
        if not raw:
            return _error(400, "missing 'path' (or POST the audio bytes)")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            return _error(400, "path must be absolute")
        if path.suffix.lower() not in AUDIO_EXTS:
            return _error(400, f"not an audio file type: {path.suffix or '(none)'}")
        if not path.is_file():
            return _error(404, "file not found")
        work = (analyzer.analyze_file, path, options, use_cache)
    else:
        work = (analyzer.analyze_bytes, data, q.get("name"), options, use_cache)

    # One analysis at a time: they all share the GPU, and a repeat request for the same song
    # then finds the first one's result in the cache.
    async with request.app[ANALYZE_LOCK]:
        try:
            result = await asyncio.to_thread(*work)
        except subprocess.CalledProcessError:
            return _error(422, "could not decode the audio")
        except ValueError as e:
            return _error(422, str(e))
    return web.Response(text=result.to_json(), content_type="application/json")


def make_app(embedder, analyzer) -> web.Application:
    app = web.Application(middlewares=[guard], client_max_size=MAX_UPLOAD_BYTES)
    app[EMBEDDER] = embedder
    app[ANALYZER] = analyzer
    app[EMBED_LOCK] = asyncio.Lock()
    app[ANALYZE_LOCK] = asyncio.Lock()
    app[ORIGINS] = analyze_origins()
    app.router.add_get("/", websocket)
    app.router.add_get("/health", health)
    app.router.add_route("OPTIONS", "/health", health)
    for method in ("GET", "POST", "OPTIONS"):
        app.router.add_route(method, "/analyze", analyze)
    return app


async def serve_forever(host: str, port: int) -> None:
    from .embedder import Embedder

    embedder = Embedder()
    analyzer = SongAnalyzer(embedder=embedder)
    log.info("model %s ready on %s", embedder.model_id, embedder.device)
    runner = web.AppRunner(make_app(embedder, analyzer), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("listening on ws://%s:%d (extension) and http://%s:%d/analyze", host, port, host, port)
    try:
        await asyncio.Future()
    finally:
        await runner.cleanup()
