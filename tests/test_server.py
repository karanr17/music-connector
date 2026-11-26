import asyncio
import subprocess

import pytest
from aiohttp.test_utils import TestClient, TestServer

from music_connector.analysis import AnalyzeOptions, SongAnalysis, SongAnalyzer
from music_connector.server import make_app

from .conftest import FakeEmbedder, click_track

PLAYER = "http://localhost:5173"


@pytest.fixture(scope="module")
def sample() -> SongAnalysis:
    a = SongAnalyzer(embedder=FakeEmbedder(), calibration=None, cache_dir=None)
    return a.analyze_wav(
        click_track(seconds=20), sha256="f" * 64, file_name="x.wav", options=AnalyzeOptions()
    )


class StubAnalyzer:
    calibration_name = "stub"

    def __init__(self, result: SongAnalysis, fail: Exception | None = None):
        self.result, self.fail, self.calls = result, fail, []

    def analyze_file(self, path, options, use_cache):
        self.calls.append(("file", str(path), options, use_cache))
        if self.fail:
            raise self.fail
        return self.result

    def analyze_bytes(self, data, name, options, use_cache):
        self.calls.append(("bytes", len(data), name, options, use_cache))
        if self.fail:
            raise self.fail
        return self.result


def run(analyzer, scenario):
    async def go():
        client = TestClient(TestServer(make_app(None, analyzer)))
        await client.start_server()
        try:
            return await scenario(client)
        finally:
            await client.close()

    return asyncio.run(go())


def test_post_bytes_returns_analysis_with_cors(sample):
    stub = StubAnalyzer(sample)

    async def scenario(c):
        r = await c.post(
            "/analyze?name=song.mp3&windows=1",
            data=b"ID3fakeaudio",
            headers={"Origin": PLAYER, "Content-Type": "audio/mpeg"},
        )
        return r.status, r.headers.get("Access-Control-Allow-Origin"), await r.text()

    status, acao, body = run(stub, scenario)
    assert status == 200
    assert acao == PLAYER
    assert SongAnalysis.model_validate_json(body) == sample
    kind, size, name, options, use_cache = stub.calls[0]
    assert (kind, size, name, options.windows, use_cache) == ("bytes", 12, "song.mp3", True, True)


def test_post_json_path(sample, tmp_path):
    f = tmp_path / "song.mp3"
    f.write_bytes(b"x")
    stub = StubAnalyzer(sample)

    async def scenario(c):
        r = await c.post("/analyze", json={"path": str(f), "nearest": 5, "refresh": True})
        return r.status

    assert run(stub, scenario) == 200
    kind, path, options, use_cache = stub.calls[0]
    assert (kind, path, options.nearest, use_cache) == ("file", str(f), 5, False)


def test_get_with_path_query(sample, tmp_path):
    f = tmp_path / "song.flac"
    f.write_bytes(b"x")

    async def scenario(c):
        return (await c.get("/analyze", params={"path": str(f)})).status

    assert run(StubAnalyzer(sample), scenario) == 200


@pytest.mark.parametrize(
    ("path", "status"),
    [("relative/song.mp3", 400), ("/etc/passwd", 400), ("/no/such/song.mp3", 404), (None, 400)],
)
def test_bad_paths_are_rejected_before_analysis(sample, path, status):
    stub = StubAnalyzer(sample)

    async def scenario(c):
        return (await c.post("/analyze", json={} if path is None else {"path": path})).status

    assert run(stub, scenario) == status
    assert stub.calls == []


def test_preflight_from_player_origin():
    async def scenario(c):
        r = await c.options(
            "/analyze",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        return r.status, dict(r.headers)

    status, headers = run(StubAnalyzer(None), scenario)
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:5173"
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert headers["Access-Control-Allow-Private-Network"] == "true"


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "http://localhost:3000", "chrome-extension://abc"]
)
def test_other_origins_are_refused(sample, origin):
    stub = StubAnalyzer(sample)

    async def scenario(c):
        r = await c.post("/analyze", data=b"abc", headers={"Origin": origin})
        return r.status, r.headers.get("Access-Control-Allow-Origin")

    assert run(stub, scenario) == (403, None)
    assert stub.calls == []


def test_extra_origins_from_env(sample, monkeypatch):
    monkeypatch.setenv("MC_ALLOWED_ORIGINS", "http://localhost:4173/, http://example.test")

    async def scenario(c):
        r = await c.post("/analyze", data=b"abc", headers={"Origin": "http://localhost:4173"})
        return r.status

    assert run(StubAnalyzer(sample), scenario) == 200


def test_dns_rebinding_host_is_refused(sample):
    async def scenario(c):
        return (await c.get("/health", headers={"Host": "attacker.example:8765"})).status

    assert run(StubAnalyzer(sample), scenario) == 403


def test_websocket_refuses_web_pages():
    async def scenario(c):
        r = await c.get(
            "/",
            headers={
                "Origin": PLAYER,
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
                "Sec-WebSocket-Version": "13",
            },
        )
        return r.status

    assert run(StubAnalyzer(None), scenario) == 403


def test_websocket_accepts_local_scripts_without_origin():
    async def scenario(c):
        ws = await c.ws_connect("/")
        await ws.send_str('{"type": "hello", "sampleRate": 48000}')
        await ws.send_str('{"type": "seek", "position": 3}')
        await ws.close()
        return ws.closed

    assert run(StubAnalyzer(None), scenario)


def test_decode_errors_are_422(sample):
    stub = StubAnalyzer(sample, fail=subprocess.CalledProcessError(1, "ffmpeg"))

    async def scenario(c):
        r = await c.post("/analyze", data=b"not audio")
        return r.status, await r.json()

    status, body = run(stub, scenario)
    assert status == 422
    assert "decode" in body["error"]


def test_health():
    async def scenario(c):
        r = await c.get("/health", headers={"Origin": PLAYER})
        return r.status, await r.json(), r.headers.get("Access-Control-Allow-Origin")

    status, body, acao = run(StubAnalyzer(None), scenario)
    assert status == 200
    assert body == {"ok": True, "schemaVersion": "1.0", "calibration": "stub"}
    assert acao == PLAYER
