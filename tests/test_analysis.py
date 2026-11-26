import json
import shutil
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import soundfile as sf

from music_connector.analysis import AnalyzeOptions, SongAnalysis, SongAnalyzer, json_schema
from music_connector.analysis import descriptors as desc
from music_connector.audio import hop_windows

from .conftest import click_track

ROOT = Path(__file__).parents[1]
SCHEMA_FILE = ROOT / "schema" / "song-analysis.v1.schema.json"


def _analyzer(fake_embedder, tmp_path=None, **kw) -> SongAnalyzer:
    return SongAnalyzer(embedder=fake_embedder, calibration=None, cache_dir=tmp_path, **kw)


@pytest.fixture
def analysis(fake_embedder) -> SongAnalysis:
    return _analyzer(fake_embedder).analyze_wav(
        click_track(), sha256="0" * 64, file_name="click.wav", options=AnalyzeOptions(windows=True)
    )


def test_committed_schema_is_up_to_date():
    assert json.loads(SCHEMA_FILE.read_text()) == json_schema(), "run: uv run mc schema -o " + str(
        SCHEMA_FILE.relative_to(ROOT)
    )


def test_output_validates_against_committed_schema(analysis):
    jsonschema.validate(json.loads(analysis.to_json()), json.loads(SCHEMA_FILE.read_text()))


def test_json_round_trip_is_lossless(analysis):
    text = analysis.to_json()
    again = SongAnalysis.model_validate_json(text)
    assert again == analysis
    assert again.to_json() == text


def test_wire_format_is_camel_case_and_versioned(analysis):
    d = json.loads(analysis.to_json())
    assert d["schemaVersion"] == "1.0"
    assert {"durationS", "sampleRate", "sha256"} <= d["source"].keys()
    assert "boundaryStrength" in d["segments"][0]
    assert "nearest" not in d  # optional sections are omitted, not null


def test_readers_ignore_unknown_fields_from_newer_minor_versions(analysis):
    d = json.loads(analysis.to_json())
    d["schemaVersion"] = "1.3"
    d["somethingNew"] = {"x": 1}
    assert SongAnalysis.model_validate(d).schema_version == "1.3"


def test_examples_validate():
    schema = json.loads(SCHEMA_FILE.read_text())
    examples = sorted((ROOT / "examples").glob("*.analysis.json"))
    for p in examples:
        jsonschema.validate(json.loads(p.read_text()), schema)


def test_click_track_tempo_beats_and_energy(analysis):
    assert analysis.tempo.bpm == pytest.approx(120, rel=0.03)
    gaps = np.diff(analysis.beats)
    assert np.median(gaps) == pytest.approx(0.5, abs=0.02)
    assert set(analysis.downbeats) <= set(analysis.beats)
    assert len(analysis.downbeats) == pytest.approx(len(analysis.beats) / 4, abs=1)
    e = analysis.energy
    assert len(e.rms) == len(e.low) == len(e.mid) == len(e.high)
    assert len(e.rms) == pytest.approx(analysis.source.duration_s * e.frame_rate, abs=2)
    assert all(0.0 <= v <= 1.0 for v in e.rms)


def test_segments_tile_the_song_and_descriptors_cover_every_axis(analysis):
    segs = analysis.segments
    assert segs[0].start == 0 and segs[-1].end == pytest.approx(analysis.source.duration_s, abs=1e-3)
    assert all(a.end == b.start for a, b in zip(segs[:-1], segs[1:], strict=True))
    assert set(analysis.descriptors) == set(desc.VOCABULARY)
    assert all(len(v) == 3 for v in analysis.descriptors.values())
    assert analysis.windows is not None
    assert len(analysis.windows.starts) == len(analysis.windows.descriptors)
    assert analysis.analyzer.calibration in {"uncalibrated", "cc-reference-79"}


def test_nearest_is_opt_in_and_failure_tolerant(fake_embedder):
    def broken(vec, k):
        raise ConnectionError("no database")

    a = _analyzer(fake_embedder, nearest_fn=broken)
    wav = click_track(seconds=25)
    assert a.analyze_wav(wav, sha256="1" * 64, file_name=None, options=AnalyzeOptions()).nearest is None
    assert (
        a.analyze_wav(wav, sha256="1" * 64, file_name=None, options=AnalyzeOptions(nearest=3)).nearest == []
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_file_analysis_is_cached_by_content(fake_embedder, tmp_path):
    a = _analyzer(fake_embedder, tmp_path / "cache")
    wav = click_track(seconds=25)
    f1, f2 = tmp_path / "a.wav", tmp_path / "copy-of-a.wav"
    sf.write(f1, wav, 24_000)
    shutil.copy(f1, f2)
    first = a.analyze_file(f1)
    calls = fake_embedder.window_calls
    second = a.analyze_file(f2)
    assert fake_embedder.window_calls == calls
    assert second.source.sha256 == first.source.sha256
    a.analyze_file(f2, use_cache=False)
    assert fake_embedder.window_calls == calls + 1
    a.analyze_file(f1, AnalyzeOptions(windows=True))
    assert fake_embedder.window_calls == calls + 2, "different options are a different cache entry"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_bytes_and_file_share_the_cache(fake_embedder, tmp_path):
    a = _analyzer(fake_embedder, tmp_path / "cache")
    f = tmp_path / "a.wav"
    sf.write(f, click_track(seconds=25), 24_000)
    from_file = a.analyze_file(f)
    calls = fake_embedder.window_calls
    from_bytes = a.analyze_bytes(f.read_bytes(), "a.wav")
    assert fake_embedder.window_calls == calls
    assert from_bytes == from_file


def test_hop_windows_overlap_and_end_aligned_tail():
    sr = 100
    wav = np.arange(int(38 * sr), dtype=np.float32)
    wins, starts = hop_windows(wav, sr, 10.0, 5.0)
    assert starts[:6] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert starts[-1] == 28.0
    assert wins[-1][-1] == wav[-1]


def test_calibrated_z_scores_and_top_k():
    n = len(desc.prompts())
    rng = np.random.default_rng(1)
    song_cos = rng.normal(0.2, 0.05, size=(40, n))
    cal = desc.Calibration.from_cosines("test", song_cos, song_cos)
    assert desc.Calibration.from_dict(cal.to_dict()).vocabulary == desc.vocabulary_id()
    cos = cal.song_mean.copy()
    genre = desc.axis_indices()["genre"]
    cos[genre[5]] += 3 * cal.song_std[genre[5]]
    z = desc.zscores(cos, cal.song_mean, cal.song_std)
    top = desc.top_k(cos, z, 2)
    assert top["genre"][0]["label"] == desc.VOCABULARY["genre"][5]
    assert top["genre"][0]["z"] > top["genre"][1]["z"] + 1.5
    flat = np.full(n, 0.3)
    assert np.allclose(desc.zscores(flat, np.full(n, 0.3), np.full(n, 0.05)), 0.0)


def test_calibration_for_another_vocabulary_is_ignored(tmp_path):
    n = len(desc.prompts())
    cal = desc.Calibration.from_cosines("old", np.ones((3, n)) * 0.1, np.ones((3, n)) * 0.1).to_dict()
    cal["vocabulary"] = "something-else"
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    loaded = desc.load_calibration(p)
    assert loaded is None or loaded.name != "old"
