import pytest

from music_connector import ingest


@pytest.fixture(autouse=True)
def no_model_or_db(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must fail before loading the model or touching the DB")

    monkeypatch.setattr(ingest, "Embedder", boom)
    monkeypatch.setattr(ingest.db, "connect", boom)


def test_missing_path_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.mp3"):
        ingest.ingest_files([tmp_path / "nope.mp3"], [])


def test_folder_without_audio_fails_fast(tmp_path):
    (tmp_path / "readme.txt").touch()
    with pytest.raises(FileNotFoundError, match="no audio files"):
        ingest.ingest_files([tmp_path], [])
