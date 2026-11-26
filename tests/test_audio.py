import numpy as np

from music_connector.audio import MIN_TAIL_S, find_audio_files, windows
from music_connector.embedder import pool

SR = 1000
WIN_S = 10.0
SIZE = int(SR * WIN_S)


def test_exact_multiple_gives_back_to_back_windows():
    wav = np.arange(3 * SIZE, dtype=np.float32)
    wins, starts = windows(wav, SR, WIN_S)
    assert wins.shape == (3, SIZE)
    assert starts == [0.0, 10.0, 20.0]
    assert wins[1][0] == SIZE


def test_long_tail_becomes_end_aligned_window():
    tail = int(MIN_TAIL_S * SR)
    wav = np.arange(2 * SIZE + tail, dtype=np.float32)
    wins, starts = windows(wav, SR, WIN_S)
    assert len(wins) == 3
    assert starts[-1] == (len(wav) - SIZE) / SR
    assert wins[-1][-1] == wav[-1]


def test_short_tail_is_dropped():
    wav = np.zeros(2 * SIZE + int(MIN_TAIL_S * SR) - 1, dtype=np.float32)
    wins, starts = windows(wav, SR, WIN_S)
    assert len(wins) == 2
    assert starts == [0.0, 10.0]


def test_clip_shorter_than_a_window_is_looped_not_padded():
    wav = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    wins, starts = windows(wav, SR, WIN_S)
    assert wins.shape == (1, SIZE)
    assert starts == [0.0]
    assert not np.any(wins == 0)
    assert list(wins[0][:6]) == [1, 2, 3, 1, 2, 3]


def test_pool_identical_windows_has_zero_spread():
    v = np.tile(np.array([0.6, 0.8], dtype=np.float32), (4, 1))
    song, spread = pool(v)
    np.testing.assert_allclose(song, [0.6, 0.8], atol=1e-6)
    assert abs(spread) < 1e-6


def test_pool_is_unit_length_and_spread_grows_with_disagreement():
    tight = np.array([[1.0, 0.0], [0.99, 0.141]], dtype=np.float32)
    loose = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    song, s_tight = pool(tight)
    _, s_loose = pool(loose)
    assert abs(np.linalg.norm(song) - 1.0) < 1e-6
    assert song.dtype == np.float32
    assert s_loose > s_tight > 0


def test_find_audio_files(tmp_path):
    (tmp_path / "a").mkdir()
    for name in ("a/x.MP3", "a/y.flac", "a/notes.txt", "z.wav"):
        (tmp_path / name).touch()
    found = find_audio_files([tmp_path / "a", tmp_path / "z.wav", tmp_path / "a/notes.txt"])
    assert [p.name for p in found] == ["x.MP3", "y.flac", "z.wav"]
