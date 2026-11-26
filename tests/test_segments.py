import numpy as np

from music_connector.analysis.segments import (
    PRIME,
    SegmentConfig,
    contrast_novelty,
    group_letter,
    label,
    pick_boundaries,
    segment,
)

WIN_S, HOP_S, BEAT_S = 10.0, 5.0, 0.5


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def synthetic_song(plan: list[tuple[str, float]], seed: int = 0, noise: float = 0.05):
    """Features for a song made of sections, e.g. [("A", 20), ("B", 20), ("A", 20)]: each
    letter gets its own embedding direction and its own chroma/timbre profile."""
    rng = np.random.default_rng(seed)
    letters = sorted({name for name, _ in plan})
    emb_dir = {k: _unit(rng.normal(size=64)) for k in letters}
    feat_dir = {k: _unit(rng.normal(size=25)) for k in letters}
    starts = np.cumsum([0.0] + [d for _, d in plan])
    duration = float(starts[-1])

    def section_at(t):
        return plan[min(np.searchsorted(starts, t, side="right") - 1, len(plan) - 1)][0]

    beats = np.arange(0.0, duration, BEAT_S)
    beat_features = np.array([feat_dir[section_at(t + BEAT_S / 2)] for t in beats[:-1]])
    beat_features += noise * rng.normal(size=beat_features.shape)
    win_starts = np.arange(0.0, duration - WIN_S + 1e-9, HOP_S)
    # A window's embedding is the mix of the sections it covers.
    vecs = []
    for s in win_starts:
        mix = sum(emb_dir[section_at(t)] for t in np.arange(s, s + WIN_S, 1.0))
        vecs.append(_unit(mix + noise * rng.normal(size=64)))
    return {
        "duration": duration,
        "beats": beats,
        "downbeats": beats[::4],
        "beat_features": beat_features,
        "window_starts": win_starts,
        "window_vecs": np.array(vecs),
        "window_s": WIN_S,
    }, starts[1:-1]


def test_finds_section_changes_and_repeats():
    plan = [("A", 30), ("B", 30), ("A", 30), ("C", 30)]
    feats, truth = synthetic_song(plan)
    segs = segment(**feats)
    bounds = [s.start for s in segs[1:]]
    assert len(bounds) == len(truth)
    assert np.allclose(bounds, truth, atol=2.0)
    assert [s.label for s in segs] == ["A", "B", "A", "C"]
    assert segs[2].similarity > 0.9
    assert segs[0].similarity is None


def test_boundaries_land_on_downbeats():
    feats, _ = synthetic_song([("A", 31), ("B", 29), ("C", 30)])
    segs = segment(**feats)
    downbeats = set(np.round(feats["downbeats"], 6))
    assert all(round(s.start, 6) in downbeats for s in segs[1:])


def test_a_uniform_song_is_one_segment():
    feats, _ = synthetic_song([("A", 120)])
    segs = segment(**feats)
    assert len(segs) == 1
    assert segs[0].start == 0.0 and segs[0].end == 120.0


def test_segments_tile_the_song_and_carry_energy():
    feats, _ = synthetic_song([("A", 25), ("B", 25), ("A", 25)])
    rms = np.r_[np.full(250, 0.2), np.full(250, 0.8), np.full(250, 0.2)]
    segs = segment(**feats, rms=rms, rms_rate=10.0)
    assert segs[0].start == 0.0 and segs[-1].end == feats["duration"]
    assert all(a.end == b.start for a, b in zip(segs[:-1], segs[1:], strict=True))
    assert segs[1].energy > segs[0].energy


def test_novelty_peaks_at_the_change():
    vecs = np.r_[np.tile([1.0, 0.0], (10, 1)), np.tile([0.0, 1.0], (10, 1))]
    starts = np.arange(20.0)
    times = np.arange(2.0, 19.0)
    nov = contrast_novelty(vecs, starts, starts + 1, times, span=4.0)
    assert times[np.argmax(nov)] == 10.0


def test_pick_boundaries_respects_min_gap_and_edges():
    times = np.arange(0.0, 60.0, 1.0)
    nov = np.zeros_like(times)
    nov[[2, 20, 23, 40]] = [1.0, 0.9, 0.8, 0.7]
    cfg = SegmentConfig(min_segment_s=8.0, edge_s=4.0, max_rate_s=10.0)
    picked = [t for t, _ in pick_boundaries(times, nov, 60.0, cfg)]
    assert picked == [20.0, 40.0]


def test_labels_primes_for_weaker_repeats():
    a = _unit(np.array([1.0, 0.0, 0.0]))
    a2 = _unit(np.array([1.0, 0.25, 0.0]))  # cos 0.97 to a: same group, a variation
    b = _unit(np.array([0.0, 1.0, 0.0]))
    cfg = SegmentConfig(same_cosine=0.9, variant_below=0.99)
    out = label(np.array([a, b, a, a2]), cfg)
    assert [(g, v) for g, v, _ in out] == [(0, 0), (1, 0), (0, 0), (0, 1)]
    assert group_letter(0) + PRIME * out[3][1] == "A′"


def test_group_letters_continue_past_z():
    assert [group_letter(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]
