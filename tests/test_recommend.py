from collections import Counter

import numpy as np
import pytest

from music_connector.config import RecommendConfig
from music_connector.recommend import (
    Candidate,
    Pick,
    _allocate,
    _drop_duplicates,
    _mmr_fill,
    _score,
    _smooth_order,
)
from music_connector.taste import TasteCluster

from .conftest import unit

CFG = RecommendConfig()


def cand(i, vec, score=0.0, artist=None):
    return Candidate(i, f"song {i}", artist, None, unit(*vec), score)


@pytest.mark.parametrize("total", [0, 1, 7, 25, 100])
def test_allocate_always_sums_to_total(total):
    assert sum(_allocate(total, CFG.slots).values()) == total


def test_allocate_largest_remainder():
    # 10 * (0.55, 0.20, 0.15, 0.10) = 5.5, 2, 1.5, 1 -> two .5 remainders, one leftover slot.
    alloc = _allocate(10, CFG.slots)
    assert alloc["indie"] == 2 and alloc["wildcard"] == 1
    assert alloc["taste"] + alloc["classic"] == 7
    assert alloc["taste"] in (5, 6) and alloc["classic"] in (1, 2)


def test_allocate_normalises_shares():
    assert _allocate(4, {"a": 3.0, "b": 1.0}) == {"a": 3, "b": 1}
    assert _allocate(5, {"a": 30, "b": 20}) == {"a": 3, "b": 2}


def test_mmr_prefers_diverse_pick_over_near_duplicate_of_a_pick():
    a = cand(1, (1, 0, 0), score=0.9)
    near_a = cand(2, (1, 0.25, 0), score=0.89)
    other = cand(3, (0, 1, 0), score=0.8)
    picks: list[Pick] = []
    unfilled = _mmr_fill([a, near_a, other], 2, "taste", picks, Counter(), CFG)
    assert unfilled == 0
    assert [p.candidate.id for p in picks] == [1, 3]


def test_mmr_respects_max_per_artist_and_reports_shortfall():
    group = [cand(i, (1, i, 0), score=1.0, artist="same") for i in range(1, 5)]
    picks: list[Pick] = []
    artists: Counter = Counter()
    unfilled = _mmr_fill(group, 4, "taste", picks, artists, CFG)
    assert len(picks) == CFG.max_per_artist
    assert unfilled == 4 - CFG.max_per_artist
    assert artists["same"] == CFG.max_per_artist


def test_mmr_skips_recordings_already_picked():
    first = cand(1, (1, 0, 0), score=1.0)
    reupload = cand(2, (1, 0.01, 0), score=1.0)
    picks = [Pick(first, "taste")]
    unfilled = _mmr_fill([first, reupload], 1, "indie", picks, Counter(), CFG)
    assert unfilled == 1 and len(picks) == 1


def test_drop_duplicates_against_heard_songs():
    heard = np.stack([unit(1, 0, 0)])
    group = [cand(1, (1, 0.001, 0)), cand(2, (0, 1, 0))]
    assert [c.id for c in _drop_duplicates(group, heard, CFG)] == [2]
    assert _drop_duplicates(group, np.empty((0, 0)), CFG) == group


def test_score_uses_best_medoid_and_dislike_penalty():
    clusters = [TasteCluster(10, "rock", unit(1, 0, 0), 2.0, [10]),
                TasteCluster(20, "jazz", unit(0, 1, 0), 1.0, [20])]
    a, b = cand(1, (0, 1, 0)), cand(2, (1, 0, 1))
    _score([a, b], clusters, np.empty((0, 3), dtype=np.float32), CFG)
    assert a.near == "jazz" and a.score == pytest.approx(1.0)
    assert b.near == "rock" and b.score == pytest.approx(np.sqrt(0.5))

    _score([a], clusters, np.stack([unit(0, 1, 0)]), CFG)
    assert a.score == pytest.approx(1.0 - CFG.dislike_penalty)


def test_smooth_order_starts_strong_then_chains_neighbours():
    picks = [Pick(cand(1, (1, 0, 0), score=0.5), "taste"),
             Pick(cand(2, (0, 1, 0), score=0.9), "taste"),
             Pick(cand(3, (0.2, 1, 0), score=0.1), "taste"),
             Pick(cand(4, (1, 0.2, 0), score=0.2), "taste")]
    assert [p.candidate.id for p in _smooth_order(picks)] == [2, 3, 4, 1]
