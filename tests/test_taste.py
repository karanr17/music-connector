import numpy as np
import pytest

from music_connector import taste
from music_connector.config import EventWeights, RecommendConfig

from .conftest import FakeConn, unit

W = EventWeights()


def test_weights_sum_kinds_and_decay_with_age():
    conn = FakeConn(
        events=[
            (1, "like", 0.0),
            (1, "complete", 0.0),
            (2, "skip", 90.0),
            (3, "like", 180.0),
            (3, "dislike", 0.0),
        ]
    )
    got = taste.song_weights(conn, half_life_days=90)
    assert got[1] == pytest.approx(W.like + W.complete)
    assert got[2] == pytest.approx(W.skip / 2)
    assert got[3] == pytest.approx(W.like / 4 + W.dislike)


def test_future_timestamps_are_not_amplified():
    conn = FakeConn(events=[(1, "like", -5.0)])
    assert taste.song_weights(conn, 90)[1] == pytest.approx(W.like)


def test_event_weights_lookup_matches_schema_kinds():
    for kind in ("like", "dislike", "skip", "complete", "replay", "play"):
        assert isinstance(W.get(kind), float)
    assert W.get("like") > W.get("replay") > W.get("complete") > W.get("play") > 0
    assert W.get("dislike") < W.get("skip") < 0


def test_build_splits_distinct_tastes_and_picks_weighted_medoid():
    songs = {
        1: ("rock a", unit(1, 0.05, 0)),
        2: ("rock b", unit(1, -0.05, 0)),
        3: ("rock c", unit(1, 0, 0.02)),
        4: ("jazz", unit(0, 0, 1)),
        5: ("hated", unit(0, 1, 0)),
    }
    events = [
        (1, "complete", 0),
        (2, "complete", 0),
        (3, "like", 0),
        (3, "like", 0),
        (4, "like", 0),
        (5, "dislike", 0),
    ]
    clusters, dis_vecs = taste.build(FakeConn(events, songs), RecommendConfig())

    assert [sorted(k.member_ids) for k in clusters] == [[1, 2, 3], [4]]
    rock, jazz = clusters
    assert rock.importance == pytest.approx(2 * W.complete + 2 * W.like)
    assert rock.medoid_id == 3
    assert jazz.medoid_title == "jazz"
    np.testing.assert_allclose(dis_vecs, [songs[5][1]])


def test_build_with_no_positive_evidence():
    clusters, dis = taste.build(FakeConn([(1, "skip", 0)], {1: ("x", unit(1, 0))}))
    assert clusters == []
    assert len(dis) == 1
