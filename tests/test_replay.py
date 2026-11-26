import numpy as np
import pytest

from music_connector import replay
from music_connector.config import RecommendConfig
from music_connector.preference import Projection
from music_connector.replay import Event, Fitted
from music_connector.synthetic import make_world, simulate_log


def test_rank_counts_ties_half():
    assert replay.rank_of_target(0.9, np.array([0.1, 0.2])) == 1.0
    assert replay.rank_of_target(0.1, np.array([0.5, 0.2])) == 3.0
    assert replay.rank_of_target(0.0, np.zeros(4)) == 3.0


def test_hit_rate_and_ndcg_on_fixed_ranks():
    ranks = [1, 3, 11, 2]
    assert replay.hit_rate(ranks, 10) == 0.75
    assert replay.ndcg(ranks, 10) == pytest.approx((1 + 1 / np.log2(4) + 1 / np.log2(3)) / 4)
    assert np.isnan(replay.hit_rate([], 10)) and np.isnan(replay.ndcg([], 10))


def test_auc():
    assert replay.auc([(0.9, 1), (0.8, 1), (0.1, 0)]) == 1.0
    assert replay.auc([(0.1, 1), (0.9, 0)]) == 0.0
    assert replay.auc([(0.5, 1), (0.5, 0)]) == 0.5
    assert np.isnan(replay.auc([(0.5, 1)]))


def test_calibration_metrics():
    perfect = [(0.0, 0), (1.0, 1)] * 5
    assert replay.brier(perfect) == 0.0 and replay.ece(perfect) == 0.0
    # Always predicting 0.8 for a 50/50 outcome: Brier (0.64 + 0.04) / 2, ECE |0.8 - 0.5|.
    over = [(0.8, 1), (0.8, 0)] * 10
    assert replay.brier(over) == pytest.approx(0.34)
    assert replay.ece(over) == pytest.approx(0.3)
    assert replay.reliability(over) == [(pytest.approx(0.8), 0.5, 20)]


def _fixture():
    """Songs 0-9 are liked, the rest not; one event a day, alternating a like and a skip."""
    X = np.eye(40)
    events = [
        Event(float(d), d % 10 if d % 2 == 0 else 10 + d % 30, "like" if d % 2 == 0 else "skip")
        for d in range(60)
    ]
    return X, events


def _oracle_method(X, history, now, rng) -> Fitted:
    return Fitted(lambda idx: (np.asarray(idx) < 10).astype(float), lambda idx: (np.asarray(idx) < 10) * 1.0)


def test_replay_scores_an_oracle_perfectly_and_random_near_chance():
    X, events = _fixture()
    res = replay.replay(
        X,
        events,
        {"oracle": _oracle_method, "random": replay.random_method},
        k=5,
        n_neg=20,
        warmup=20,
        refit_every=5,
        oracle=lambda day: (np.arange(40) < 10) * 1.0,
    )
    s = replay.summarise(res, k=5)
    assert s["oracle"]["hit_rate@5"] == 1.0 and s["oracle"]["auc"] == 1.0
    assert s["oracle"]["brier"] == 0.0
    assert s["oracle"]["n_positive_targets"] == 20 and s["oracle"]["n_labelled_events"] == 40
    assert s["random"]["auc"] < 0.8
    # Only the even songs of 0-9 are ever played, so the five odd ones are liked and unheard.
    assert s["oracle"]["oracle_like_rate@5"] == 1.0


def test_replay_is_deterministic_with_a_seed():
    X, events = _fixture()
    run = lambda: replay.summarise(  # noqa: E731
        replay.replay(X, events, {"random": replay.random_method}, k=5, n_neg=10, warmup=20, seed=3)
    )
    assert run() == run()


def test_methods_only_see_the_past():
    X, events = _fixture()
    seen = []

    def spy(X, history, now, rng):
        seen.append((len(history), now))
        return Fitted(lambda idx: np.zeros(len(idx)))

    replay.replay(X, events, {"spy": spy}, warmup=20, refit_every=10)
    assert seen == [(20, 19.0), (30, 29.0), (40, 39.0), (50, 49.0)]


def test_real_methods_run_on_a_small_synthetic_world():
    world = make_world(n_songs=300, seed=0, dim=32)
    events = simulate_log(world, n_events=200, seed=1)
    assert events == simulate_log(world, n_events=200, seed=1)
    cfg = RecommendConfig(pca_dim=8)
    methods = {
        "most-played": replay.most_played_method(cfg),
        "v1": replay.v1_method(cfg),
        "v2": replay.v2_method(Projection.fit(world.X, cfg.pca_dim), cfg, sample=False),
        "ts": replay.v2_method(Projection.fit(world.X, cfg.pca_dim), cfg),
    }
    s = replay.summarise(replay.replay(world.X, events, methods, refit_every=50, oracle=world.like_prob))
    for row in s.values():
        assert 0.0 <= row["hit_rate@10"] <= 1.0 and 0.0 <= row["auc"] <= 1.0
    assert "brier" in s["v2"] and "brier" not in s["ts"]
