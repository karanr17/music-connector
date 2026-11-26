import numpy as np
import pytest

from music_connector import preference
from music_connector.config import RecommendConfig
from music_connector.preference import Posterior, Projection

from .conftest import unit


def test_projection_is_centred_unit_variance_with_bias():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)) @ np.diag([5.0, 2.0, 0.5]) @ rng.standard_normal((3, 12)) + 3.0
    proj = Projection.fit(X, dim=3)
    Z = proj(X)
    assert Z.shape == (200, 4) and proj.dim == 4
    np.testing.assert_allclose(Z[:, -1], 1.0)
    np.testing.assert_allclose(Z[:, :3].mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z[:, :3].std(axis=0, ddof=1), 1.0, rtol=1e-6)


def test_projection_dim_is_capped_by_library_size():
    X = np.random.default_rng(1).standard_normal((5, 20))
    assert Projection.fit(X, dim=16).dim == 4 + 1
    assert Projection.fit(X[:1], dim=16).dim == 2


def test_posterior_update_matches_weighted_ridge_closed_form():
    rng = np.random.default_rng(2)
    Phi, y, w = rng.standard_normal((30, 4)), rng.standard_normal(30), rng.uniform(0.2, 1.0, 30)
    post = Posterior.prior(4, precision=2.0).update(Phi, y, w, noise_var=0.5)
    A = 2.0 * np.eye(4) + Phi.T @ (Phi * (w / 0.5)[:, None])
    np.testing.assert_allclose(post.precision, A)
    np.testing.assert_allclose(post.mean, np.linalg.solve(A, Phi.T @ (w * y / 0.5)))


def test_posterior_updates_compose_like_one_batch():
    rng = np.random.default_rng(3)
    Phi, y, w = rng.standard_normal((20, 3)), rng.standard_normal(20), np.ones(20)
    prior = Posterior.prior(3, 1.0)
    once = prior.update(Phi, y, w, 1.0)
    twice = prior.update(Phi[:8], y[:8], w[:8], 1.0).update(Phi[8:], y[8:], w[8:], 1.0)
    np.testing.assert_allclose(once.mean, twice.mean)
    np.testing.assert_allclose(once.precision, twice.precision)
    assert prior.update(Phi[:0], y[:0], w[:0], 1.0) is prior


def test_posterior_recovers_true_weights_with_enough_data():
    rng = np.random.default_rng(4)
    true = np.array([0.5, -0.3, 0.1])
    Phi = rng.standard_normal((5000, 3))
    y = Phi @ true + 0.3 * rng.standard_normal(5000)
    post = Posterior.prior(3, 1.0).update(Phi, y, np.ones(5000), 0.09)
    np.testing.assert_allclose(post.mean, true, atol=0.02)


def test_sampling_is_deterministic_with_a_seed_and_matches_the_covariance():
    post = Posterior(np.array([1.0, -1.0]), np.array([[4.0, 1.0], [1.0, 2.0]]))
    a = post.sample(np.random.default_rng(7))
    b = post.sample(np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, post.sample(np.random.default_rng(8)))

    rng = np.random.default_rng(0)
    draws = np.stack([post.sample(rng) for _ in range(20000)])
    np.testing.assert_allclose(draws.mean(axis=0), post.mean, atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), np.linalg.inv(post.precision), atol=0.02)
    np.testing.assert_allclose(post.sample(rng, scale=0.0), post.mean)


def _two_taste_library():
    """Two taste regions; within region A, songs leaning towards axis 2 are liked, the rest skipped."""
    rng = np.random.default_rng(5)
    a = np.stack([unit(1, s, 0.1 * rng.standard_normal()) for s in rng.uniform(-0.6, 0.6, 60)])
    b = np.stack([unit(0.1 * rng.standard_normal(), 0, 1) for _ in range(20)])
    return np.vstack([a, b]), np.stack([unit(1, 0, 0), unit(0, 0, 1)])


def test_fit_learns_preference_within_a_cluster_that_cosine_cannot_see():
    X, medoids = _two_taste_library()
    A = X[:60]
    kinds = ["like" if v[1] > 0 else "skip" for v in A]
    model = preference.fit(X, medoids, A, kinds, np.zeros(60), RecommendConfig(pca_dim=3))
    pred, k = model.predict(np.stack([unit(1, 0.5, 0), unit(1, -0.5, 0)]), model.weights(None))
    assert list(k) == [0, 0]
    assert pred[0] > 0.7 and pred[1] < 0.3


def test_cluster_without_events_falls_back_to_the_global_mean():
    X, medoids = _two_taste_library()
    model = preference.fit(X, medoids, X[:10], ["like"] * 10, np.zeros(10), RecommendConfig(pca_dim=3))
    np.testing.assert_allclose(model.cluster_posts[1].mean, model.global_post.mean)
    assert not np.allclose(model.cluster_posts[0].mean, model.global_post.mean)


def test_old_events_count_less():
    X, medoids = _two_taste_library()
    cfg = RecommendConfig(pca_dim=3)
    fresh = preference.fit(X, medoids, X[:5], ["like"] * 5, np.zeros(5), cfg)
    old = preference.fit(X, medoids, X[:5], ["like"] * 5, np.full(5, 3 * cfg.recency_half_life_days), cfg)
    assert np.trace(old.global_post.precision) < np.trace(fresh.global_post.precision)


def test_rewards_follow_listen_fraction_bands():
    r = preference.reward
    assert r("like") == r("complete") == r("replay") == 1.0
    assert r("dislike") == r("skip") == 0.0
    assert r("play") == 0.5


def test_fit_with_no_events_is_the_prior():
    X, medoids = _two_taste_library()
    model = preference.fit(X, medoids, np.empty((0, 3)), [], np.empty(0), RecommendConfig(pca_dim=3))
    pred, _ = model.predict(X[:3], model.weights(None))
    np.testing.assert_allclose(pred, 0.5)


@pytest.mark.parametrize("seed", [0, 1])
def test_weights_draw_is_reproducible(seed):
    X, medoids = _two_taste_library()
    model = preference.fit(X, medoids, X[:10], ["like"] * 10, np.zeros(10), RecommendConfig(pca_dim=3))
    a = model.weights(np.random.default_rng(seed))
    np.testing.assert_array_equal(a, model.weights(np.random.default_rng(seed)))
    assert a.shape == (2, model.projection.dim)
