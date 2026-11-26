"""v2 preference model: Thompson sampling over Bayesian linear regression, one posterior per taste cluster.

Songs are projected from the 512-d MuQ-MuLan space to a few PCA coordinates fit on the whole
library (unsupervised, so it sees no feedback). Each feedback event becomes a target in [-1, 1]
and updates a conjugate Gaussian posterior over linear weights. A global posterior is fit on all
events; each taste cluster's posterior starts from the global mean and adds only the events whose
song is nearest that cluster's medoid, so a cluster with few events borrows strength from the rest.
"""

from dataclasses import dataclass

import numpy as np

from .config import RecommendConfig

# kind -> (reward in [0, 1], evidence weight). The events table stores no per-event listen
# fraction, but the live capture's kinds are fraction bands: skip < 30 s, play 30 s to 80 %,
# complete >= 80 %. A play is therefore a half reward rather than a positive.
EVENT_REWARD: dict[str, tuple[float, float]] = {
    "like": (1.0, 1.0),
    "dislike": (0.0, 1.0),
    "replay": (1.0, 0.6),
    "complete": (1.0, 0.5),
    "play": (0.5, 0.3),
    "skip": (0.0, 0.7),
}


def reward(kind: str) -> float:
    return EVENT_REWARD[kind][0]


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    # Rows are principal axes pre-divided by their std, so projected coordinates have unit
    # variance and a N(0, I) prior on the weights is on a sensible scale.
    basis: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, dim: int) -> "Projection":
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        dim = max(1, min(dim, len(X) - 1, vt.shape[0]))
        std = s[:dim] / np.sqrt(max(len(X) - 1, 1))
        return cls(mean, vt[:dim] / np.maximum(std, 1e-8)[:, None])

    @property
    def dim(self) -> int:
        return self.basis.shape[0] + 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Projected features with a trailing bias column."""
        Z = (np.atleast_2d(X) - self.mean) @ self.basis.T
        return np.hstack([Z, np.ones((len(Z), 1))])


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray
    precision: np.ndarray

    @classmethod
    def prior(cls, dim: int, precision: float, mean: np.ndarray | None = None) -> "Posterior":
        return cls(np.zeros(dim) if mean is None else mean.copy(), precision * np.eye(dim))

    def update(self, Phi: np.ndarray, y: np.ndarray, w: np.ndarray, noise_var: float) -> "Posterior":
        if not len(Phi):
            return self
        precision = self.precision + (Phi * (w / noise_var)[:, None]).T @ Phi
        rhs = self.precision @ self.mean + Phi.T @ (w * y / noise_var)
        return Posterior(np.linalg.solve(precision, rhs), precision)

    def sample(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        # theta = mean + L^-T z has covariance (L L^T)^-1 = precision^-1, without inverting.
        L = np.linalg.cholesky(self.precision)
        z = rng.standard_normal(len(self.mean))
        return self.mean + scale * np.linalg.solve(L.T, z)


@dataclass
class PreferenceModel:
    projection: Projection
    medoids: np.ndarray
    global_post: Posterior
    cluster_posts: list[Posterior]

    def nearest_cluster(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) @ self.medoids.T).argmax(axis=1)

    def weights(self, rng: np.random.Generator | None, exploration: float = 1.0) -> np.ndarray:
        """One weight vector per cluster: a posterior draw, or the posterior mean when rng is None."""
        if rng is None:
            return np.stack([p.mean for p in self.cluster_posts])
        return np.stack([p.sample(rng, exploration) for p in self.cluster_posts])

    def predict(self, X: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expected reward in [0, 1] (unclipped) under per-cluster weights `theta`, and the cluster used."""
        k = self.nearest_cluster(X)
        lin = np.einsum("ij,ij->i", self.projection(X), theta[k])
        return 0.5 + 0.5 * lin, k


def fit(
    library: np.ndarray,
    medoids: np.ndarray,
    event_vecs: np.ndarray,
    kinds: list[str],
    ages_days: np.ndarray,
    cfg: RecommendConfig = RecommendConfig(),
    projection: Projection | None = None,
) -> PreferenceModel:
    """`library` fits the projection (pass `projection` to reuse one); events are recency-weighted
    with the same half-life as v1's taste weights."""
    proj = projection or Projection.fit(library, cfg.pca_dim)
    medoids = np.atleast_2d(medoids)
    Phi = proj(event_vecs) if len(event_vecs) else np.empty((0, proj.dim))
    y = np.array([2 * EVENT_REWARD[k][0] - 1 for k in kinds], dtype=float)
    decay = 0.5 ** (np.maximum(np.asarray(ages_days, dtype=float), 0.0) / cfg.recency_half_life_days)
    w = np.array([EVENT_REWARD[k][1] for k in kinds], dtype=float) * decay

    glob = Posterior.prior(proj.dim, cfg.ts_prior_precision).update(Phi, y, w, cfg.ts_noise_var)
    owner = (event_vecs @ medoids.T).argmax(axis=1) if len(event_vecs) else np.empty(0, dtype=int)
    posts = []
    for k in range(len(medoids)):
        m = owner == k
        prior = Posterior.prior(proj.dim, cfg.ts_cluster_pooling, mean=glob.mean)
        posts.append(prior.update(Phi[m], y[m], w[m], cfg.ts_noise_var))
    return PreferenceModel(proj, medoids, glob, posts)
