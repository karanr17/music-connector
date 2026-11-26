"""A synthetic library and listener for the offline replay, since one person's real log is small.

Why synthetic rather than a public dataset: no public listening dataset ships MuQ-MuLan
embeddings, and computing them needs the audio plus the CC BY-NC model. Instead the catalogue is
built to match the geometry of a real MuQ-MuLan library (unit vectors, a strong shared component,
mean pairwise cosine ~0.42 with sd ~0.19, which is what the owner's library measures in aggregate),
and the listener is built so the questions the models differ on are actually present:

- several liked genres of different strength, and disliked ones;
- a preference *within* each liked genre along a hidden direction, which a linear model can learn
  and cosine-to-medoid cannot;
- a taste shift halfway through, which tests the recency weighting;
- a logging policy like real use (mostly neighbours of liked songs, some exploration, some
  replays), so the log has the same exposure bias as a real one.
"""

from dataclasses import dataclass

import numpy as np

from .replay import Event

DIM = 512


@dataclass
class World:
    X: np.ndarray
    genre: np.ndarray
    # (days at which each phase starts, true like-probability per song in that phase)
    phases: list[tuple[float, np.ndarray]]

    def like_prob(self, day: float) -> np.ndarray:
        current = self.phases[0][1]
        for start, p in self.phases:
            if day >= start:
                current = p
        return current


def _unit(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def make_world(
    n_songs: int = 3000,
    n_genres: int = 6,
    n_sub: int = 4,
    n_styles: int = 6,
    days: float = 180.0,
    seed: int = 0,
    dim: int = DIM,
) -> World:
    """`genre` holds the sub-genre (n_genres * n_sub of them); taste is defined per sub-genre."""
    rng = np.random.default_rng(seed)
    shared = _unit(rng.standard_normal(dim))
    centres = _unit(rng.standard_normal((n_genres, dim)))
    subs = _unit(rng.standard_normal((n_genres * n_sub, dim)))
    genre = rng.integers(n_genres * n_sub, size=n_songs)
    noise = _unit(rng.standard_normal((n_songs, dim)))
    # Continuous style factors shared across genres (think energy, tempo, acoustic vs electronic).
    styles = _unit(rng.standard_normal((n_styles, dim)))
    z = rng.standard_normal((n_songs, n_styles))
    # Weights grid-searched so pairwise cosine is ~0.34 +- 0.19 with 5th/95th percentiles of
    # ~0.13/0.77, close to the aggregate measured on a real library (0.42 +- 0.19, 0.12/0.76).
    X = _unit(
        0.6 * shared + 0.8 * centres[genre // n_sub] + 0.3 * subs[genre] + 0.3 * noise + 0.2 * z @ styles
    )
    n_genres = n_genres * n_sub

    # Within each sub-genre the listener prefers one end of one style axis.
    axis = rng.integers(n_styles, size=n_genres)
    sign = rng.choice([-1.0, 1.0], size=n_genres)
    proj = sign[genre] * z[np.arange(n_songs), axis[genre]]

    def probs(affinity: np.ndarray) -> np.ndarray:
        liked = affinity[genre] > 0
        logit = affinity[genre] + np.where(liked, 1.5 * proj, 0.3 * proj)
        return 1.0 / (1.0 + np.exp(-logit))

    affinity = np.full(n_genres, -2.0)
    affinity[:3] = [1.5, 1.0, 0.5]
    affinity[3:8] = -0.5
    shifted = affinity.copy()
    shifted[0], shifted[8] = -1.0, 1.5
    return World(X, genre, [(0.0, probs(affinity)), (days / 2, probs(shifted))])


def _outcome(liked: bool, rng: np.random.Generator) -> str:
    if liked:
        return str(rng.choice(["like", "complete", "replay", "play"], p=[0.2, 0.6, 0.1, 0.1]))
    return str(rng.choice(["skip", "dislike", "play"], p=[0.7, 0.1, 0.2]))


def simulate_log(world: World, n_events: int = 2000, days: float = 180.0, seed: int = 1) -> list[Event]:
    rng = np.random.default_rng(seed)
    X, n = world.X, len(world.X)
    times = np.sort(rng.uniform(0, days, n_events))
    events: list[Event] = []
    positives: list[int] = []
    for t in times:
        u = rng.random()
        if positives and u < 0.5:
            anchor = X[positives[rng.integers(len(positives))]]
            song = int(rng.choice(np.argsort(-(X @ anchor))[1:51]))
        elif positives and u < 0.65:
            song = positives[rng.integers(len(positives))]
        else:
            song = int(rng.integers(n))
        liked = rng.random() < world.like_prob(t)[song]
        kind = _outcome(liked, rng)
        events.append(Event(float(t), song, kind))
        if kind in ("like", "complete", "replay"):
            positives.append(song)
    return events
