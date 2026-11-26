"""Offline time-ordered replay of the events log: no database, no model, just arrays.

Every `refit_every` events, each method is fit on the events strictly before that point and then
scores the next events it has not seen:

- Ranking (positive next events only): the target song against `n_neg` songs never heard before,
  giving hit-rate@k and NDCG@k. Sampled-negative metrics flatter every method equally; they are
  for comparing methods, not absolute quality.
- Discrimination: AUC of the score for positive vs negative next events.
- Calibration (methods that output probabilities): Brier score, expected calibration error,
  and a reliability curve.
- Oracle (synthetic data only, where the true like-probability is known): the mean true
  like-probability of each method's top-k unheard songs.

A "positive" is an event whose reward is above 0.5 (like, complete, replay), a negative is below
(dislike, skip); a play is neither and only counts as history.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import preference, taste
from .config import RecommendConfig
from .recommend import v1_scores


@dataclass(frozen=True)
class Event:
    day: float
    song: int
    kind: str


@dataclass
class Fitted:
    scores: Callable[[np.ndarray], np.ndarray]
    prob: Callable[[np.ndarray], np.ndarray] | None = None


# A method is fit(X, history, now_day, rng) -> Fitted.
Method = Callable[[np.ndarray, list[Event], float, np.random.Generator], Fitted]


def _label(kind: str) -> int | None:
    r = preference.reward(kind)
    return None if r == 0.5 else int(r > 0.5)


def _base_rate(history: list[Event]) -> float:
    labels = [lbl for e in history if (lbl := _label(e.kind)) is not None]
    return float(np.mean(labels)) if labels else 0.5


def _constant(n: int, v: float) -> np.ndarray:
    return np.full(n, v)


def random_method(X, history, now, rng) -> Fitted:
    p = _base_rate(history)
    return Fitted(lambda idx: rng.random(len(idx)), lambda idx: _constant(len(idx), p))


def _weights(history: list[Event], now: float, cfg: RecommendConfig) -> dict[int, float]:
    rows = [(e.song, e.kind, now - e.day) for e in history]
    return taste.weights_from_events(rows, cfg.recency_half_life_days)


def most_played_method(cfg: RecommendConfig = RecommendConfig()) -> Method:
    """Single-user popularity: the v1 recency-weighted event total per song."""

    def fit(X, history, now, rng) -> Fitted:
        w = np.zeros(len(X))
        for song, v in _weights(history, now, cfg).items():
            w[song] = v
        p = _base_rate(history)
        return Fitted(lambda idx: w[idx], lambda idx: _constant(len(idx), p))

    return fit


def _taste(X, history, now, cfg) -> tuple[np.ndarray, np.ndarray]:
    weights = _weights(history, now, cfg)
    liked = {i: v for i, v in weights.items() if v > 0}
    ids = sorted(liked)
    clusters = taste.cluster(ids, [""] * len(ids), X[ids], liked, cfg)
    disliked = [i for i, v in weights.items() if v < 0]
    medoids = np.stack([k.medoid_vec for k in clusters]) if clusters else np.empty((0, X.shape[1]))
    return medoids, X[disliked] if disliked else np.empty((0, X.shape[1]))


def v1_method(cfg: RecommendConfig = RecommendConfig()) -> Method:
    def fit(X, history, now, rng) -> Fitted:
        medoids, dis = _taste(X, history, now, cfg)
        if not len(medoids):
            return Fitted(lambda idx: np.zeros(len(idx)))
        return Fitted(lambda idx: v1_scores(X[idx], medoids, dis, cfg.dislike_penalty)[0])

    return fit


def v2_method(
    projection: preference.Projection, cfg: RecommendConfig = RecommendConfig(), sample: bool = True
) -> Method:
    """`sample=True` scores with one Thompson draw per refit, as a playlist build does; False uses
    the posterior mean and also reports it as a probability for calibration."""

    def fit(X, history, now, rng) -> Fitted:
        medoids, _ = _taste(X, history, now, cfg)
        if not len(medoids):
            medoids = X.mean(axis=0, keepdims=True)
        songs = np.array([e.song for e in history], dtype=int)
        model = preference.fit(
            X,
            medoids,
            X[songs],
            [e.kind for e in history],
            np.array([now - e.day for e in history]),
            cfg,
            projection=projection,
        )
        theta = model.weights(rng if sample else None, cfg.ts_exploration)

        def scores(idx):
            return model.predict(X[idx], theta)[0]

        return Fitted(scores, None if sample else (lambda idx: np.clip(scores(idx), 0.0, 1.0)))

    return fit


@dataclass
class MethodResult:
    ranks: list[float] = field(default_factory=list)
    scored: list[tuple[float, int]] = field(default_factory=list)
    probs: list[tuple[float, int]] = field(default_factory=list)
    oracle: list[tuple[float, float]] = field(default_factory=list)  # (day, like-rate)


def rank_of_target(target: float, negatives: np.ndarray) -> float:
    """1-based rank; ties count half, so constant scores land in the middle, not at the top."""
    return 1.0 + float((negatives > target).sum()) + 0.5 * float((negatives == target).sum())


def hit_rate(ranks, k: int) -> float:
    r = np.asarray(ranks, dtype=float)
    return float((r <= k).mean()) if len(r) else float("nan")


def ndcg(ranks, k: int) -> float:
    r = np.asarray(ranks, dtype=float)
    if not len(r):
        return float("nan")
    return float(np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0).mean())


def auc(pairs) -> float:
    """Mann-Whitney AUC with ties counted half; nan when one class is missing."""
    if not pairs:
        return float("nan")
    s, y = (np.asarray(v, dtype=float) for v in zip(*pairs, strict=True))
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def reliability(pairs, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed positive rate, count) per non-empty equal-width bin."""
    if not pairs:
        return []
    p, y = (np.asarray(v, dtype=float) for v in zip(*pairs, strict=True))
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [(float(p[b == i].mean()), float(y[b == i].mean()), int((b == i).sum())) for i in np.unique(b)]


def ece(pairs, bins: int = 10) -> float:
    rel = reliability(pairs, bins)
    n = sum(c for *_, c in rel)
    return float(sum(c * abs(p - o) for p, o, c in rel) / n) if n else float("nan")


def brier(pairs) -> float:
    if not pairs:
        return float("nan")
    p, y = (np.asarray(v, dtype=float) for v in zip(*pairs, strict=True))
    return float(((p - y) ** 2).mean())


def replay(
    X: np.ndarray,
    events: list[Event],
    methods: dict[str, Method],
    *,
    k: int = 10,
    n_neg: int = 99,
    warmup: int = 50,
    refit_every: int = 10,
    seed: int = 0,
    oracle: Callable[[float], np.ndarray] | None = None,
) -> dict[str, MethodResult]:
    """`oracle(day)` returns every song's true like-probability at that time (synthetic data only)."""
    events = sorted(events, key=lambda e: e.day)
    rng = np.random.default_rng(seed)
    results = {name: MethodResult() for name in methods}
    for start in range(warmup, len(events), refit_every):
        history, block = events[:start], events[start : start + refit_every]
        now = history[-1].day
        heard = {e.song for e in history}
        unheard = np.array([i for i in range(len(X)) if i not in heard])
        # Negatives are drawn once per event and shared by all methods, so they face the same test.
        tests = []
        for e in block:
            pool = unheard[unheard != e.song]
            negs = rng.choice(pool, size=min(n_neg, len(pool)), replace=False) if len(pool) else pool
            tests.append((e, _label(e.kind), negs))
        for name, method in methods.items():
            fitted = method(X, history, now, np.random.default_rng(rng.integers(2**63)))
            res = results[name]
            for e, label, negs in tests:
                if label is None:
                    continue
                s = fitted.scores(np.concatenate([[e.song], negs]).astype(int))
                res.scored.append((float(s[0]), label))
                if label == 1:
                    res.ranks.append(rank_of_target(s[0], s[1:]))
                if fitted.prob is not None:
                    res.probs.append((float(fitted.prob(np.array([e.song]))[0]), label))
            if oracle is not None and len(unheard):
                s = fitted.scores(unheard)
                # Random tie-break, so a method that scores everything equal is not credited
                # with whatever order the unheard songs happen to be in.
                order = np.lexsort((rng.random(len(s)), -s))[:k]
                res.oracle.append((now, float(oracle(now)[unheard[order]].mean())))
    return results


def summarise(results: dict[str, MethodResult], k: int = 10) -> dict[str, dict[str, float | int]]:
    out = {}
    for name, r in results.items():
        row: dict[str, float | int] = {
            f"hit_rate@{k}": hit_rate(r.ranks, k),
            f"ndcg@{k}": ndcg(r.ranks, k),
            "auc": auc(r.scored),
            "n_positive_targets": len(r.ranks),
            "n_labelled_events": len(r.scored),
        }
        if r.probs:
            row |= {"brier": brier(r.probs), "ece": ece(r.probs)}
        if r.oracle:
            row[f"oracle_like_rate@{k}"] = float(np.mean([v for _, v in r.oracle]))
        out[name] = row
    return out
