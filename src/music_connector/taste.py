"""Multi-interest taste profile, after PinnerSage (Pal et al., KDD 2020).

Instead of averaging everything you like into one vector, cluster liked songs with Ward
linkage and represent each cluster by its medoid (a real song) plus an importance score.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .config import EventWeights, RecommendConfig
from .db import as_array


@dataclass
class TasteCluster:
    medoid_id: int
    medoid_title: str
    medoid_vec: np.ndarray
    importance: float
    member_ids: list[int]


def song_weights(conn, half_life_days: float, weights: EventWeights = EventWeights()
                 ) -> dict[int, float]:
    """Net taste evidence per song: event weights × exponential recency decay, summed."""
    rows = conn.execute(
        "SELECT song_id, kind, extract(epoch FROM now() - at) / 86400.0 FROM events"
    ).fetchall()
    totals: dict[int, float] = defaultdict(float)
    for song_id, kind, age_days in rows:
        decay = 0.5 ** (max(float(age_days), 0.0) / half_life_days)
        totals[song_id] += weights.get(kind) * decay
    return dict(totals)


def _embeddings(conn, ids) -> tuple[list[int], list[str], np.ndarray]:
    rows = conn.execute(
        "SELECT id, title, embedding FROM songs WHERE id = ANY(%s) AND embedding IS NOT NULL",
        (list(ids),),
    ).fetchall()
    if not rows:
        return [], [], np.empty((0, 0), dtype=np.float32)
    return [r[0] for r in rows], [r[1] for r in rows], np.stack([as_array(r[2]) for r in rows])


def build(conn, cfg: RecommendConfig = RecommendConfig()
          ) -> tuple[list[TasteCluster], np.ndarray]:
    """Returns (taste clusters sorted by importance, disliked-song vectors)."""
    weights = song_weights(conn, cfg.recency_half_life_days)
    liked = {i: w for i, w in weights.items() if w > 0}
    disliked = [i for i, w in weights.items() if w < 0]

    ids, titles, X = _embeddings(conn, liked)
    _, _, dis_vecs = _embeddings(conn, disliked)
    if not ids:
        return [], dis_vecs

    w = np.array([liked[i] for i in ids])
    labels = (np.ones(len(ids), dtype=int) if len(ids) == 1 else
              fcluster(linkage(X, method="ward"), t=cfg.cluster_distance, criterion="distance"))

    clusters = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        sims = X[idx] @ X[idx].T
        # Weighted medoid: the member closest to the others, counting stronger likes more.
        m = idx[np.argmax(sims @ w[idx])]
        clusters.append(TasteCluster(ids[m], titles[m], X[m], float(w[idx].sum()),
                                     [ids[i] for i in idx]))
    clusters.sort(key=lambda c: c.importance, reverse=True)
    return clusters, dis_vecs
