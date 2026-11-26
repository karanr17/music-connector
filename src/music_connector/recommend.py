"""v1 playlist builder: taste-cluster retrieval + exploration slots + MMR diversity + smooth ordering.

No learned model yet; the Thompson-sampling preference model replaces `_score` in v2.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import taste
from .config import RecommendConfig
from .db import as_array

SONG_COLS = "id, title, artist, youtube_id, embedding"


@dataclass
class Candidate:
    id: int
    title: str
    artist: str | None
    youtube_id: str | None
    vec: np.ndarray
    score: float = 0.0
    near: str = ""


@dataclass
class Pick:
    candidate: Candidate
    slot: str


def _excluded_ids(conn, cfg: RecommendConfig, weights: dict[int, float]) -> list[int]:
    recent = conn.execute(
        "SELECT DISTINCT song_id FROM events WHERE at > now() - make_interval(days => %s)",
        (cfg.exclude_heard_days,),
    ).fetchall()
    disliked = [i for i, w in weights.items() if w < 0]
    return [r[0] for r in recent] + disliked


def _rows_to_candidates(rows) -> list[Candidate]:
    return [Candidate(r[0], r[1], r[2], r[3], as_array(r[4])) for r in rows]


def _nearest(conn, vec, exclude, limit, pool: str | None = None) -> list[Candidate]:
    pool_sql = "AND %s = ANY(pools)" if pool else ""
    params = [list(exclude)] + ([pool] if pool else []) + [vec, limit]
    return _rows_to_candidates(
        conn.execute(
            f"""SELECT {SONG_COLS} FROM songs
            WHERE embedding IS NOT NULL AND NOT (id = ANY(%s)) {pool_sql}
            ORDER BY embedding <=> %s LIMIT %s""",
            params,
        ).fetchall()
    )


def _random(conn, exclude, limit) -> list[Candidate]:
    return _rows_to_candidates(
        conn.execute(
            f"""SELECT {SONG_COLS} FROM songs
            WHERE embedding IS NOT NULL AND NOT (id = ANY(%s)) ORDER BY random() LIMIT %s""",
            (list(exclude), limit),
        ).fetchall()
    )


def _excluded_vecs(conn, exclude) -> np.ndarray:
    rows = conn.execute(
        "SELECT embedding FROM songs WHERE id = ANY(%s) AND embedding IS NOT NULL", (list(exclude),)
    ).fetchall()
    return np.stack([as_array(r[0]) for r in rows]) if rows else np.empty((0, 0), dtype=np.float32)


def _drop_duplicates(group: list[Candidate], avoid: np.ndarray, cfg: RecommendConfig) -> list[Candidate]:
    if not group or not avoid.size:
        return group
    sims = np.stack([c.vec for c in group]) @ avoid.T
    return [c for c, s in zip(group, sims.max(axis=1), strict=True) if s <= cfg.duplicate_cosine]


def _score(cands: list[Candidate], clusters, dis_vecs: np.ndarray, cfg: RecommendConfig) -> None:
    if not cands:
        return
    C = np.stack([c.vec for c in cands])
    M = np.stack([k.medoid_vec for k in clusters])
    sims = C @ M.T
    best = sims.argmax(axis=1)
    penalty = (C @ dis_vecs.T).max(axis=1) if len(dis_vecs) else np.zeros(len(cands))
    for c, k, s, p in zip(cands, best, sims.max(axis=1), penalty, strict=True):
        c.score = float(s - cfg.dislike_penalty * max(p, 0.0))
        c.near = clusters[k].medoid_title


def _allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of `total` items by fractional shares."""
    norm = sum(shares.values()) or 1.0
    raw = {k: total * v / norm for k, v in shares.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def _mmr_fill(
    group: list[Candidate], quota: int, slot: str, picks: list[Pick], artists: Counter, cfg: RecommendConfig
) -> int:
    """Greedy maximal-marginal-relevance selection. Returns how many slots went unfilled."""
    taken = {p.candidate.id for p in picks}
    pool = {c.id: c for c in group if c.id not in taken}
    for _ in range(quota):
        chosen_vecs = np.stack([p.candidate.vec for p in picks]) if picks else None
        best, best_val = None, -np.inf
        for c in pool.values():
            if c.artist and artists[c.artist] >= cfg.max_per_artist:
                continue
            redundancy = float((chosen_vecs @ c.vec).max()) if chosen_vecs is not None else 0.0
            if redundancy > cfg.duplicate_cosine:
                continue
            val = cfg.mmr_lambda * c.score - (1 - cfg.mmr_lambda) * redundancy
            if val > best_val:
                best, best_val = c, val
        if best is None:
            return quota
        picks.append(Pick(best, slot))
        artists[best.artist] += 1
        del pool[best.id]
        quota -= 1
    return 0


def _smooth_order(picks: list[Pick]) -> list[Pick]:
    """Nearest-neighbour chain from the strongest pick, so adjacent songs sound alike."""
    if len(picks) < 3:
        return picks
    remaining = sorted(picks, key=lambda p: p.candidate.score, reverse=True)
    order = [remaining.pop(0)]
    while remaining:
        last = order[-1].candidate.vec
        i = max(range(len(remaining)), key=lambda j: float(remaining[j].candidate.vec @ last))
        order.append(remaining.pop(i))
    return order


def build_playlist(
    conn, cfg: RecommendConfig = RecommendConfig()
) -> tuple[list[Pick], list[taste.TasteCluster]]:
    clusters, dis_vecs = taste.build(conn, cfg)
    if not clusters:
        raise LookupError(
            "No taste signal yet: rate some embedded songs (`mc rate`) or embed songs you play "
            "a lot (Takeout imports only count once their audio is embedded)."
        )
    conn.execute("SET hnsw.iterative_scan = relaxed_order")  # keeps filtered ANN from under-returning
    weights = taste.song_weights(conn, cfg.recency_half_life_days)
    exclude = set(_excluded_ids(conn, cfg, weights)) | {m for k in clusters for m in k.member_ids}
    avoid = _excluded_vecs(conn, exclude)

    quotas = _allocate(cfg.size, cfg.slots)
    picks: list[Pick] = []
    artists: Counter = Counter()
    shortfall = 0

    per_cluster = _allocate(quotas.pop("taste", 0), {str(i): k.importance for i, k in enumerate(clusters)})
    taste_pool: list[Candidate] = []
    for i, k in enumerate(clusters):
        group = _drop_duplicates(
            _nearest(conn, k.medoid_vec, exclude, cfg.neighbours_per_cluster), avoid, cfg
        )
        _score(group, clusters, dis_vecs, cfg)
        taste_pool += group
        shortfall += _mmr_fill(group, per_cluster[str(i)], "taste", picks, artists, cfg)

    for slot, quota in quotas.items():
        if slot == "wildcard":
            group = _random(conn, exclude, 50)
        else:
            group = [c for k in clusters for c in _nearest(conn, k.medoid_vec, exclude, 50, pool=slot)]
        group = _drop_duplicates(group, avoid, cfg)
        _score(group, clusters, dis_vecs, cfg)
        shortfall += _mmr_fill(group, quota, slot, picks, artists, cfg)

    if shortfall:
        _mmr_fill(taste_pool, shortfall, "taste", picks, artists, cfg)
    return _smooth_order(picks), clusters
