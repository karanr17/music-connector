"""Segmentation and recurrence labelling. Pure numpy/scipy on precomputed features, so it can be
tested on synthetic inputs.

Boundaries come from novelty (how different the music just before a time is from the music
just after it) on two feature sets: beat-synchronous chroma + timbre (precise in time) and
MuLan window embeddings (coarse, 5 s hop, but they see arrangement changes chroma misses).
Boundaries are then placed on downbeats. Which segments are "the same material" is decided
from MuLan embeddings alone: on Sunflower the two choruses scored 0.95-0.97 while verse vs
chorus was ~0.91.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

PRIME = "′"


@dataclass(frozen=True)
class SegmentConfig:
    """Defaults were tuned on 25 SALAMI-annotated live recordings kept apart from the README's
    quality-check set (`scripts/eval_analysis.py tune`)."""

    # Seconds of context compared on each side of a candidate boundary.
    span_s: float = 8.0
    min_segment_s: float = 10.0
    # No boundary closer than this to the start or end of the song.
    edge_s: float = 4.0
    # A boundary needs novelty above median + k * MAD of all candidates.
    threshold_mads: float = 0.5
    # At most one boundary per this many seconds of audio on average.
    max_rate_s: float = 10.0
    # Chroma/timbre novelty is sharper in time; the 5 s-hop embeddings mostly confirm it.
    feature_weight: float = 0.85
    # Segments whose embeddings are at least this similar share a letter (average linkage).
    same_cosine: float = 0.92
    # Within a letter, occurrences below this similarity to the first one get a prime.
    variant_below: float = 0.95
    # Off by default: on the tuning set it removed more true boundaries than false ones.
    merge_adjacent_same: bool = False
    # Novelty is normalised per song, so a song with no real changes would still get its noise
    # picked; below this raw contrast nothing counts as a change.
    min_contrast: float = 0.03


@dataclass
class SegmentInfo:
    start: float
    end: float
    group: int
    variant: int
    similarity: float | None
    boundary_strength: float
    energy: float = 0.0
    vector: np.ndarray | None = field(default=None, repr=False)

    @property
    def letter(self) -> str:
        return group_letter(self.group)

    @property
    def label(self) -> str:
        return self.letter + PRIME * self.variant


def group_letter(i: int) -> str:
    """0 → A … 25 → Z, 26 → AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def contrast_novelty(
    vecs: np.ndarray, starts: np.ndarray, ends: np.ndarray, times: np.ndarray, span: float
) -> np.ndarray:
    """For each time T, compare items lying wholly in [T - span, T] with items in [T, T + span]:
    (mean within-side similarity) - (mean cross similarity). High = a change happens at T."""
    out = np.zeros(len(times))
    if len(vecs) < 2:
        return out
    ssm = _unit(vecs) @ _unit(vecs).T
    tol = 1e-6
    for k, t in enumerate(times):
        before = np.flatnonzero((ends <= t + tol) & (starts >= t - span - tol))
        after = np.flatnonzero((starts >= t - tol) & (ends <= t + span + tol))
        if len(before) == 0 or len(after) == 0:
            continue
        within = (ssm[np.ix_(before, before)].mean() + ssm[np.ix_(after, after)].mean()) / 2
        out[k] = max(within - ssm[np.ix_(before, after)].mean(), 0.0)
    return out


def _normalise(x: np.ndarray) -> np.ndarray:
    ref = np.percentile(x, 95) if len(x) else 0.0
    return np.clip(x / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(x)


def pick_boundaries(
    times: np.ndarray, novelty: np.ndarray, duration: float, cfg: SegmentConfig
) -> list[tuple[float, float]]:
    """Greedy peak picking: strongest candidates first, keeping `min_segment_s` between
    boundaries. Returns sorted (time, strength 0..1)."""
    ok = (times >= cfg.edge_s) & (times <= duration - cfg.edge_s)
    times, novelty = times[ok], novelty[ok]
    if len(times) == 0:
        return []
    med = np.median(novelty)
    mad = np.median(np.abs(novelty - med)) + 1e-9
    floor = med + cfg.threshold_mads * mad
    cap = max(int(duration / cfg.max_rate_s), 0)
    top = novelty.max() + 1e-9
    chosen: list[tuple[float, float]] = []
    for i in np.argsort(-novelty, kind="stable"):
        if len(chosen) >= cap or novelty[i] <= floor:
            break
        if all(abs(times[i] - t) >= cfg.min_segment_s for t, _ in chosen):
            chosen.append((float(times[i]), float(novelty[i] / top)))
    return sorted(chosen)


def segment_vectors(
    bounds: list[float], window_starts: np.ndarray, window_vecs: np.ndarray, window_s: float
) -> np.ndarray:
    """One unit vector per segment: windows weighted by how much of them lies inside it,
    squared so windows straddling a boundary count little."""
    starts = np.asarray(window_starts, dtype=float)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        overlap = np.clip(np.minimum(starts + window_s, b) - np.maximum(starts, a), 0, None) / window_s
        w = overlap**2
        if w.sum() == 0:
            w = np.exp(-np.abs(starts + window_s / 2 - (a + b) / 2))
        out.append(_unit((w[:, None] * window_vecs).sum(axis=0)))
    return np.array(out)


def label(vectors: np.ndarray, cfg: SegmentConfig) -> list[tuple[int, int, float | None]]:
    """(group, variant, similarity to the group's first occurrence) per segment. Groups are
    numbered in order of first appearance, so the song always starts with A."""
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        clusters = np.array([1])
    else:
        dist = np.clip(1.0 - _unit(vectors) @ _unit(vectors).T, 0.0, 2.0)
        condensed = dist[np.triu_indices(n, 1)]
        clusters = fcluster(linkage(condensed, "average"), t=1.0 - cfg.same_cosine, criterion="distance")
    order: dict[int, int] = {}
    first: dict[int, int] = {}
    out = []
    for i, c in enumerate(clusters):
        if c not in order:
            order[c] = len(order)
            first[c] = i
            out.append((order[c], 0, None))
            continue
        sim = float(_unit(vectors[i]) @ _unit(vectors[first[c]]))
        out.append((order[c], int(sim < cfg.variant_below), sim))
    return out


def segment(
    *,
    duration: float,
    beats: np.ndarray,
    downbeats: np.ndarray,
    beat_features: np.ndarray,
    window_starts: np.ndarray,
    window_vecs: np.ndarray,
    window_s: float,
    rms: np.ndarray | None = None,
    rms_rate: float = 10.0,
    cfg: SegmentConfig = SegmentConfig(),
) -> list[SegmentInfo]:
    beats = np.asarray(beats, dtype=float)
    window_starts = np.asarray(window_starts, dtype=float)
    candidates = np.asarray(downbeats, dtype=float) if len(downbeats) >= 4 else beats
    if len(candidates) < 2:
        # No usable beat grid (ambient, speech): fall back to a 1 s grid.
        candidates = np.arange(0.0, duration, 1.0)

    # Windows are as long as the span, so few lie wholly on one side of a candidate; place
    # each at its centre instead.
    centres = window_starts + window_s / 2
    emb = contrast_novelty(window_vecs, centres, centres, candidates, cfg.span_s)
    if len(beat_features) >= 2:
        n = len(beat_features)
        feat = contrast_novelty(beat_features, beats[:n], beats[1 : n + 1], candidates, cfg.span_s)
        nov = cfg.feature_weight * _normalise(feat) + (1 - cfg.feature_weight) * _normalise(emb)
        raw = cfg.feature_weight * feat + (1 - cfg.feature_weight) * emb
    else:
        nov, raw = _normalise(emb), emb
    nov = np.where(raw >= cfg.min_contrast, nov, 0.0)

    picked = pick_boundaries(candidates, nov, duration, cfg)
    bounds = [0.0] + [t for t, _ in picked] + [duration]
    strengths = [0.0] + [s for _, s in picked]
    vecs = segment_vectors(bounds, window_starts, window_vecs, window_s)
    labels = label(vecs, cfg)

    segs = [
        SegmentInfo(a, b, g, v, sim, s, vector=vec)
        for a, b, (g, v, sim), s, vec in zip(bounds[:-1], bounds[1:], labels, strengths, vecs, strict=True)
    ]
    if cfg.merge_adjacent_same:
        segs = _merge_adjacent(segs, window_starts, window_vecs, window_s, cfg)
    if rms is not None and len(rms):
        for s in segs:
            a, b = int(s.start * rms_rate), max(int(s.end * rms_rate), int(s.start * rms_rate) + 1)
            s.energy = float(np.mean(rms[a:b])) if a < len(rms) else 0.0
    return segs


def _merge_adjacent(segs, window_starts, window_vecs, window_s, cfg) -> list[SegmentInfo]:
    """A boundary between two segments with the same label marks no change of material, so
    drop it and relabel."""
    merged = [segs[0]]
    for s in segs[1:]:
        if s.group == merged[-1].group and s.variant == merged[-1].variant:
            merged[-1].end = s.end
        else:
            merged.append(s)
    if len(merged) == len(segs):
        return segs
    bounds = [s.start for s in merged] + [merged[-1].end]
    vecs = segment_vectors(bounds, window_starts, window_vecs, window_s)
    for s, (g, v, sim), vec in zip(merged, label(vecs, cfg), vecs, strict=True):
        s.group, s.variant, s.similarity, s.vector = g, v, sim, vec
    return merged
