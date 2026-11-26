"""Offline replay eval: v1 cosine-to-medoid vs v2 Thompson sampling vs random and most-played.

    uv run python scripts/replay_eval.py synthetic   # writes docs/replay-eval.{png,json}
    uv run python scripts/replay_eval.py owner       # your local events table, read-only

Only aggregate metrics are printed or written: never song ids, titles or events. The owner run
opens a read-only transaction and writes nothing to the database or the repo.

The v2 hyper-parameters in config.py were picked on a synthetic world with a different seed
(world 7, log 8) from the one reported here (world 0, logs 1-5).
"""

import json
import sys
from pathlib import Path

import numpy as np

from music_connector import preference, replay
from music_connector.config import RecommendConfig
from music_connector.synthetic import make_world, simulate_log

DOCS = Path(__file__).resolve().parent.parent / "docs"
K = 10
LOG_SEEDS = (1, 2, 3, 4, 5)
METHODS = ("random", "most-played", "v1 cosine", "v2 posterior mean", "v2 Thompson")


def methods(X: np.ndarray, cfg: RecommendConfig) -> dict[str, replay.Method]:
    proj = preference.Projection.fit(X, cfg.pca_dim)
    return {
        "random": replay.random_method,
        "most-played": replay.most_played_method(cfg),
        "v1 cosine": replay.v1_method(cfg),
        "v2 posterior mean": replay.v2_method(proj, cfg, sample=False),
        "v2 Thompson": replay.v2_method(proj, cfg, sample=True),
    }


def run_synthetic(cfg: RecommendConfig):
    world = make_world(seed=0)
    runs = []
    for s in LOG_SEEDS:
        events = simulate_log(world, seed=s)
        runs.append(
            replay.replay(
                world.X, events, methods(world.X, cfg), k=K, refit_every=25, seed=s, oracle=world.like_prob
            )
        )
    summaries = [replay.summarise(r, K) for r in runs]
    agg = {}
    for name in METHODS:
        agg[name] = {}
        for metric in summaries[0][name]:
            vals = np.array([s[name][metric] for s in summaries], dtype=float)
            agg[name][metric] = {"mean": round(float(vals.mean()), 4), "sd": round(float(vals.std()), 4)}
    meta = {
        "data": "synthetic (labelled: not real listening data)",
        "songs": len(world.X),
        "events_per_log": len(simulate_log(world, seed=LOG_SEEDS[0])),
        "logs": len(LOG_SEEDS),
        "taste_shift_day": world.phases[1][0],
    }
    return meta, agg, runs


def _owner_data():
    import psycopg
    from pgvector.psycopg import register_vector

    from music_connector.config import DATABASE_URL
    from music_connector.db import as_array

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        register_vector(conn)
        songs = conn.execute("SELECT id, embedding FROM songs WHERE embedding IS NOT NULL").fetchall()
        rows = conn.execute(
            """SELECT e.song_id, e.kind, extract(epoch FROM e.at) / 86400.0 FROM events e
               JOIN songs s ON s.id = e.song_id WHERE s.embedding IS NOT NULL"""
        ).fetchall()
        conn.rollback()
    index = {sid: i for i, (sid, _) in enumerate(songs)}
    X = np.stack([as_array(v) for _, v in songs]) if songs else np.empty((0, 512))
    return X, [replay.Event(float(day), index[sid], kind) for sid, kind, day in rows]


def run_owner(cfg: RecommendConfig, warmup: int) -> dict:
    X, events = _owner_data()
    out: dict = {
        "data": "owner's local events table (aggregate only)",
        "songs": len(X),
        "events": len(events),
    }
    labelled = sum(replay.preference.reward(e.kind) != 0.5 for e in events[warmup:])
    if len(events) <= warmup or len(X) < 3 or labelled == 0:
        out["result"] = (
            f"too few events to replay: need more than {warmup} feedback events plus held-out "
            "likes/completions/skips after them"
        )
        return out
    n_neg = min(99, len(X) - 2)
    res = replay.replay(X, events, methods(X, cfg), k=K, n_neg=n_neg, warmup=warmup, refit_every=5)
    out["n_negatives"] = n_neg
    out["metrics"] = replay.summarise(res, K)
    return out


PALETTE = {
    "random": "#8c8c8c",
    "most-played": "#f47230",
    "v1 cosine": "#257de0",
    "v2 posterior mean": "#00b894",
    "v2 Thompson": "#6a4ce2",
}


def plot(meta, agg, runs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    ink, faint = "#191919", "#666666"

    for name in METHODS:
        days = np.array([d for d, _ in runs[0][name].oracle])
        vals = np.mean([[v for _, v in r[name].oracle] for r in runs], axis=0)
        a.plot(days, vals, color=PALETTE[name], lw=2)
        a.annotate(
            name,
            (days[-1], vals[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=ink,
            fontsize=8,
        )
    a.axvline(meta["taste_shift_day"], color=faint, lw=1, ls="--")
    a.text(meta["taste_shift_day"] + 2, 0.97, "taste shift", color=faint, fontsize=8, va="top")
    a.set(xlabel="day of the log", ylabel=f"true like-probability of top {K}", ylim=(0, 1))
    a.set_title(f"Oracle: quality of the top {K} unheard songs", loc="left", color=ink)
    a.set_xlim(0, days[-1] * 1.32)
    a.set_xticks(np.arange(0, days[-1] + 1, 30))
    a.grid(axis="y", color="#e6e6e6", lw=0.8)

    metrics = [(f"hit_rate@{K}", f"HR@{K}"), (f"ndcg@{K}", f"NDCG@{K}"), ("auc", "AUC")]
    width = 0.15
    for i, name in enumerate(METHODS):
        x = np.arange(len(metrics)) + (i - 2) * width
        m = [agg[name][k]["mean"] for k, _ in metrics]
        sd = [agg[name][k]["sd"] for k, _ in metrics]
        b.bar(x, m, width * 0.88, yerr=sd, color=PALETTE[name], label=name, error_kw={"lw": 0.8})
    b.set_xticks(range(len(metrics)), [lbl for _, lbl in metrics])
    b.set(ylim=(0, 1))
    b.set_title("Next-event replay (held-out)", loc="left", color=ink)
    b.legend(frameon=False, fontsize=7.5, loc="upper left")
    b.grid(axis="y", color="#e6e6e6", lw=0.8)

    pairs = [p for r in runs for p in r["v2 posterior mean"].probs]
    rel = replay.reliability(pairs, bins=10)
    c.plot([0, 1], [0, 1], color=faint, lw=1, ls="--")
    c.plot(
        [p for p, _, _ in rel], [o for _, o, _ in rel], "o-", color=PALETTE["v2 posterior mean"], lw=2, ms=5
    )
    c.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted P(positive)", ylabel="observed positive rate")
    ece = agg["v2 posterior mean"]["ece"]["mean"]
    c.set_title(f"v2 calibration (ECE {ece:.3f})", loc="left", color=ink)
    c.grid(color="#e6e6e6", lw=0.8)

    fig.suptitle(
        f"Synthetic data, not real listening: {meta['songs']} songs, {meta['logs']} logs of "
        f"{meta['events_per_log']} events (mean ± sd)",
        x=0.01,
        ha="left",
        color=faint,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    cfg = RecommendConfig()
    if source == "owner":
        print(json.dumps(run_owner(cfg, warmup=cfg.ts_min_events), indent=2))
        return
    meta, agg, runs = run_synthetic(cfg)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "replay-eval.json").write_text(json.dumps({"meta": meta, "metrics": agg}, indent=2) + "\n")
    plot(meta, agg, runs, DOCS / "replay-eval.png")
    print(json.dumps({"meta": meta, "metrics": agg}, indent=2))


if __name__ == "__main__":
    main()
