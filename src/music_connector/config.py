import os
from dataclasses import dataclass, field

DATABASE_URL = os.environ.get(
    "MC_DATABASE_URL",
    f"postgresql://music:music@localhost:{os.environ.get('MC_DB_PORT', '5433')}/music_connector",
)
MODEL_ID = os.environ.get("MC_MODEL_ID", "OpenMuQ/MuQ-MuLan-large")
SAMPLE_RATE = 24_000
WINDOW_S = 10.0
EMBED_DIM = 512


@dataclass(frozen=True)
class EventWeights:
    """How strongly each event counts as taste evidence. Negative = dislike evidence."""

    like: float = 1.0
    dislike: float = -1.0
    skip: float = -0.7
    complete: float = 0.3
    replay: float = 0.6
    # Takeout plays are weak: autoplay and background listening inflate them.
    play: float = 0.1

    def get(self, kind: str) -> float:
        return getattr(self, kind)


@dataclass(frozen=True)
class RecommendConfig:
    size: int = 25
    # Slot quotas; slots a pool can't fill roll over to "taste".
    slots: dict[str, float] = field(
        default_factory=lambda: {"taste": 0.55, "indie": 0.20, "classic": 0.15, "wildcard": 0.10}
    )
    recency_half_life_days: float = 90.0
    # Ward merge-distance cutoff on unit vectors. Lower = more, tighter taste clusters.
    cluster_distance: float = 1.2
    mmr_lambda: float = 0.7
    max_per_artist: int = 2
    # Above this cosine two songs are treated as the same recording (re-uploads, bitrates,
    # lyric videos); measured 0.998 for one track at two bitrates.
    duplicate_cosine: float = 0.97
    dislike_penalty: float = 0.5
    neighbours_per_cluster: int = 200
    exclude_heard_days: int = 30
