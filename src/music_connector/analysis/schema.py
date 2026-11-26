"""SongAnalysis v1: the JSON contract between music-connector and its consumers (the visual
music player first). `schema/song-analysis.v1.schema.json` is generated from these models.

Versioning: `schemaVersion` is "1.<minor>". Minor bumps only add optional fields, so a v1
reader must ignore fields it does not know. Removing or changing the meaning of a field is a
new major version with a new schema file.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

SCHEMA_VERSION = "1.0"

Seconds = float


class _Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class Source(_Model):
    sha256: str = Field(description="SHA-256 of the audio file bytes; the cache key.")
    file_name: str | None = Field(None, description="Base name only, never a full path.")
    duration_s: Seconds
    sample_rate: int = Field(description="Rate the audio was analysed at (Hz).")


class Analyzer(_Model):
    name: Literal["music-connector"] = "music-connector"
    version: str
    model: str = Field(description="Audio-text embedding model used for windows and descriptors.")
    calibration: str = Field(
        description="Corpus the descriptor z-scores are relative to, e.g. 'cc-reference-90' or 'library'."
    )


class Tempo(_Model):
    bpm: float
    confidence: float = Field(
        ge=0,
        le=1,
        description="Onset autocorrelation at the beat period (1 = metronomic). Below ~0.45 the "
        "tempo may be off by 2x or wrong; a weak warning, not a guarantee.",
    )
    beats_per_bar: int = Field(4, description="Assumed meter; v1 only detects 4/4 downbeat phase.")


class Energy(_Model):
    """Envelopes sampled at `frameRate` Hz from t = 0. Each series is scaled to 0..1 within this
    song (48 dB range below its own 99th percentile), so they show change over time, not absolute
    level; `loudnessDb` gives the absolute level."""

    frame_rate: float
    loudness_db: float = Field(description="95th percentile RMS level in dBFS.")
    rms: list[float]
    low: list[float] = Field(description="20-250 Hz: kick, bass.")
    mid: list[float] = Field(description="250-4000 Hz: vocals, most instruments.")
    high: list[float] = Field(description="4-12 kHz: hats, cymbals, air.")


class Descriptor(_Model):
    label: str
    cos: float = Field(description="Raw cosine between the audio and the text prompt.")
    z: float = Field(
        description="Cosine standardised against the calibration corpus (prompt bias partly "
        "removed, divided by the axis's spread). Above ~1.5 is a strong hint; still a hint, not a fact."
    )


Descriptors = dict[str, list[Descriptor]]
"""Axis name (genre, mood, instruments, vocals, texture) → top-k prompts ranked by z."""


class Segment(_Model):
    start: Seconds
    end: Seconds
    label: str = Field(description="Recurrence label: same letter = same material; ′ = a variation.")
    group: str = Field(description="The letter alone, e.g. 'A' for both 'A' and 'A′'.")
    variant: int = Field(0, description="0 for the first form of a group, 1 for a variation (′).")
    similarity: float | None = Field(
        None, description="Cosine to the group's first occurrence; absent on that first occurrence."
    )
    boundary_strength: float = Field(
        ge=0, le=1, description="Novelty at this segment's start (0 for the first segment)."
    )
    energy: float = Field(ge=0, le=1, description="Mean of `energy.rms` over the segment.")
    descriptors: Descriptors = Field(default_factory=dict)


class Windows(_Model):
    """Per-window detail, only present when requested (it is the bulk of the file)."""

    window_s: Seconds
    hop_s: Seconds
    starts: list[Seconds]
    descriptors: list[Descriptors]


class NearestSong(_Model):
    song_id: int
    title: str
    artist: str | None
    cosine: float


class SongAnalysis(_Model):
    schema_version: str = Field(SCHEMA_VERSION, pattern=r"^1\.\d+$")
    analyzer: Analyzer
    source: Source
    tempo: Tempo
    beats: list[Seconds]
    downbeats: list[Seconds]
    onsets: list[Seconds]
    energy: Energy
    segments: list[Segment]
    descriptors: Descriptors = Field(description="Whole-song descriptors.")
    windows: Windows | None = None
    nearest: list[NearestSong] | None = Field(
        None, description="Closest songs in the local library; only when requested."
    )

    def to_json(self, indent: int | None = None) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True, indent=indent)


def json_schema() -> dict:
    schema = SongAnalysis.model_json_schema(by_alias=True, mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/karanr17/music-connector/schema/song-analysis.v1.schema.json"
    # It has a default in Python but is always written, and readers branch on it.
    schema["required"] = ["schemaVersion", *schema["required"]]
    schema["title"] = "SongAnalysis"
    schema["description"] = (__doc__ or "").strip()
    return schema
