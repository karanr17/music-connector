"""Zero-shot MuLan descriptors, calibrated per prompt.

Raw text-audio cosines are not fully comparable across prompts: some prompts ("male vocals")
score high against almost everything. So each cosine is standardised against how the prompts
score on a reference corpus (see `zscores`), and prompts are ranked by that within an axis.
Calibration stats are kept separately for whole songs and for single windows, because window
cosines spread wider than song (mean-pooled) cosines.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

# Tempo, energy and era are deliberately absent: MuLan got all three wrong on the probe song,
# and DSP measures tempo and energy directly.
VOCABULARY: dict[str, list[str]] = {
    "genre": [
        "rock",
        "pop",
        "hip hop",
        "electronic dance music",
        "jazz",
        "classical",
        "folk",
        "country",
        "blues",
        "reggae",
        "heavy metal",
        "r&b and soul",
        "funk",
        "ambient",
        "latin",
        "orchestral film score",
        "punk",
        "dubstep",
        "drum and bass",
        "trip hop",
        "techno",
        "house",
        "gospel",
        "bluegrass",
    ],
    "mood": [
        "happy",
        "sad",
        "melancholic",
        "energetic",
        "calm",
        "dark",
        "romantic",
        "aggressive",
        "dreamy",
        "uplifting",
        "tense",
        "playful",
        "nostalgic",
        "epic",
    ],
    "instruments": [
        "acoustic guitar",
        "electric guitar",
        "distorted guitar",
        "piano",
        "synthesizer",
        "violin and strings",
        "cello",
        "drum kit",
        "drum machine",
        "808 bass",
        "bass guitar",
        "brass section",
        "saxophone",
        "organ",
        "electric piano",
        "flute",
        "banjo",
        "harmonica",
        "turntable scratching",
    ],
    "vocals": [
        "male vocals",
        "female vocals",
        "instrumental, no vocals",
        "rap vocals",
        "choir",
        "spoken word",
    ],
    "texture": [
        "lo-fi",
        "live concert recording",
        "clean polished studio production",
        "spacious, heavy reverb",
        "dry and intimate",
        "distorted and gritty",
    ],
}

PROMPT_TEMPLATES = {
    "genre": "{} music",
    "mood": "{} music",
    "instruments": "music with {}",
    "vocals": "{}",
    "texture": "{} sound",
}


def prompts(vocab: dict[str, list[str]] = VOCABULARY) -> list[tuple[str, str, str]]:
    """(axis, label, prompt text) for every descriptor."""
    return [
        (axis, label, PROMPT_TEMPLATES[axis].format(label))
        for axis, labels in vocab.items()
        for label in labels
    ]


def vocabulary_id(vocab: dict[str, list[str]] = VOCABULARY) -> str:
    text = json.dumps([p for _, _, p in prompts(vocab)])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class Calibration:
    name: str
    vocabulary: str
    song_mean: np.ndarray
    song_std: np.ndarray
    window_mean: np.ndarray
    window_std: np.ndarray
    songs: int = 0
    windows: int = 0

    @classmethod
    def from_cosines(cls, name: str, song_cos: np.ndarray, window_cos: np.ndarray) -> "Calibration":
        return cls(
            name=name,
            vocabulary=vocabulary_id(),
            song_mean=song_cos.mean(axis=0),
            song_std=song_cos.std(axis=0) + 1e-6,
            window_mean=window_cos.mean(axis=0),
            window_std=window_cos.std(axis=0) + 1e-6,
            songs=len(song_cos),
            windows=len(window_cos),
        )

    def to_dict(self) -> dict:
        def r(a):
            return [round(float(x), 5) for x in a]

        return {
            "name": self.name,
            "vocabulary": self.vocabulary,
            "songs": self.songs,
            "windows": self.windows,
            "prompts": [p for _, _, p in prompts()],
            "song": {"mean": r(self.song_mean), "std": r(self.song_std)},
            "window": {"mean": r(self.window_mean), "std": r(self.window_std)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            name=d["name"],
            vocabulary=d["vocabulary"],
            song_mean=np.array(d["song"]["mean"]),
            song_std=np.array(d["song"]["std"]),
            window_mean=np.array(d["window"]["mean"]),
            window_std=np.array(d["window"]["std"]),
            songs=d.get("songs", 0),
            windows=d.get("windows", 0),
        )


USER_CALIBRATION = Path(
    os.environ.get("MC_CALIBRATION", Path.home() / ".config" / "music-connector" / "calibration.json")
)


def load_calibration(path: Path | None = None) -> Calibration | None:
    """A user calibration (from `mc calibrate`) wins over the packaged reference one. Either is
    ignored if it was built for a different prompt vocabulary."""
    candidates = [path] if path else [USER_CALIBRATION]
    for p in candidates:
        if p and Path(p).exists():
            cal = Calibration.from_dict(json.loads(Path(p).read_text()))
            if cal.vocabulary == vocabulary_id():
                return cal
    packaged = resources.files(__package__).joinpath("calibration.json")
    if packaged.is_file():
        cal = Calibration.from_dict(json.loads(packaged.read_text()))
        if cal.vocabulary == vocabulary_id():
            return cal
    return None


# How much of each prompt's own bias to remove. Full per-prompt z-scores (1.0) ranked worse
# than raw cosines on the tuning set: prompts with a narrow spread ("choir", "spoken word")
# got inflated. Removing a quarter of the bias and using the axis-wide spread did best there
# and on the held-out set (README "Analysis quality").
PROMPT_BIAS_WEIGHT = 0.25


def zscores(cos: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """Standardise [..., n_prompts] cosines against calibration stats: subtract each prompt's
    corpus mean shrunk towards its axis mean, divide by the axis's pooled spread. Without
    calibration, fall back to z within each axis for this one input."""
    z = np.empty_like(cos, dtype=float)
    for idx in axis_indices().values():
        sub = cos[..., idx]
        if mean is None or std is None:
            z[..., idx] = (sub - sub.mean(axis=-1, keepdims=True)) / (sub.std(axis=-1, keepdims=True) + 1e-6)
            continue
        m = mean[idx]
        centre = PROMPT_BIAS_WEIGHT * m + (1 - PROMPT_BIAS_WEIGHT) * m.mean()
        z[..., idx] = (sub - centre) / np.sqrt(np.mean(std[idx] ** 2))
    return z


def axis_indices() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, (axis, _, _) in enumerate(prompts()):
        out.setdefault(axis, []).append(i)
    return out


def top_k(cos: np.ndarray, z: np.ndarray, k: int) -> dict[str, list[dict]]:
    """Per axis, the k prompts with the highest z, as {label, cos, z} dicts."""
    labels = [label for _, label, _ in prompts()]
    out = {}
    for axis, idx in axis_indices().items():
        best = sorted(idx, key=lambda i: -z[i])[:k]
        out[axis] = [
            {"label": labels[i], "cos": round(float(cos[i]), 3), "z": round(float(z[i]), 2)} for i in best
        ]
    return out
