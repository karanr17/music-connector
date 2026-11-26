"""Signal-level analysis with librosa: tempo, beats, downbeats, onsets, energy envelopes, and the
beat-synchronous features segmentation uses. MuLan is not involved: it can't hear tempo or
energy reliably (see the README quality check)."""

from dataclasses import dataclass

import librosa
import numpy as np

HOP = 512
ENERGY_RATE = 10.0
ENERGY_RANGE_DB = 48.0
BANDS = {"low": (20.0, 250.0), "mid": (250.0, 4000.0), "high": (4000.0, 12000.0)}
BEATS_PER_BAR = 4


@dataclass
class EnergyCurves:
    frame_rate: float
    loudness_db: float
    rms: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray


@dataclass
class Dsp:
    bpm: float
    beat_confidence: float
    beats: np.ndarray
    downbeats: np.ndarray
    onsets: np.ndarray
    energy: EnergyCurves
    # [n_beats, d] chroma + timbre, one row per beat (row i covers beats[i] .. beats[i + 1]).
    beat_features: np.ndarray


def _scale_db(db: np.ndarray) -> np.ndarray:
    ref = np.percentile(db, 99)
    return np.clip((db - (ref - ENERGY_RANGE_DB)) / ENERGY_RANGE_DB, 0.0, 1.0)


def energy_curves(wav: np.ndarray, sr: int) -> EnergyCurves:
    hop = int(round(sr / ENERGY_RATE))
    power = np.abs(librosa.stft(wav, n_fft=4096, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    bands = {
        name: _scale_db(10 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum(axis=0) + 1e-10))
        for name, (lo, hi) in BANDS.items()
    }
    rms_db = 20 * np.log10(librosa.feature.rms(y=wav, frame_length=4096, hop_length=hop)[0] + 1e-6)
    n = min(len(rms_db), *(len(b) for b in bands.values()))
    return EnergyCurves(
        frame_rate=sr / hop,
        loudness_db=float(np.percentile(rms_db, 95)),
        rms=_scale_db(rms_db)[:n],
        **{k: v[:n] for k, v in bands.items()},
    )


def beat_confidence(onset_env: np.ndarray, beat_frames: np.ndarray) -> float:
    """How periodic the onsets are at the tracked beat period: the normalised autocorrelation
    peak there. (Onset strength *at* the beats is useless for this: the tracker puts beats on
    onsets, so it is always high.) A weak signal: on the quality-check set every correct tempo
    scored >= 0.45, but only two of the five errors scored below that."""
    if len(beat_frames) < 4:
        return 0.0
    env = onset_env - onset_env.mean()
    ac = librosa.autocorrelate(env)
    if ac[0] <= 0:
        return 0.0
    lag = int(round(np.median(np.diff(beat_frames))))
    return float(np.clip(ac[max(lag - 2, 1) : lag + 3].max() / ac[0], 0.0, 1.0))


def downbeat_phase(
    low_onset: np.ndarray, chroma_change: np.ndarray, beats_per_bar: int = BEATS_PER_BAR
) -> int:
    """Pick which beat in each bar is the downbeat: the phase where bass onsets and harmonic
    changes land most often. A heuristic for 4/4 pop/rock; it has no notion of other meters."""
    if len(low_onset) < 2 * beats_per_bar:
        return 0

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    score = z(low_onset) + z(chroma_change)
    return int(np.argmax([score[p::beats_per_bar].mean() for p in range(beats_per_bar)]))


def analyse(wav: np.ndarray, sr: int) -> Dsp:
    spec = np.abs(librosa.stft(wav, n_fft=2048, hop_length=HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)

    confidence = beat_confidence(onset_env, beat_frames)

    chroma = librosa.feature.chroma_stft(S=spec, sr=sr)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=14)[1:]
    low_bins = librosa.mel_frequencies(n_mels=128, fmax=sr / 2) < 200
    low_onset = librosa.onset.onset_strength(S=log_mel[low_bins], sr=sr, hop_length=HOP)

    if len(beat_frames) >= 2:
        sync_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median, pad=False)
        sync_mfcc = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean, pad=False)
        unit = sync_chroma / (np.linalg.norm(sync_chroma, axis=0, keepdims=True) + 1e-9)
        # Change entering beat i, compared with the previous beat's span.
        change = np.r_[0.0, 1.0 - np.sum(unit[:, 1:] * unit[:, :-1], axis=0)]
        n = len(beat_frames) - 1
        lo = np.array([low_onset[max(f - 2, 0) : f + 3].max() for f in beat_frames[:n]])
        phase = downbeat_phase(lo, change[:n])
        mfcc_z = (sync_mfcc - sync_mfcc.mean(axis=1, keepdims=True)) / (
            sync_mfcc.std(axis=1, keepdims=True) + 1e-9
        )
        beat_features = np.vstack([unit, mfcc_z / np.sqrt(mfcc_z.shape[0])]).T
    else:
        phase = 0
        beat_features = np.zeros((0, 25))

    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time")
    return Dsp(
        bpm=bpm,
        beat_confidence=confidence,
        beats=beats,
        downbeats=beats[phase::BEATS_PER_BAR],
        onsets=np.asarray(onsets),
        energy=energy_curves(wav, sr),
        beat_features=beat_features,
    )
