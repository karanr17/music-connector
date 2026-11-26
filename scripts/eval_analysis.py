"""Quality check for `mc analyze`: segmentation against human annotations, descriptors against
source tags, tempo against the uploader's BPM. Produces the README table.

    uv run python scripts/eval_analysis.py features ~/music-connector-eval-audio
    uv run python scripts/eval_analysis.py calibrate ~/music-connector-eval-audio/calibration
    uv run python scripts/eval_analysis.py report ~/music-connector-eval-audio

`features` loads the model once and caches everything model-dependent under `<root>/.features`,
so `calibrate` and `report` (and segmentation tuning) run without the model.

Expected layout of the eval folder (see README "Analysis quality"):
  salami/sources.json + salami_<id>.mp3 + salami_<id>_ann{1,2}.txt   SALAMI Internet Archive subset
  ccmixter/sources.json + ccm_<id>.mp3                                ccMixter, with uploader tags
  extra/*.mp3 + extra/*.annotation.txt                                anything else, hand-annotated
"""

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from music_connector.analysis import descriptors as desc
from music_connector.analysis.analyzer import HOP_S, WINDOW_S
from music_connector.analysis.segments import SegmentConfig, segment
from music_connector.audio import AUDIO_EXTS, decode, hop_windows
from music_connector.config import SAMPLE_RATE

PACKAGED_CALIBRATION = Path(__file__).parents[1] / "src/music_connector/analysis/calibration.json"


def audio_files(root: Path) -> list[Path]:
    """Audio under root, excluding the calibration corpus when it sits inside the eval folder."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTS
        and not {".features", "calibration"} & set(p.relative_to(root).parts)
    )


def feature_path(root: Path, audio: Path) -> Path:
    return root / ".features" / (audio.relative_to(root).with_suffix(".npz").as_posix().replace("/", "__"))


def cmd_features(root: Path) -> None:
    from music_connector.analysis import dsp
    from music_connector.embedder import Embedder

    files = [f for f in audio_files(root) if not feature_path(root, f).exists()]
    if not files:
        print("all features cached")
        return
    emb = Embedder()
    (root / ".features").mkdir(exist_ok=True)
    text_path = root / ".features" / f"text-{desc.vocabulary_id()}.npy"
    if not text_path.exists():
        np.save(text_path, emb.embed_text([p for _, _, p in desc.prompts()]))
    for f in files:
        t0 = time.perf_counter()
        wav = decode(f)
        if len(wav) == 0:
            print("skip (no audio)", f)
            continue
        t1 = time.perf_counter()
        d = dsp.analyse(wav, SAMPLE_RATE)
        t2 = time.perf_counter()
        wins, starts = hop_windows(wav, SAMPLE_RATE, WINDOW_S, HOP_S)
        vecs = emb.embed_windows(wins)
        emb.sync()
        t3 = time.perf_counter()
        np.savez_compressed(
            feature_path(root, f),
            duration=len(wav) / SAMPLE_RATE,
            bpm=d.bpm,
            beat_confidence=d.beat_confidence,
            beats=d.beats,
            downbeats=d.downbeats,
            beat_features=d.beat_features,
            rms=d.energy.rms,
            rms_rate=d.energy.frame_rate,
            win_starts=np.asarray(starts),
            win_vecs=vecs,
            timing=np.array([t1 - t0, t2 - t1, t3 - t2]),
        )
        print(f"{f.name}: decode {t1 - t0:.1f}s dsp {t2 - t1:.1f}s model {t3 - t2:.1f}s", flush=True)


def load(root: Path, audio: Path) -> dict:
    with np.load(feature_path(root, audio)) as z:
        return {k: z[k] for k in z.files}


def text_vectors(root: Path) -> np.ndarray:
    return np.load(root / ".features" / f"text-{desc.vocabulary_id()}.npy")


def song_vector(feats: dict) -> np.ndarray:
    v = feats["win_vecs"].mean(axis=0)
    return v / np.linalg.norm(v)


def cmd_calibrate(root: Path, out: Path, name: str) -> None:
    text = text_vectors(root)
    song_cos, win_cos = [], []
    for f in audio_files(root):
        feats = load(root, f)
        song_cos.append(song_vector(feats) @ text.T)
        win_cos.append(feats["win_vecs"] @ text.T)
    cal = desc.Calibration.from_cosines(name, np.array(song_cos), np.vstack(win_cos))
    out.write_text(json.dumps(cal.to_dict(), indent=1) + "\n")
    print(f"wrote {out}: {cal.songs} songs, {cal.windows} windows")


# --- ground truth -------------------------------------------------------------------------------

GENRE_TAGS = {
    "hip_hop": "hip hop",
    "rock": "rock",
    "jazz": "jazz",
    "nu_jazz": "jazz",
    "folk": "folk",
    "reggae": "reggae",
    "dub": "reggae",
    "ambient": "ambient",
    "classical": "classical",
    "funk": "funk",
    "funky": "funk",
    "blues": "blues",
    "country": "country",
    "pop": "pop",
    "dubstep": "dubstep",
    "latin": "latin",
    "reggaeton": "latin",
    "bossa_nova": "latin",
    "techno": "techno",
    "house": "house",
    "dance": "electronic dance music",
    "electro": "electronic dance music",
    "electronica": "electronic dance music",
    "trip_hop": "trip hop",
    "downtempo": "trip hop",
    "dnb": "drum and bass",
    "orchestral": "orchestral film score",
    "music_for_film": "orchestral film score",
    "soul": "r&b and soul",
    "nu_soul": "r&b and soul",
}
INSTRUMENT_TAGS = {
    "guitar": {"acoustic guitar", "electric guitar", "distorted guitar"},
    "acoustic": {"acoustic guitar"},
    "electric_guitar": {"electric guitar", "distorted guitar"},
    "telecaster": {"electric guitar"},
    "piano": {"piano"},
    "synthesizer": {"synthesizer"},
    "violin": {"violin and strings"},
    "cello": {"cello", "violin and strings"},
    "drums": {"drum kit", "drum machine"},
    "bass": {"bass guitar", "808 bass"},
    "rhodes": {"electric piano"},
    "electric_piano": {"electric piano"},
    "wooden_flute": {"flute"},
    "flute": {"flute"},
}
VOCAL_TAGS = {
    "male_vocals": "male vocals",
    "female_vocals": "female vocals",
    "instrumental": "instrumental, no vocals",
    "spoken_word": "spoken word",
    "choir": "choir",
}


def tag_truth(tags: str) -> dict[str, set[str]]:
    tags_ = set(tags.split(","))
    truth = {
        "genre": {GENRE_TAGS[t] for t in tags_ if t in GENRE_TAGS},
        "instruments": set().union(*[INSTRUMENT_TAGS[t] for t in tags_ if t in INSTRUMENT_TAGS]),
        "vocals": {VOCAL_TAGS[t] for t in tags_ if t in VOCAL_TAGS},
    }
    return {k: v for k, v in truth.items() if v}


def read_annotation(path: Path) -> list[tuple[float, str]]:
    """SALAMI-style 'time<TAB>label' lines; the last line ('End') closes the final segment."""
    out = []
    for line in path.read_text().splitlines():
        parts = re.split(r"\s+", line.strip(), maxsplit=1)
        if len(parts) == 2:
            out.append((float(parts[0]), parts[1].strip()))
    return out


def ann_segments(ann: list[tuple[float, str]]) -> list[tuple[float, float, str]]:
    return [(a, b, lab.rstrip("'")) for (a, lab), (b, _) in zip(ann[:-1], ann[1:], strict=True)]


# --- metrics -----------------------------------------------------------------------------------


def hit_rate(ref: list[float], est: list[float], window: float = 3.0) -> tuple[float, float, float]:
    """Boundary precision, recall and F at ±window seconds, one-to-one matching."""
    ref, est = sorted(ref), sorted(est)
    pairs = sorted(
        (abs(r - e), i, j) for i, r in enumerate(ref) for j, e in enumerate(est) if abs(r - e) <= window
    )
    used_r, used_e, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
            hits += 1
    p = hits / len(est) if est else 0.0
    r = hits / len(ref) if ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def pairwise_f(ref: list[tuple[float, float, str]], est: list[tuple[float, float, str]], end: float) -> float:
    """Pairwise frame clustering F (0.5 s frames): do two frames share a label in both?"""
    t = np.arange(0.0, end, 0.5)

    def frame_labels(segs):
        out = np.full(len(t), "", dtype=object)
        for a, b, lab in segs:
            out[(t >= a) & (t < b)] = lab
        return out

    r, e = frame_labels(ref), frame_labels(est)
    same_r = r[:, None] == r[None, :]
    same_e = e[:, None] == e[None, :]
    iu = np.triu_indices(len(t), 1)
    sr, se = same_r[iu], same_e[iu]
    both = np.sum(sr & se)
    p = both / max(se.sum(), 1)
    rc = both / max(sr.sum(), 1)
    return float(2 * p * rc / (p + rc)) if p + rc else 0.0


def inner(bounds: list[float], end: float) -> list[float]:
    return [b for b in bounds if 0.5 < b < end - 0.5]


def run_segments(feats: dict, cfg: SegmentConfig):
    return segment(
        duration=float(feats["duration"]),
        beats=feats["beats"],
        downbeats=feats["downbeats"],
        beat_features=feats["beat_features"],
        window_starts=feats["win_starts"],
        window_vecs=feats["win_vecs"],
        window_s=WINDOW_S,
        rms=feats["rms"],
        rms_rate=float(feats["rms_rate"]),
        cfg=cfg,
    )


def annotated_tracks(root: Path) -> list[tuple[str, Path, list[Path]]]:
    out = []
    for f in audio_files(root):
        anns = sorted(f.parent.glob(f"{f.stem}_ann*.txt")) or sorted(
            f.parent.glob(f"{f.stem}.annotation.txt")
        )
        if anns:
            out.append((f.stem, f, anns))
    return out


def seg_scores(root: Path, cfg: SegmentConfig) -> list[dict]:
    rows = []
    for name, f, anns in annotated_tracks(root):
        feats = load(root, f)
        dur = float(feats["duration"])
        segs = run_segments(feats, cfg)
        est_b = inner([s.start for s in segs], dur)
        est_l = [(s.start, s.end, s.letter) for s in segs]
        hr, pw, agree = [], [], []
        refs = [ann_segments(read_annotation(a)) for a in anns]
        for ref in refs:
            end = min(dur, ref[-1][1])
            ref_b = inner([a for a, _, _ in ref], end)
            hr.append(hit_rate(ref_b, [b for b in est_b if b < end])[2])
            pw.append(pairwise_f(ref, est_l, end))
        if len(refs) == 2:
            end = min(refs[0][-1][1], refs[1][-1][1])
            agree = [
                hit_rate(inner([a for a, _, _ in refs[0]], end), inner([a for a, _, _ in refs[1]], end))[2],
                pairwise_f(refs[0], refs[1], end),
            ]
        uniform = list(np.arange(20.0, dur - 4, 20.0))
        rows.append(
            {
                "name": name,
                "hr3": max(hr),
                "pwf": max(pw),
                "agree": agree,
                "uniform_hr3": max(
                    hit_rate(inner([a for a, _, _ in ref], min(dur, ref[-1][1])), uniform)[2] for ref in refs
                ),
                "n_est": len(segs),
                "n_ref": np.mean([len(r) for r in refs]),
                "labels": "".join(s.letter for s in segs),
                "ref_labels": " / ".join("".join(lab[:1] for _, _, lab in r) for r in refs),
            }
        )
    return rows


def descriptor_scores(root: Path, cal: desc.Calibration | None) -> list[dict]:
    text = text_vectors(root)
    labels = [lab for _, lab, _ in desc.prompts()]
    axes = desc.axis_indices()
    tracks = []
    if (root / "ccmixter/sources.json").exists():
        for s in json.loads((root / "ccmixter/sources.json").read_text()):
            tracks.append((root / "ccmixter" / s["file"], tag_truth(s["tags"]), s))
    live = {"texture": {"live concert recording"}}
    if (root / "salami/sources.json").exists():
        for s in json.loads((root / "salami/sources.json").read_text()):
            if s["ok"]:
                tracks.append((root / "salami" / s["file"], live, s))
    tracks += [(p, live, {}) for p in sorted((root / "lma").glob("*.mp3"))]
    rows = []
    for path, truth, src in tracks:
        if not feature_path(root, path).exists():
            continue
        feats = load(root, path)
        cos = song_vector(feats) @ text.T
        z_cal = desc.zscores(cos, cal.song_mean, cal.song_std) if cal else None
        row = {"name": path.stem, "truth": truth, "src": src}
        for mode, score in (("raw", cos), ("cal", z_cal)):
            if score is None:
                continue
            for axis, want in truth.items():
                ranked = [labels[i] for i in sorted(axes[axis], key=lambda i: -score[i])]
                row[(mode, axis)] = (ranked[0] in want, bool(set(ranked[:3]) & want), ranked[:3])
        rows.append(row)
    return rows


def tempo_scores(root: Path) -> list[dict]:
    rows = []
    for s in json.loads((root / "ccmixter/sources.json").read_text()):
        path = root / "ccmixter" / s["file"]
        if not s.get("bpm") or not feature_path(root, path).exists():
            continue
        est, ref = float(load(root, path)["bpm"]), float(s["bpm"])
        ok = abs(est - ref) / ref <= 0.04
        octave = any(abs(est * k - ref) / ref <= 0.04 for k in (0.5, 2.0, 1.5, 2 / 3))
        rows.append({"name": path.stem, "est": est, "ref": ref, "ok": ok, "octave": octave and not ok})
    return rows


def cmd_report(root: Path, cfg: SegmentConfig) -> None:
    cal = desc.load_calibration(PACKAGED_CALIBRATION)
    seg = seg_scores(root, cfg)
    print("## Segmentation (HR3F = boundary F at ±3 s; PWF = pairwise label F)\n")
    print("| track | est / ref segments | HR3F | PWF | annotators agree (HR3F / PWF) | uniform 20 s HR3F |")
    print("|---|---|---|---|---|---|")
    for r in seg:
        ag = f"{r['agree'][0]:.2f} / {r['agree'][1]:.2f}" if r["agree"] else "–"
        print(
            f"| {r['name']} | {r['n_est']} / {r['n_ref']:.0f} | {r['hr3']:.2f} | {r['pwf']:.2f} | {ag} "
            f"| {r['uniform_hr3']:.2f} |  `{r['labels']}` vs `{r['ref_labels']}`"
        )
    m = {k: np.mean([r[k] for r in seg]) for k in ("hr3", "pwf", "uniform_hr3")}
    ag = [r["agree"] for r in seg if r["agree"]]
    print(
        f"\nmean HR3F {m['hr3']:.2f}  PWF {m['pwf']:.2f}  uniform-20s HR3F {m['uniform_hr3']:.2f}  "
        f"annotator agreement HR3F {np.mean([a[0] for a in ag]):.2f} PWF {np.mean([a[1] for a in ag]):.2f}\n"
    )

    rows = descriptor_scores(root, cal)
    print(f"## Descriptors (calibration: {cal.name if cal else 'none'})\n")
    for axis in ("genre", "instruments", "vocals", "texture"):
        for mode in ("raw", "cal"):
            hits = [r[(mode, axis)] for r in rows if (mode, axis) in r]
            if hits:
                print(
                    f"{axis:12} {mode}: top-1 {sum(h[0] for h in hits)}/{len(hits)}  "
                    f"top-3 {sum(h[1] for h in hits)}/{len(hits)}"
                )
    print()
    for r in rows:
        bits = []
        for axis, want in r["truth"].items():
            if ("cal", axis) in r:
                h = r[("cal", axis)]
                bits.append(f"{axis}: {'✓' if h[0] else ('~' if h[1] else '✗')} {h[2]} want {sorted(want)}")
        print(r["name"], "|", " ; ".join(bits))
    print("\n## Tempo vs uploader BPM\n")
    for t in tempo_scores(root):
        flag = "ok" if t["ok"] else ("octave" if t["octave"] else "MISS")
        print(f"{t['name']}: est {t['est']:.1f} tag {t['ref']:.0f} {flag}")


def cmd_tune(root: Path) -> None:
    base = SegmentConfig()
    grid = []
    for thr in (0.5, 1.0, 1.5, 2.0):
        for same in (0.90, 0.92, 0.94, 0.95, 0.96):
            for rate in (10.0, 14.0, 18.0):
                for fw in (0.3, 0.5, 0.7):
                    for merge in (True, False):
                        cfg = replace(
                            base,
                            threshold_mads=thr,
                            same_cosine=same,
                            max_rate_s=rate,
                            feature_weight=fw,
                            merge_adjacent_same=merge,
                        )
                        s = seg_scores(root, cfg)
                        grid.append(
                            (
                                np.mean([r["hr3"] for r in s]),
                                np.mean([r["pwf"] for r in s]),
                                thr,
                                same,
                                rate,
                                fw,
                                merge,
                            )
                        )
    for g in sorted(grid, key=lambda g: -(g[0] + g[1]))[:15]:
        print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "calibrate", "report", "tune"])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=PACKAGED_CALIBRATION)
    ap.add_argument("--name", default="cc-reference")
    a = ap.parse_args()
    root = a.root.expanduser()
    if a.command == "features":
        cmd_features(root)
    elif a.command == "calibrate":
        cmd_calibrate(root, a.out, a.name)
    elif a.command == "report":
        cmd_report(root, SegmentConfig())
    elif a.command == "tune":
        cmd_tune(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
