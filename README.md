# music-connector

A personal music recommender that runs on your Mac. Full songs are embedded with
[MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) and stored in Postgres + pgvector.
Playlists are built from your taste clusters plus exploration slots for indie, older and random music.

**Status: v1.** Embeds local files, imports Google Takeout history, records feedback by hand and
builds playlists in the terminal. Still to come: the Brave extension (live capture while you listen),
writing playlists to YouTube, and the Thompson-sampling preference model (v2).

## Setup

Requires Docker, [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```sh
docker compose up -d --wait   # Postgres 17 + pgvector on localhost:5433
uv sync
uv run mc bench               # MuQ-MuLan speed on this machine (downloads the model on first run)
```

## Usage

```sh
uv run mc ingest ~/Music/some-album --pool indie       # embed full audio files
uv run mc import-takeout ~/Downloads/watch-history.json
uv run mc songs
uv run mc rate "Somebody Else" like                    # like | dislike | skip | complete | replay
uv run mc similar "Somebody Else"
uv run mc search "dark slow R&B with heavy reverb"     # describe a sound in words
uv run mc taste                                        # your taste clusters
uv run mc recommend --size 25
```

`--pool` tags songs for the playlist's exploration slots: `indie`, `classic` (older decades), `new`.

## Performance (M3 Pro, 18 GB, measured)

| | |
|---|---|
| Model load | ~10–16 s (663M parameters, on the GPU via MPS) |
| Decode (ffmpeg → 24 kHz mono) | ~0.3 s per song |
| Embedding | ~0.14 s per 10 s window, **~70× faster than real time** (a 4-minute song takes ~3.5 s) |
| 5,000 songs × 3.5 min | **~4 h** embedding + ~0.3 h decoding |
| GPU memory | up to ~9 GiB at batch size 16; batch 1 is just as fast and is the default |

Notes:
- **fp16 is unusable on MPS**: autocast gives NaN embeddings, so the model runs in fp32.
- **`transformers` is pinned below 4.50**: `muq` 0.1.0 breaks with newer releases
  (`'EasyDict' object has no attribute '_attn_implementation'`).
- `mc ingest` stops macOS idle sleep while it runs. Closing the lid on battery still
  suspends it; earlier runs took hours because of that, not because of the pipeline.
- Re-uploads of the same recording score ~0.998 cosine. The playlist builder treats anything
  above 0.97 as a duplicate.

### Google Takeout

At takeout.google.com, select only **YouTube and YouTube Music → history**, and set the format to
**JSON**. Plays of songs that haven't been embedded yet are kept. They start counting towards your
taste once their audio is embedded.

## How recommendations work (v1)

1. **Embedding:** every song is cut into 10 s windows at 24 kHz, and each window becomes a 512-d
   MuQ-MuLan vector. The song vector is their re-normalised mean. `embedding_spread` records how
   much the windows differ: low means a consistent song, high means very different sections.
   Window vectors are kept in `song_windows`.
2. **Taste:** each event (like 1.0, replay 0.6, complete 0.3, play 0.1, skip −0.7, dislike −1.0)
   is weighted by recency, with a 90-day half-life. Songs with positive totals are grouped by Ward
   clustering, and each cluster is represented by its weighted medoid (PinnerSage, KDD 2020).
3. **Playlist:** slots are split 55% taste (shared across clusters by importance), 20% indie,
   15% classic and 10% wildcard. Slots a pool can't fill go back to taste. Candidates are the
   pgvector nearest neighbours of each medoid. Their score is similarity to the medoid, with a
   penalty for closeness to disliked songs. MMR then selects for diversity, with at most 2 songs
   per artist. Finally the playlist is ordered so each song leads to its closest neighbour.

Settings are in `src/music_connector/config.py`.

## Licences

MuQ-MuLan weights are licensed CC BY-NC 4.0 (non-commercial). This project is for personal use.
