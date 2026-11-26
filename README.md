# music-connector

A personal music recommender that runs on your Mac. Full songs are embedded with
[MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) and stored in Postgres + pgvector.
Playlists are built from your taste clusters plus exploration slots for indie, older and random music.

**Status: v1 + live capture.** Embeds local files, and also songs while you listen in Brave
(YouTube Music / YouTube). Imports Google Takeout history and builds playlists in the terminal.
Still to come: writing playlists to YouTube, and the Thompson-sampling preference model (v2).

## Live capture in Brave

1. Start the server: `uv run mc serve` (listens on `ws://127.0.0.1:8765`, loads the model once).
2. Go to `brave://extensions`, turn on **Developer mode**, click **Load unpacked**, and choose
   the `extension/` folder. Pin the "Music Connector" icon.
3. Open music.youtube.com (or a music video on youtube.com) and **click the icon**. The badge
   shows `ON` when audio is streaming to the server, and `…` while it waits for `mc serve`.
   Click again to stop. You need to click once per tab after each browser restart, because
   Chrome only allows tab capture after an explicit click.

What happens while you listen:

- The tab's audio is cut into 10 s windows and embedded as you listen. No audio is written to
  disk. When a track ends, the song vector, its window vectors and the fraction of the song you
  heard are saved. If you listen again and hear more of it, the vector is replaced.
- Listen events: **complete** (≥ 80% heard or reached the end; a repeat also logs **replay**),
  **skip** (moved on before 30 s), **play** (anything in between). 👍/👎 clicks in the player
  are logged as **like**/**dislike**.
- Pauses, buffering, seeks and ads are excluded, and so are silent windows. YouTube's volume
  slider and loudness normalisation are undone, so live and file embeddings match: a live
  capture at 50% volume scored 0.9999 cosine against the same file.
- youtube.com is captured only when you click the icon on that tab, because most of YouTube
  isn't music.

`scripts/simulate_extension.py` replays local files through the same protocol to test the
server without a browser.

Limits: only listening in Brave on this Mac is captured, not on your phone. The like-button
selectors depend on YouTube's current page markup and may need updating when it changes. Up to
~0.1 s of the previous song can leak into the first window of the next.

## Setup

Requires macOS or Linux with Docker, [uv](https://docs.astral.sh/uv/) and ffmpeg
(`brew install uv ffmpeg`). uv installs Python 3.12 itself if needed. The model runs on Apple
Silicon (MPS), CUDA or CPU; the timings below are for an M3 Pro.

```sh
git clone <repo-url> music-connector && cd music-connector
cp .env.example .env          # optional: change MC_DB_PORT if 5433 is taken
docker compose up -d --wait   # Postgres 17 + pgvector on localhost:5433; creates the schema
uv sync
uv run mc songs               # checks the database connection: "Embedded songs (0 of 0)"
uv run mc bench               # MuQ-MuLan speed on this machine (downloads ~2.5 GB of weights once)
```

Every setting has a default, so `.env` is optional. docker compose reads it automatically; to
point `mc` at a different port or database, export the variables or run
`uv run --env-file .env mc ...`. See `.env.example` for the list.

Tests need neither the model nor the database:

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
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
uv run mc serve                                        # live capture from the Brave extension
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

## Privacy

Everything stays on your machine. Audio is decoded in memory and never written to disk; your
listening history, ratings and embeddings live only in the local Postgres volume; the server
binds to 127.0.0.1 and only accepts connections from the extension or local scripts. Nothing is
sent anywhere except the one-time model download from Hugging Face.

This repository contains no listening data: no Takeout exports, no event history, no audio. The
`.gitignore` excludes `.env`, `watch-history*.json`, Takeout folders, audio files and local data
directories so they can't be committed by accident.

## Licences

The code is MIT-licensed (see `LICENSE`). The MuQ-MuLan weights it downloads are licensed
**CC BY-NC 4.0 (non-commercial)** and are not distributed here; see `NOTICE`. Anything that runs
the model, including this project, is limited to non-commercial use.
