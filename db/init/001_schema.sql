CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE songs (
    id               bigserial PRIMARY KEY,
    youtube_id       text UNIQUE,
    local_path       text UNIQUE,
    title            text NOT NULL,
    artist           text,
    release_year     int,
    -- Exploration pools used by playlist slot quotas: 'indie', 'classic', 'new'.
    pools            text[] NOT NULL DEFAULT '{}',
    source           text NOT NULL,
    duration_s       real,
    -- Mean of the 10 s window vectors, re-normalised to unit length.
    embedding        vector(512),
    -- Mean cosine distance of windows from the song vector: low = consistent song,
    -- high = sections that sound very different (e.g. one strong hook).
    embedding_spread real,
    -- Fraction of the song the embedding was computed from; < 1 for partial listens.
    coverage         real,
    embedding_model  text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX songs_embedding_hnsw ON songs USING hnsw (embedding vector_cosine_ops);

CREATE TABLE song_windows (
    song_id   bigint NOT NULL REFERENCES songs ON DELETE CASCADE,
    start_s   real   NOT NULL,
    embedding vector(512) NOT NULL,
    PRIMARY KEY (song_id, start_s)
);

CREATE TABLE events (
    id      bigserial PRIMARY KEY,
    song_id bigint NOT NULL REFERENCES songs ON DELETE CASCADE,
    kind    text NOT NULL CHECK (kind IN ('like', 'dislike', 'skip', 'complete', 'replay', 'play')),
    at      timestamptz NOT NULL DEFAULT now(),
    source  text NOT NULL DEFAULT 'manual'
);

CREATE INDEX events_song_idx ON events (song_id);
