from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

EVENT_KINDS = ("like", "dislike", "skip", "complete", "replay", "play")


def _find_song(conn, query: str) -> tuple[int, str, str | None]:
    if query.isdigit():
        row = conn.execute("SELECT id, title, artist FROM songs WHERE id = %s", (int(query),)).fetchone()
        if row:
            return row
    rows = conn.execute(
        """SELECT id, title, artist FROM songs
           WHERE title ILIKE %s OR coalesce(artist, '') || ' ' || title ILIKE %s
           ORDER BY embedding IS NULL, id LIMIT 6""",
        (f"%{query}%", f"%{query}%"),
    ).fetchall()
    if not rows:
        raise typer.BadParameter(f"no song matches {query!r}")
    if len(rows) > 1:
        options = "\n".join(f"  {r[0]}: {r[2] or '?'} – {r[1]}" for r in rows)
        raise typer.BadParameter(f"{query!r} is ambiguous; use an id:\n{options}")
    return rows[0]


def _song_table(title: str, rows, extra: str | None = None) -> Table:
    t = Table(title=title)
    t.add_column("id", justify="right")
    t.add_column("artist")
    t.add_column("title")
    if extra:
        t.add_column(extra, justify="right")
    for r in rows:
        t.add_row(str(r[0]), r[2] or "", r[1], *([f"{r[3]:.3f}"] if extra else []))
    return t


@app.command()
def bench(
    files: Annotated[
        list[Path] | None, typer.Argument(help="Audio files; synthetic audio if omitted")
    ] = None,
    song_seconds: float = 210.0,
    batch_sizes: str = "1,4,8,16",
    repeats: int = 3,
    project_songs: int = 5000,
    device: str | None = None,
):
    """Benchmark MuQ-MuLan embedding speed on this machine."""
    from .bench import run

    run(files or [], song_seconds, [int(b) for b in batch_sizes.split(",")], repeats, project_songs, device)


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="Audio files or folders (searched recursively)")],
    pool: Annotated[list[str] | None, typer.Option(help="Exploration pool tag: indie, classic, new")] = None,
    force: bool = False,
):
    """Embed full local audio files into the database."""
    from .ingest import ingest_files

    try:
        n = ingest_files(paths, pool or [], force)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"Embedded {n} new song(s).")


@app.command("import-takeout")
def import_takeout(
    path: Annotated[Path, typer.Argument(help="Takeout watch-history.json")],
    include_youtube: Annotated[
        bool, typer.Option(help="Also take plain-YouTube plays from '- Topic' / VEVO channels")
    ] = False,
):
    """Seed listening history from Google Takeout."""
    from . import db
    from .takeout import import_history, top_played

    songs, plays = import_history(path, include_youtube)
    console.print(f"Imported {plays} plays of {songs} songs.")
    with db.connect() as conn:
        t = Table(title="Most played")
        for col in ("id", "artist", "title", "plays", "embedded"):
            t.add_column(col)
        for r in top_played(conn):
            t.add_row(str(r[0]), r[2] or "", r[1], str(r[3]), "yes" if r[4] else "–")
        console.print(t)


@app.command()
def rate(
    song: Annotated[str, typer.Argument(help="Song id or part of its title")],
    kind: Annotated[str, typer.Argument(help=" | ".join(EVENT_KINDS))] = "like",
):
    """Record feedback for a song."""
    from . import db

    if kind not in EVENT_KINDS:
        raise typer.BadParameter(f"kind must be one of {EVENT_KINDS}")
    with db.connect() as conn:
        song_id, title, artist = _find_song(conn, song)
        db.add_event(conn, song_id, kind)
        conn.commit()
    console.print(f"{kind}: {artist or '?'} – {title}")


@app.command()
def songs(limit: int = 50):
    """List songs in the database."""
    from . import db

    with db.connect() as conn:
        rows = conn.execute(
            """SELECT id, title, artist, embedding_spread FROM songs
               WHERE embedding IS NOT NULL ORDER BY id LIMIT %s""",
            (limit,),
        ).fetchall()
        total, embedded = conn.execute("SELECT count(*), count(embedding) FROM songs").fetchone()
    console.print(_song_table(f"Embedded songs ({embedded} of {total})", rows, "spread"))


@app.command()
def similar(song: str, limit: int = 10):
    """Songs that sound closest to a given song."""
    from . import db

    with db.connect() as conn:
        song_id, title, artist = _find_song(conn, song)
        rows = conn.execute(
            """SELECT b.id, b.title, b.artist, 1 - (b.embedding <=> a.embedding)
               FROM songs a JOIN songs b ON b.id <> a.id AND b.embedding IS NOT NULL
               WHERE a.id = %s AND a.embedding IS NOT NULL
               ORDER BY b.embedding <=> a.embedding LIMIT %s""",
            (song_id, limit),
        ).fetchall()
    console.print(_song_table(f"Closest to {artist or '?'} – {title}", rows, "cosine"))


@app.command()
def search(text: str, limit: int = 10):
    """Find songs by describing how they sound (MuQ-MuLan text→audio)."""
    from . import db
    from .embedder import Embedder

    vec = Embedder().embed_text([text])[0]
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT id, title, artist, 1 - (embedding <=> %s) FROM songs
               WHERE embedding IS NOT NULL ORDER BY embedding <=> %s LIMIT %s""",
            (vec, vec, limit),
        ).fetchall()
    console.print(_song_table(f"“{text}”", rows, "match"))


@app.command("taste")
def show_taste(cluster_distance: float | None = None):
    """Show your taste clusters (PinnerSage-style)."""
    from . import db, taste
    from .config import RecommendConfig

    cfg = RecommendConfig(**({"cluster_distance": cluster_distance} if cluster_distance else {}))
    with db.connect() as conn:
        clusters, dislikes = taste.build(conn, cfg)
        if not clusters:
            console.print("No embedded songs with positive feedback yet.")
            return
        t = Table(title=f"{len(clusters)} taste cluster(s) · {len(dislikes)} disliked song(s)")
        for col in ("#", "medoid", "importance", "songs"):
            t.add_column(col)
        for i, k in enumerate(clusters, 1):
            t.add_row(str(i), k.medoid_title, f"{k.importance:.2f}", str(len(k.member_ids)))
        console.print(t)


@app.command()
def recommend(size: int = 25):
    """Build a playlist from your taste clusters plus exploration slots."""
    from . import db
    from .config import RecommendConfig
    from .recommend import build_playlist

    with db.connect() as conn:
        try:
            picks, clusters = build_playlist(conn, RecommendConfig(size=size))
        except LookupError as e:
            console.print(f"[yellow]{e}[/]")
            raise typer.Exit(1) from None
    t = Table(title=f"Playlist · {len(picks)} songs from {len(clusters)} taste cluster(s)")
    for col in ("#", "id", "artist", "title", "slot", "score", "near"):
        t.add_column(col)
    for i, p in enumerate(picks, 1):
        c = p.candidate
        t.add_row(str(i), str(c.id), c.artist or "", c.title, p.slot, f"{c.score:.3f}", c.near)
    console.print(t)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765):
    """Receive live audio from the Brave extension and embed songs as you listen."""
    import asyncio
    import logging

    from .server import serve_forever

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    try:
        asyncio.run(serve_forever(host, port))
    except KeyboardInterrupt:
        pass


def main():
    app()
