"""Import YouTube / YouTube Music watch history from a Google Takeout `watch-history.json`."""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import db

TITLE_PREFIX = "Watched "


def _video_id(url: str) -> str | None:
    return (parse_qs(urlparse(url).query).get("v") or [None])[0]


def _artist(entry: dict) -> str | None:
    subs = entry.get("subtitles") or []
    if not subs:
        return None
    return subs[0].get("name", "").removesuffix(" - Topic") or None


def _is_music(entry: dict, include_youtube: bool) -> bool:
    if entry.get("header") == "YouTube Music":
        return True
    if not include_youtube:
        return False
    # Plain-YouTube history mixes music with everything else; auto-generated "- Topic"
    # channels and VEVO only ever host music.
    channel = ((entry.get("subtitles") or [{}])[0]).get("name", "")
    return channel.endswith(" - Topic") or "VEVO" in channel


def import_history(path: Path, include_youtube: bool = False) -> tuple[int, int]:
    """Returns (songs, plays) imported. Re-running replaces earlier Takeout plays."""
    entries = json.loads(path.read_text())
    plays = []
    for e in entries:
        url = e.get("titleUrl")
        vid = _video_id(url) if url else None
        if not vid or not _is_music(e, include_youtube):
            continue
        title = e.get("title", "").removeprefix(TITLE_PREFIX)
        at = datetime.fromisoformat(e["time"].replace("Z", "+00:00"))
        plays.append((vid, title, _artist(e), at))

    with db.connect() as conn:
        conn.execute("DELETE FROM events WHERE source = 'takeout'")
        ids = {}
        for vid, title, artist, at in plays:
            if vid not in ids:
                ids[vid] = db.upsert_song(conn, title=title, artist=artist, youtube_id=vid, source="takeout")
            db.add_event(conn, ids[vid], "play", at=at, source="takeout")
        conn.commit()
    return len(ids), len(plays)


def top_played(conn, limit: int = 20) -> list[tuple]:
    return conn.execute(
        """SELECT s.id, s.title, s.artist, count(*) AS plays, s.embedding IS NOT NULL
           FROM events e JOIN songs s ON s.id = e.song_id
           WHERE e.kind = 'play' GROUP BY s.id ORDER BY plays DESC LIMIT %s""",
        (limit,),
    ).fetchall()
