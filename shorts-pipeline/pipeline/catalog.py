"""Pull channel metadata (no video download) and rank videos by virality."""
from __future__ import annotations

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import log, read_json, work_path, write_json


def _ydl(opts: dict):
    import yt_dlp

    base = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def fetch_catalog(cfg: dict, limit: int | None = None, enrich_top: int | None = None) -> list[dict]:
    """Flat listing of the channel's /videos tab, then full metadata for the top candidates.

    The flat listing is cheap (one request) and gives id/title/duration/view_count.
    Likes, comments and upload date need a per-video request, so we only enrich the
    most-viewed subset (3x top_n by default) that could plausibly make the cut.
    """
    limit = limit or cfg["channel"]["catalog_limit"]
    url = cfg["channel"]["url"].rstrip("/") + "/videos"
    log.info("Fetching flat catalog: %s (limit %d)", url, limit)
    with _ydl({"extract_flat": "in_playlist", "playlistend": limit}) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info or {}).get("entries") or [] if e and e.get("id")]
    log.info("Flat catalog: %d entries", len(entries))

    min_dur = cfg["ranking"]["min_duration_sec"]
    videos = []
    for e in entries:
        dur = e.get("duration") or 0
        if dur and dur < min_dur:
            continue
        videos.append(
            {
                "id": e["id"],
                "title": e.get("title") or "",
                "url": f"https://www.youtube.com/watch?v={e['id']}",
                "duration": dur,
                "view_count": e.get("view_count") or 0,
                "like_count": None,
                "comment_count": None,
                "upload_date": None,
            }
        )

    enrich_top = enrich_top or max(cfg["channel"]["top_n"] * 3, 30)
    videos.sort(key=lambda v: v["view_count"], reverse=True)
    to_enrich = videos[:enrich_top]
    log.info("Enriching %d videos with full metadata", len(to_enrich))

    def enrich(v):
        with _ydl({}) as ydl:
            full = ydl.extract_info(v["url"], download=False)
        if not full:
            return v
        v.update(
            {
                "title": full.get("title") or v["title"],
                "duration": full.get("duration") or v["duration"],
                "view_count": full.get("view_count") or v["view_count"],
                "like_count": full.get("like_count"),
                "comment_count": full.get("comment_count"),
                "upload_date": full.get("upload_date"),
                "chapters": full.get("chapters") or [],
            }
        )
        return v

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(enrich, v): v for v in to_enrich}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("enrich failed for %s: %s", futures[fut]["id"], exc)
            if i % 10 == 0:
                log.info("  enriched %d/%d", i, len(to_enrich))

    write_json(work_path(cfg, "catalog.json"), videos)
    return videos


def _norm(values: list[float]) -> list[float]:
    """Log-scale min-max normalisation; robust to the one 1B-view outlier."""
    logs = [math.log1p(max(v, 0.0)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(x - lo) / (hi - lo) for x in logs]


def rank_catalog(cfg: dict, videos: list[dict] | None = None) -> list[dict]:
    videos = videos or read_json(work_path(cfg, "catalog.json"), [])
    if not videos:
        raise SystemExit("No catalog. Run `catalog` first.")
    today = dt.date.today()
    min_age = cfg["ranking"]["min_age_days"]

    rows = []
    for v in videos:
        if not v.get("upload_date"):
            continue  # only enriched videos are rankable
        up = dt.datetime.strptime(v["upload_date"], "%Y%m%d").date()
        age = max((today - up).days, 1)
        if age < min_age:
            continue
        views = v["view_count"] or 0
        likes = v.get("like_count") or 0
        comments = v.get("comment_count") or 0
        rows.append(
            {
                **v,
                "age_days": age,
                "views_per_day": views / age,
                "like_ratio": likes / views if views else 0,
                "comment_ratio": comments / views if views else 0,
            }
        )
    if not rows:
        raise SystemExit("Nothing rankable (no upload dates). Re-run `catalog`.")

    w = cfg["ranking"]["weights"]
    cols = {k: _norm([r[k] for r in rows]) for k in ("views_per_day", "view_count", "like_ratio", "comment_ratio")}
    for i, r in enumerate(rows):
        r["score"] = round(
            w["views_per_day"] * cols["views_per_day"][i]
            + w["views_total"] * cols["view_count"][i]
            + w["like_ratio"] * cols["like_ratio"][i]
            + w["comment_ratio"] * cols["comment_ratio"][i],
            4,
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    write_json(work_path(cfg, "ranked.json"), rows)
    return rows


def print_ranking(rows: list[dict], n: int) -> None:
    print(f"\n{'#':>3}  {'score':>6}  {'views':>7}  {'v/day':>7}  {'like%':>5}  {'age':>5}  title")
    for r in rows[:n]:
        print(
            f"{r['rank']:>3}  {r['score']:>6.3f}  {r['view_count']/1e6:>6.1f}M  "
            f"{r['views_per_day']/1e3:>6.0f}k  {r['like_ratio']*100:>5.2f}  {r['age_days']:>4}d  {r['title'][:70]}"
        )
    print()
