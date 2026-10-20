"""CLI entry point.  Run `python -m pipeline.cli --help`."""
from __future__ import annotations

import argparse
import sys

from . import analyze as an
from . import catalog as cat
from . import render as rn
from . import segments as sg
from . import sources as so
from . import transcribe as tr
from . import upload as up
from .common import load_config, log, read_json, setup_logging, work_path


def _process_video(cfg: dict, video: dict, force: bool) -> list[dict]:
    if not video.get("local_file"):
        log.warning("Skipping #%s %s: no local file", video.get("rank"), video.get("title"))
        return []
    vid, src = video["id"], video["local_file"]
    transcript = tr.transcribe(cfg, vid, src, force=force)
    analysis = an.analyze(cfg, vid, src, force=force)
    segs = sg.find_segments(cfg, video, transcript, analysis, force=force)
    metas = []
    for s in segs:
        s["video_rank"] = video.get("rank", 0)
        try:
            metas.append(rn.render_clip(cfg, s, transcript, analysis, force=force))
        except Exception as exc:  # noqa: BLE001
            log.error("Render failed for %s clip %s: %s", vid, s.get("clip_rank"), exc)
    return metas


def cmd_catalog(cfg, a):
    cat.fetch_catalog(cfg, limit=a.limit)
    rows = cat.rank_catalog(cfg)
    cat.print_ranking(rows, a.show)


def cmd_rank(cfg, a):
    rows = cat.rank_catalog(cfg)
    cat.print_ranking(rows, a.show)


def cmd_match(cfg, a):
    m = so.match_sources(cfg, top_n=a.top)
    if a.download:
        cfg["sources"]["allow_download"] = True
    so.download_missing(cfg, m)


def cmd_process(cfg, a):
    if a.local:
        videos = so.local_only_sources(cfg, a.source)
    else:
        videos = read_json(work_path(cfg, "sources.json"), [])
        if not videos:
            raise SystemExit("No matched sources. Run `catalog` then `match` first (or use --local).")
    if a.only:
        videos = [v for v in videos if v["id"] in a.only or any(o in (v.get("local_file") or "") for o in a.only)]
    if a.top:
        videos = videos[: a.top]
    all_meta = []
    for v in videos:
        log.info("=== #%s %s", v.get("rank"), v.get("title"))
        all_meta.extend(_process_video(cfg, v, a.force))
    up.enqueue(cfg, all_meta)
    log.info("Rendered %d Shorts. Queue:", len(all_meta))
    up.print_queue(cfg)


def cmd_run(cfg, a):
    cat.fetch_catalog(cfg, limit=a.limit)
    rows = cat.rank_catalog(cfg)
    cat.print_ranking(rows, a.top or cfg["channel"]["top_n"])
    m = so.match_sources(cfg, rows, top_n=a.top)
    if a.download:
        cfg["sources"]["allow_download"] = True
    m = so.download_missing(cfg, m)
    all_meta = []
    for v in m:
        log.info("=== #%s %s", v.get("rank"), v.get("title"))
        all_meta.extend(_process_video(cfg, v, a.force))
    up.enqueue(cfg, all_meta)
    up.print_queue(cfg)
    if a.upload:
        up.drain(cfg, limit=a.upload)


def cmd_queue(cfg, a):
    up.print_queue(cfg)


def cmd_upload(cfg, a):
    up.drain(cfg, limit=a.limit, dry_run=a.dry_run, schedule=not a.no_schedule)


def cmd_whoami(cfg, a):
    print(up.whoami(cfg))


def main(argv=None):
    p = argparse.ArgumentParser(prog="shorts", description="Long-form channel videos -> ranked 9:16 Shorts")
    p.add_argument("-c", "--config", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("catalog", help="fetch channel metadata and rank it")
    s.add_argument("--limit", type=int, default=None, help="most recent N uploads to consider")
    s.add_argument("--show", type=int, default=25)
    s.set_defaults(fn=cmd_catalog)

    s = sub.add_parser("rank", help="re-rank the cached catalog")
    s.add_argument("--show", type=int, default=25)
    s.set_defaults(fn=cmd_rank)

    s = sub.add_parser("match", help="match ranked videos to local source files")
    s.add_argument("--top", type=int, default=None)
    s.add_argument("--download", action="store_true", help="download videos with no local file")
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("process", help="transcribe, score, cut and render matched videos")
    s.add_argument("--local", action="store_true", help="ignore the catalog; process every file in the source folder")
    s.add_argument("--source", default=None, help="override source folder (with --local)")
    s.add_argument("--only", nargs="*", default=None, help="video ids or filename fragments")
    s.add_argument("--top", type=int, default=None)
    s.add_argument("--force", action="store_true", help="recompute cached transcripts/segments/renders")
    s.set_defaults(fn=cmd_process)

    s = sub.add_parser("run", help="catalog -> rank -> match -> process (-> upload)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--top", type=int, default=None)
    s.add_argument("--download", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--upload", type=int, default=0, metavar="N", help="upload up to N queued Shorts at the end")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("queue", help="show the upload queue")
    s.set_defaults(fn=cmd_queue)

    s = sub.add_parser("upload", help="upload pending Shorts from the queue")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-schedule", action="store_true", help="publish immediately instead of spacing out publishAt")
    s.set_defaults(fn=cmd_upload)

    s = sub.add_parser("whoami", help="show which YouTube channel the OAuth token belongs to")
    s.set_defaults(fn=cmd_whoami)

    a = p.parse_args(argv)
    setup_logging(a.verbose)
    cfg = load_config(a.config)
    a.fn(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
