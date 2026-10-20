"""Match ranked videos to local source files; optionally download what's missing."""
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path

from .common import log, read_json, slugify, work_path, write_json

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi", ".mxf"}
YT_ID_RE = re.compile(r"(?<![\w-])([\w-]{11})(?![\w-])")


def list_sources(source_dir: str | Path) -> list[Path]:
    d = Path(source_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))


def _load_manifest(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                out[row[0].strip()] = row[1].strip()
    return out


def _title_key(s: str) -> str:
    s = re.sub(r"\.[^.]+$", "", s)
    s = re.sub(r"[\W_]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def match_sources(cfg: dict, ranked: list[dict] | None = None, top_n: int | None = None) -> list[dict]:
    ranked = ranked or read_json(work_path(cfg, "ranked.json"), [])
    top_n = top_n or cfg["channel"]["top_n"]
    src_dir = cfg["paths"]["source_dir"]
    files = list_sources(src_dir)
    manifest = _load_manifest(cfg["paths"]["manifest"])
    threshold = cfg["sources"]["match_threshold"]
    log.info("Matching top %d videos against %d files in %s", top_n, len(files), src_dir)

    by_name = {f.name: f for f in files}
    keys = {f: _title_key(f.name) for f in files}
    matched: list[dict] = []
    for v in ranked[:top_n]:
        local = None
        how = None
        if v["id"] in manifest and manifest[v["id"]] in by_name:
            local, how = by_name[manifest[v["id"]]], "manifest"
        if local is None:
            for f in files:
                if v["id"] in f.name:
                    local, how = f, "id-in-filename"
                    break
        if local is None and files:
            tk = _title_key(v["title"])
            best = max(files, key=lambda f: difflib.SequenceMatcher(None, tk, keys[f]).ratio())
            ratio = difflib.SequenceMatcher(None, tk, keys[best]).ratio()
            if ratio >= threshold:
                local, how = best, f"title-fuzzy({ratio:.2f})"
        entry = {**v, "local_file": str(local) if local else None, "match": how}
        matched.append(entry)
        log.info("  #%d %s -> %s", v["rank"], v["title"][:50], local.name if local else "NO LOCAL FILE")

    write_json(work_path(cfg, "sources.json"), matched)
    return matched


def download_missing(cfg: dict, matched: list[dict] | None = None) -> list[dict]:
    """Download ranked videos that have no local file (only when sources.allow_download)."""
    matched = matched or read_json(work_path(cfg, "sources.json"), [])
    if not cfg["sources"]["allow_download"]:
        missing = [m for m in matched if not m["local_file"]]
        if missing:
            log.warning(
                "%d ranked videos have no local file and sources.allow_download is false: %s",
                len(missing),
                ", ".join(m["id"] for m in missing),
            )
        return matched
    import yt_dlp

    dl_dir = Path(cfg["paths"]["source_dir"]) / "_downloaded"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for m in matched:
        if m["local_file"]:
            continue
        out = dl_dir / f"{slugify(m['title'])}-{m['id']}.mp4"
        if out.exists():
            m["local_file"], m["match"] = str(out), "downloaded"
            continue
        log.info("Downloading %s", m["title"])
        opts = {
            "format": cfg["sources"]["download_format"],
            "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([m["url"]])
        if out.exists():
            m["local_file"], m["match"] = str(out), "downloaded"
    write_json(work_path(cfg, "sources.json"), matched)
    return matched


def local_only_sources(cfg: dict, source_dir: str | Path | None = None) -> list[dict]:
    """Treat every file in the source folder as a video; no YouTube metadata needed."""
    files = list_sources(source_dir or cfg["paths"]["source_dir"])
    matched = []
    for i, f in enumerate(files, 1):
        m = YT_ID_RE.search(f.stem)
        matched.append(
            {
                "id": m.group(1) if m else slugify(f.stem, 40),
                "title": f.stem,
                "url": "",
                "rank": i,
                "score": 0,
                "local_file": str(f),
                "match": "local-only",
            }
        )
    write_json(work_path(cfg, "sources.json"), matched)
    return matched
