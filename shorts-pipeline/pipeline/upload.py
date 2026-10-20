"""Upload queue + YouTube Data API v3 uploader (OAuth desktop flow, token cached locally)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from .common import log, read_json, work_path, write_json

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]


def queue_path(cfg: dict) -> Path:
    return work_path(cfg, "queue.json")


def enqueue(cfg: dict, metas: list[dict]) -> list[dict]:
    q = read_json(queue_path(cfg), [])
    have = {item["output"] for item in q}
    for m in metas:
        if m["output"] in have:
            continue
        q.append(
            {
                "output": m["output"],
                "video_id": m["video_id"],
                "title": m["title"],
                "description": m["description"],
                "score": m.get("final", m.get("heuristic", 0)),
                "status": "pending",
                "youtube_id": None,
                "publish_at": None,
            }
        )
    q.sort(key=lambda i: i["score"], reverse=True)
    write_json(queue_path(cfg), q)
    return q


def print_queue(cfg: dict) -> None:
    q = read_json(queue_path(cfg), [])
    if not q:
        print("Queue is empty.")
        return
    print(f"\n{'status':>9}  {'score':>6}  {'yt id':>12}  title")
    for i in q:
        print(f"{i['status']:>9}  {i['score']:>6.3f}  {str(i['youtube_id'] or '-'):>12}  {i['title'][:60]}   ({Path(i['output']).name})")
    print()


def _service(cfg: dict):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    secret = Path(cfg["upload"]["client_secret"])
    token = Path(cfg["upload"]["token_file"])
    if not secret.exists():
        raise SystemExit(
            f"Missing OAuth client secret at {secret}.\n"
            "Create a Desktop OAuth client in Google Cloud Console (YouTube Data API v3 enabled) and save it there."
        )
    creds = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
            creds = flow.run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def whoami(cfg: dict) -> dict:
    yt = _service(cfg)
    res = yt.channels().list(part="snippet,statistics", mine=True).execute()
    items = res.get("items") or []
    if not items:
        raise SystemExit("OAuth account has no channel. Sign in with the account that manages the target channel.")
    ch = items[0]
    out = {"id": ch["id"], "title": ch["snippet"]["title"], "subs": ch["statistics"].get("subscriberCount")}
    log.info("Authenticated as channel: %s (%s)", out["title"], out["id"])
    return out


def drain(cfg: dict, limit: int = 5, dry_run: bool = False, schedule: bool = True) -> None:
    from googleapiclient.http import MediaFileUpload

    q = read_json(queue_path(cfg), [])
    pending = [i for i in q if i["status"] == "pending"][:limit]
    if not pending:
        log.info("Nothing pending.")
        return
    yt = None if dry_run else _service(cfg)
    if yt is not None:
        whoami(cfg)
    gap = dt.timedelta(minutes=cfg["upload"]["schedule_gap_min"])
    next_pub = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
    for item in pending:
        body = {
            "snippet": {
                "title": item["title"][:100],
                "description": item["description"][:4900],
                "tags": cfg["upload"]["tags"],
                "categoryId": cfg["upload"]["category_id"],
            },
            "status": {"privacyStatus": cfg["upload"]["privacy"], "selfDeclaredMadeForKids": False},
        }
        if schedule and cfg["upload"]["privacy"] == "private":
            body["status"]["publishAt"] = next_pub.isoformat().replace("+00:00", "Z")
            item["publish_at"] = body["status"]["publishAt"]
            next_pub += gap
        if dry_run:
            log.info("[dry-run] would upload %s as %r publishAt=%s", Path(item["output"]).name, body["snippet"]["title"], item["publish_at"])
            continue
        media = MediaFileUpload(item["output"], chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            status, resp = req.next_chunk()
            if status:
                log.info("  %s %.0f%%", Path(item["output"]).name, status.progress() * 100)
        item["status"], item["youtube_id"] = "uploaded", resp["id"]
        log.info("Uploaded %s -> https://youtube.com/shorts/%s", Path(item["output"]).name, resp["id"])
        write_json(queue_path(cfg), q)
    write_json(queue_path(cfg), q)
