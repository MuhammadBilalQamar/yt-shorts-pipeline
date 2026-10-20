from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("shorts")

ROOT = Path(__file__).resolve().parent.parent


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the project root.
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str((ROOT / val).resolve()) if not os.path.isabs(val) else val
    for key in ("client_secret", "token_file"):
        val = cfg["upload"][key]
        cfg["upload"][key] = str((ROOT / val).resolve()) if not os.path.isabs(val) else val
    return cfg


def work_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["work_dir"], *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["output_dir"], *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run(cmd: list[str], check: bool = True, capture: bool = False, **kw) -> subprocess.CompletedProcess:
    log.debug("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        [str(c) for c in cmd],
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        **kw,
    )


def ffprobe(path: str | Path) -> dict:
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture=True,
    ).stdout
    return json.loads(out)


def video_info(path: str | Path) -> dict:
    info = ffprobe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "duration": float(info["format"].get("duration", 0.0)),
        "rotation": int((v.get("tags") or {}).get("rotate", 0) or 0),
    }


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].strip("-") or "clip"


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
