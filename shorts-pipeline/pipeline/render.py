"""Render a scored segment into a captioned 1080x1920 Short with ffmpeg."""
from __future__ import annotations

import subprocess
import threading

from .captions import CaptionRenderer, build_timeline
from .common import hw_decode_flags, log, output_path, read_json, run, slugify, video_info, work_path, write_json
from .reframe import camera_path, write_sendcmd


def _ff_escape(path) -> str:
    """Escape a filesystem path for use inside a quoted filtergraph option."""
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _reframe_chain(cfg: dict, clip: dict, path: dict, info: dict) -> str:
    """Filtergraph taking [0:v] to a 1080x1920 base labelled [base]."""
    r = cfg["render"]
    W, H, fps = r["width"], r["height"], r["fps"]
    if path["mode"] == "crop":
        crop_w = int(round(info["height"] * 9 / 16 / 2) * 2)
        cuts_rel = [c - clip["start"] for c in clip.get("_scene_cuts", []) if clip["start"] < c < clip["end"]]
        cmd_file, x0 = write_sendcmd(
            path["keyframes"], info["width"], crop_w, cuts_rel,
            work_path(cfg, "sendcmd", f"{clip['video_id']}-c{clip['clip_rank']}.txt"),
        )
        return (
            f"[0:v]sendcmd=f='{_ff_escape(cmd_file)}',crop@cam=w={crop_w}:h={info['height']}:x={x0:.1f}:y=0,"
            f"scale={W}:{H}:flags=lanczos,fps={fps},format=yuv420p[base]"
        )
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=28,eq=brightness=-0.08[bgb];"
        f"[fg]scale={W}:-2:flags=lanczos[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p[base]"
    )


def _encode(cfg: dict, clip: dict, filtergraph: str, hw: list[str], out_mp4, renderer, timeline) -> subprocess.CompletedProcess:
    r = cfg["render"]
    W, H, fps = r["width"], r["height"], r["fps"]
    if r["codec"] == "h264_videotoolbox":
        venc = ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-profile:v", "high", "-allow_sw", "1"]
    else:
        venc = ["-c:v", "libx264", "-crf", str(r["crf"]), "-preset", "medium", "-profile:v", "high", "-level", "4.2"]
    af = f"loudnorm=I={r['audio_lufs']}:TP=-1.5:LRA=11,aresample=48000"
    cmd = (
        ["ffmpeg", "-v", "error", "-y"] + hw
        + ["-ss", f"{clip['start']:.3f}", "-t", f"{clip['duration']:.3f}", "-i", clip["source"]]
        + ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
        + ["-filter_complex", filtergraph, "-map", "[v]", "-map", "0:a:0?"]
        + venc + ["-af", af, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(out_mp4)]
    )
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    err_buf: list[bytes] = []
    t = threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True)
    t.start()
    try:
        renderer.stream(timeline, fps, clip["duration"], proc.stdin)
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
    proc.wait()
    t.join(timeout=5)
    return subprocess.CompletedProcess(cmd, proc.returncode, None, (err_buf[0] if err_buf else b"").decode(errors="replace"))


def render_clip(cfg: dict, clip: dict, transcript: dict, analysis: dict, force: bool = False) -> dict:
    vid = clip["video_id"]
    base = f"{clip.get('video_rank', 0):02d}-{slugify(clip['video_title'], 40)}-c{clip['clip_rank']}"
    out_mp4 = output_path(cfg, vid, base + ".mp4")
    out_json = output_path(cfg, vid, base + ".json")
    if out_mp4.exists() and out_json.exists() and out_mp4.stat().st_size > 0 and not force:
        log.info("exists, skipping: %s", out_mp4.name)
        return read_json(out_json)

    info = video_info(clip["source"])
    clip["_scene_cuts"] = analysis["scene_cuts"]
    log.info("Tracking subject for %s [%s-%s]", base, clip["start"], clip["end"])
    path = camera_path(cfg, clip, info, analysis["scene_cuts"])
    log.info("  reframe mode=%s coverage=%.0f%% detector=%s", path["mode"], path["coverage"] * 100, path["detector"])

    renderer = CaptionRenderer(cfg)
    timeline = build_timeline(cfg, clip, transcript)
    graph = _reframe_chain(cfg, clip, path, info) + ";[base][1:v]overlay=0:0:format=auto,format=yuv420p[v]"

    attempts = [hw_decode_flags(info), []] if hw_decode_flags(info) else [[]]
    res = None
    for hw in attempts:
        log.info("  encoding %s (%s)", out_mp4.name, "videotoolbox decode" if hw else "software decode")
        res = _encode(cfg, clip, graph, hw, out_mp4, renderer, timeline)
        if res.returncode == 0:
            break
        log.debug("encode attempt failed: %s", res.stderr[-400:])
    if res is None or res.returncode != 0:
        out_mp4.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {base}:\n{(res.stderr if res else '')[-2000:]}")

    poster = output_path(cfg, vid, base + ".jpg")
    run(["ffmpeg", "-v", "error", "-y", "-ss", "0.6", "-i", out_mp4, "-frames:v", "1", "-q:v", "3", poster], check=False)

    meta = {k: v for k, v in clip.items() if not k.startswith("_")}
    meta.update({"output": str(out_mp4), "poster": str(poster), "reframe": {k: v for k, v in path.items() if k != "keyframes"}})
    meta["title"] = (clip.get("llm") or {}).get("title") or _fallback_title(clip)
    meta["description"] = cfg["upload"]["description_template"].format(
        hook=(clip.get("llm") or {}).get("hook_line") or clip["hook_line"], video_url=clip.get("video_url") or ""
    ).strip()
    write_json(out_json, meta)
    log.info("  done: %s", out_mp4)
    return meta


def _fallback_title(clip: dict) -> str:
    hook = " ".join(clip["hook_line"].split()[:9]).rstrip(".,!?")
    return hook[:55] if hook else clip["video_title"][:55]
