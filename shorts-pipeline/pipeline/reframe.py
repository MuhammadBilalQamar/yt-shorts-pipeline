"""Subject-tracked 9:16 reframing: computes a smoothed horizontal camera path per clip."""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from .common import ROOT, hw_decode_flags, log

SAMPLE_FPS = 5
PROBE_W = 640
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _yunet_path() -> Path | None:
    p = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    if p.exists():
        return p
    try:
        p.parent.mkdir(exist_ok=True)
        log.info("Downloading YuNet face model")
        urllib.request.urlretrieve(YUNET_URL, p)
        return p
    except Exception as exc:  # noqa: BLE001
        log.warning("YuNet download failed (%s); falling back to Haar cascade", exc)
        return None


class FaceDetector:
    """YuNet (OpenCV DNN). OpenCV 5 removed the Haar cascades, so there is no cascade fallback;
    if the model can't be loaded the clip is rendered in blur/pillarbox mode instead."""

    def __init__(self, w: int, h: int):
        import cv2

        self.kind = "none"
        self.det = None
        model = _yunet_path()
        if model is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.det = cv2.FaceDetectorYN.create(str(model), "", (w, h), score_threshold=0.6)
                self.det.setInputSize((w, h))
                self.kind = "yunet"
            except Exception as exc:  # noqa: BLE001
                log.warning("YuNet init failed (%s); subject tracking disabled", exc)

    def __call__(self, frame) -> list[tuple[float, float, float, float]]:
        """Returns [(x, y, w, h)] in pixel coords of the given frame."""
        if self.det is None:
            return []
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces]


def _frames(src: str, start: float, dur: float, w: int, h: int, hw: list[str]):
    cmd = (
        ["ffmpeg", "-v", "error"] + hw + ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
        "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _sample(src: str, start: float, dur: float, w: int, h: int, hw: list[str]):
    """Yield probe frames; retry with software decode if the hardware path yields nothing."""
    got = False
    for f in _frames(src, start, dur, w, h, hw):
        got = True
        yield f
    if not got and hw:
        log.debug("hw-decoded probe produced no frames; retrying in software")
        yield from _frames(src, start, dur, w, h, [])


def camera_path(cfg: dict, clip: dict, info: dict, scene_cuts: list[float]) -> dict:
    """Returns {'mode': 'crop'|'blur', 'coverage': f, 'keyframes': [(t_rel, cx_norm), ...]}."""
    rcfg = cfg["render"]
    src_w, src_h = info["width"], info["height"]
    pw = PROBE_W
    ph = int(round(src_h * pw / src_w / 2) * 2)
    det = FaceDetector(pw, ph)
    crop_frac = (src_h * 9 / 16) / src_w  # crop width as a fraction of source width
    half = crop_frac / 2

    if det.kind == "none":
        return {"mode": "blur", "coverage": 0.0, "keyframes": [], "detector": "none"}

    centers: list[float | None] = []
    for frame in _sample(clip["source"], clip["start"], clip["duration"], pw, ph, hw_decode_flags(info)):
        faces = det(frame)
        if not faces:
            centers.append(None)
            continue
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        main = faces[0]
        cx = (main[0] + main[2] / 2) / pw
        # If a second face is close enough to fit in the crop, centre between them.
        if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.4 * main[2] * main[3]:
            cx2 = (faces[1][0] + faces[1][2] / 2) / pw
            if abs(cx2 - cx) < crop_frac * 0.8:
                cx = (cx + cx2) / 2
        centers.append(float(cx))

    n = len(centers)
    if n == 0:
        return {"mode": "blur", "coverage": 0.0, "keyframes": [], "detector": det.kind}
    coverage = sum(c is not None for c in centers) / n
    mode = rcfg["mode"]
    if mode == "auto":
        mode = "crop" if coverage >= rcfg["crop_min_face_coverage"] else "blur"
    if mode == "blur":
        return {"mode": "blur", "coverage": round(coverage, 3), "keyframes": [], "detector": det.kind}

    # Fill gaps by holding the last known centre (or 0.5 before the first detection).
    filled = []
    last = None
    for c in centers:
        if c is not None:
            last = c
        filled.append(last)
    first_known = next((c for c in filled if c is not None), 0.5)
    arr = np.array([first_known if c is None else c for c in filled], dtype=np.float32)

    # Smooth within shots only, so the crop jumps at scene cuts instead of panning across them.
    cut_idx = sorted({int(round((c - clip["start"]) * SAMPLE_FPS)) for c in scene_cuts if clip["start"] < c < clip["end"]})
    bounds = [0] + [i for i in cut_idx if 0 < i < n] + [n]
    win = max(1, int(rcfg["crop_smoothing_sec"] * SAMPLE_FPS))
    smooth = np.empty_like(arr)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = arr[a:b]
        if len(seg) == 0:
            continue
        k = min(win, len(seg))
        if k <= 1:
            smooth[a:b] = seg
            continue
        pad = np.pad(seg, (k // 2, k - 1 - k // 2), mode="edge")
        smooth[a:b] = np.convolve(pad, np.ones(k) / k, mode="valid")[: len(seg)]
    smooth = np.clip(smooth, half, 1 - half)

    keyframes = [(round(i / SAMPLE_FPS, 3), round(float(v), 4)) for i, v in enumerate(smooth)]
    return {"mode": "crop", "coverage": round(coverage, 3), "keyframes": keyframes, "detector": det.kind, "shots": len(bounds) - 1}


def write_sendcmd(
    keyframes: list[tuple[float, float]], src_w: int, crop_w: int, scene_cut_rel: list[float], out: Path
) -> tuple[Path, float]:
    """Write an ffmpeg `sendcmd` file that re-targets crop@cam's x expression per keyframe interval.

    ffmpeg caps the length of a single filter option, so one giant piecewise expression breaks on
    clips longer than ~30s. sendcmd sends a short linear expression at each interval instead:
    piecewise-linear pans between keyframes, hard jumps across scene cuts.
    Returns (path, initial_x).
    """
    x_max = src_w - crop_w
    if not keyframes:
        out.write_text("")
        return out, x_max / 2
    cut_set = set(round(c * SAMPLE_FPS) / SAMPLE_FPS for c in scene_cut_rel)
    step = 2  # one command per 0.4s is plenty for a smooth path
    kfs = keyframes[::step] + ([keyframes[-1]] if (len(keyframes) - 1) % step else [])

    def px(c: float) -> float:
        return min(max(c * src_w - crop_w / 2, 0.0), float(x_max))

    lines = []
    for (t0, c0), (t1, c1) in zip(kfs[:-1], kfs[1:]):
        x0, x1 = px(c0), px(c1)
        jump = any(t0 < c <= t1 for c in cut_set)
        if jump or abs(x1 - x0) < 0.5:
            expr = f"{x0:.1f}"
        else:
            expr = f"{x0:.1f}+({x1 - x0:.1f})*(t-{t0})/{t1 - t0:.3f}"
        lines.append(f"{t0:.3f}-{t1:.3f} [enter] crop@cam x '{expr}';")
    tl, cl = kfs[-1]
    lines.append(f"{tl:.3f} [enter] crop@cam x '{px(cl):.1f}';")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out, px(kfs[0][1])
