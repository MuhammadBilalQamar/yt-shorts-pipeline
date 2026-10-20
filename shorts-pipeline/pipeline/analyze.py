"""Signal analysis used by the segment scorer: audio energy curve + scene-cut timestamps."""
from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from .common import hw_decode_flags, log, read_json, video_info, work_path, write_json

WINDOW = 0.5  # seconds per energy bin


def audio_energy(wav: Path) -> list[float]:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * WINDOW)
    bins = max(1, len(pcm) // hop)
    pcm = pcm[: bins * hop].reshape(bins, hop)
    rms = np.sqrt((pcm**2).mean(axis=1) + 1e-9)
    db = 20 * np.log10(rms)
    # Normalise to 0..1 between the 5th and 99th percentile.
    lo, hi = np.percentile(db, 5), np.percentile(db, 99)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    return [round(float(x), 4) for x in norm]


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def scene_cuts(src: str | Path, threshold: float = 0.3) -> list[float]:
    vf = f"scale=320:-2,select='gt(scene,{threshold})',showinfo"
    base = ["ffmpeg", "-v", "info", "-nostats"]
    hw_pref = hw_decode_flags(video_info(src))
    for hw in ([hw_pref, []] if hw_pref else [[]]):
        cmd = base + hw + ["-i", str(src), "-an", "-vf", vf, "-f", "null", "-"]
        try:
            res = subprocess.run(cmd, text=True, capture_output=True, check=True)
            cuts = [float(m) for m in _PTS_RE.findall(res.stderr)]
            return sorted(set(round(c, 2) for c in cuts))
        except subprocess.CalledProcessError as exc:
            if hw:
                log.debug("videotoolbox decode failed, retrying in software: %s", exc.stderr[-300:])
                continue
            raise
    return []


def analyze(cfg: dict, video_id: str, src: str | Path, force: bool = False) -> dict:
    out = work_path(cfg, "analysis", f"{video_id}.json")
    if out.exists() and not force:
        return read_json(out)
    wav = work_path(cfg, "audio", f"{video_id}.wav")
    if not wav.exists():
        from .transcribe import extract_audio

        extract_audio(src, wav)
    log.info("Analysing audio energy for %s", video_id)
    energy = audio_energy(wav)
    log.info("Detecting scene cuts for %s", video_id)
    cuts = scene_cuts(src)
    data = {"video_id": video_id, "window": WINDOW, "energy": energy, "scene_cuts": cuts}
    write_json(out, data)
    log.info("Analysis %s: %d energy bins, %d scene cuts", video_id, len(energy), len(cuts))
    return data
