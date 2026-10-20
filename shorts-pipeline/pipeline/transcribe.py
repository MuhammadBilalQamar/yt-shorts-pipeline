"""Local speech-to-text with word timestamps (mlx-whisper on Metal, faster-whisper fallback)."""
from __future__ import annotations

import time
from pathlib import Path

from .common import log, read_json, run, work_path, write_json


def extract_audio(src: str | Path, dst: Path) -> Path:
    if not dst.exists():
        run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst])
    return dst


def _backend():
    try:
        import mlx_whisper  # noqa: F401

        return "mlx"
    except Exception:  # noqa: BLE001
        return "faster"


def transcribe(cfg: dict, video_id: str, src: str | Path, force: bool = False) -> dict:
    out = work_path(cfg, "transcripts", f"{video_id}.json")
    if out.exists() and not force:
        return read_json(out)
    wav = extract_audio(src, work_path(cfg, "audio", f"{video_id}.wav"))
    lang = cfg["transcribe"].get("language")
    backend = _backend()
    t0 = time.time()
    segments: list[dict] = []

    if backend == "mlx":
        import mlx_whisper

        model = cfg["transcribe"]["mlx_model"]
        log.info("Transcribing %s with mlx-whisper (%s)", video_id, model)
        res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, word_timestamps=True, language=lang, verbose=False)
        for s in res["segments"]:
            words = [
                {"w": w["word"].strip(), "s": float(w["start"]), "e": float(w["end"])}
                for w in s.get("words", [])
                if w["word"].strip()
            ]
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(), "words": words})
    else:
        from faster_whisper import WhisperModel

        model = cfg["transcribe"]["faster_whisper_model"]
        log.info("Transcribing %s with faster-whisper (%s, cpu int8)", video_id, model)
        wm = WhisperModel(model, device="cpu", compute_type="int8")
        segs, _ = wm.transcribe(str(wav), word_timestamps=True, language=lang, vad_filter=True)
        for s in segs:
            words = [{"w": w.word.strip(), "s": float(w.start), "e": float(w.end)} for w in (s.words or []) if w.word.strip()]
            segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(), "words": words})

    data = {"video_id": video_id, "backend": backend, "model": model, "segments": segments}
    write_json(out, data)
    log.info("Transcribed %s: %d segments in %.0fs", video_id, len(segments), time.time() - t0)
    return data
