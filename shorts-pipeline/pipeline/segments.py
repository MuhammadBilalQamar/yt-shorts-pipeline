"""Find and rank candidate Short segments inside a long-form video.

Two stages:
  1. Heuristic scorer over transcript + audio energy + scene cuts (always runs, fully local).
  2. Optional local-LLM (Ollama) re-rank of the top candidates for hook strength,
     retention and standalone clarity. Falls back to stage 1 if Ollama is unavailable.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from .common import log, read_json, work_path, write_json

NUM_RE = re.compile(r"\$|\b\d[\d,.]*\b|\b(million|billion|thousand|hundred|k)\b", re.I)
HOOK_WORDS = {
    "i", "we", "this", "watch", "if", "last", "never", "ever", "world", "most", "biggest", "first",
    "only", "insane", "crazy", "secret", "impossible", "challenge", "survive", "win", "wins", "won",
    "gave", "spent", "bought", "built", "buried", "trapped", "free", "money", "dollars", "prize",
    "versus", "vs", "whoever", "every", "everyone", "nobody", "you", "imagine", "what", "why", "how",
}
TENSION_WORDS = {"but", "until", "suddenly", "wait", "then", "however", "actually", "except", "unless", "instead"}
REACTION_RE = re.compile(r"\b(oh my|no way|what\?|let'?s go|wow|insane|are you kidding|holy|bro|dude)\b", re.I)
DANGLING_OPENERS = {"he", "she", "it", "they", "that", "this", "those", "these", "and", "so", "because", "which", "him", "her", "them"}
STRONG_OPENERS = {"i", "we", "if", "this", "watch", "imagine", "what", "why", "how", "today", "whoever", "you", "the"}


def _units(transcript: dict) -> list[dict]:
    """Whisper segments, re-split on terminal punctuation into sentence-ish units."""
    units = []
    for seg in transcript["segments"]:
        words = seg.get("words") or []
        if not words:
            if seg["text"]:
                units.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "words": []})
            continue
        cur: list[dict] = []
        for w in words:
            cur.append(w)
            if re.search(r"[.!?]$", w["w"]) and len(cur) >= 3:
                units.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": " ".join(x["w"] for x in cur), "words": cur})
                cur = []
        if cur:
            units.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": " ".join(x["w"] for x in cur), "words": cur})
    # Merge tiny fragments into the previous unit.
    merged: list[dict] = []
    for u in units:
        if merged and (u["end"] - u["start"] < 0.6 or len(u["text"].split()) < 2):
            p = merged[-1]
            p["end"], p["text"], p["words"] = u["end"], p["text"] + " " + u["text"], p["words"] + u["words"]
        else:
            merged.append(dict(u))
    return merged


def _energy_slice(energy: list[float], window: float, start: float, end: float) -> np.ndarray:
    a, b = int(start / window), max(int(start / window) + 1, int(math.ceil(end / window)))
    arr = np.asarray(energy[a:b], dtype=np.float32)
    return arr if arr.size else np.zeros(1, dtype=np.float32)


def _hook_score(first: dict, second: dict | None, energy: list[float], window: float) -> float:
    text = first["text"]
    words = re.findall(r"[a-z']+", text.lower())
    n = len(words)
    s = 0.0
    if NUM_RE.search(text):
        s += 0.22
    if "?" in text:
        s += 0.12
    hits = sum(1 for w in words[:10] if w in HOOK_WORDS)
    s += min(hits, 3) * 0.08
    if 4 <= n <= 14:
        s += 0.18
    elif n > 24:
        s -= 0.15
    if words and words[0] in STRONG_OPENERS:
        s += 0.12
    if words and words[0] in DANGLING_OPENERS:
        s -= 0.2  # starts mid-thought; bad continuity
    if text and text[0].islower():
        s -= 0.15
    e = _energy_slice(energy, window, first["start"], first["start"] + 3.0)
    s += 0.15 * float(e.mean())
    if second and NUM_RE.search(second["text"]):
        s += 0.05
    return max(0.0, min(1.0, s))


def _engagement_score(units: list[dict], energy: list[float], window: float, cuts: list[float]) -> float:
    start, end = units[0]["start"], units[-1]["end"]
    dur = max(end - start, 1e-3)
    text = " ".join(u["text"] for u in units)
    words = re.findall(r"[a-z']+", text.lower())
    wps = len(words) / dur
    s = 0.0
    # Speech pace: 2.3-4.2 wps feels energetic without being unintelligible.
    s += 0.2 * math.exp(-((wps - 3.2) ** 2) / (2 * 0.9**2))
    s += min(text.count("!"), 4) * 0.04
    s += min(len(REACTION_RE.findall(text)), 3) * 0.06
    s += min(sum(1 for w in words if w in TENSION_WORDS), 4) * 0.04
    s += min(len(NUM_RE.findall(text)), 4) * 0.03
    e = _energy_slice(energy, window, start, end)
    s += 0.18 * float(e.mean()) + 0.12 * float(e.std()) * 2
    cuts_in = sum(1 for c in cuts if start <= c <= end)
    cps = cuts_in / dur * 10  # cuts per 10s
    s += 0.15 * math.exp(-((cps - 4.0) ** 2) / (2 * 2.5**2))
    # Dead air penalty: fraction of bins below 0.15 energy.
    s -= 0.25 * float((e < 0.15).mean())
    return max(0.0, min(1.0, s))


def _payoff_score(units: list[dict], energy: list[float], window: float) -> float:
    last = units[-1]
    start, end = units[0]["start"], last["end"]
    s = 0.0
    if re.search(r"[.!?]$", last["text"].strip()):
        s += 0.4
    if "!" in last["text"] or REACTION_RE.search(last["text"]):
        s += 0.2
    e = _energy_slice(energy, window, start, end)
    third = max(1, len(e) // 3)
    if e[-third:].mean() >= e[:third].mean():
        s += 0.3
    if "?" in last["text"]:
        s += 0.1  # mild cliffhanger is fine for a Short that links out
    return max(0.0, min(1.0, s))


def _length_score(dur: float, cfg_seg: dict) -> float:
    return math.exp(-((dur - cfg_seg["target_len"]) ** 2) / (2 * 9.0**2))


def heuristic_candidates(cfg: dict, transcript: dict, analysis: dict) -> list[dict]:
    seg_cfg = cfg["segments"]
    units = _units(transcript)
    energy, window, cuts = analysis["energy"], analysis["window"], analysis["scene_cuts"]
    cands = []
    for i in range(len(units)):
        for j in range(i, len(units)):
            dur = units[j]["end"] - units[i]["start"]
            if dur < seg_cfg["min_len"]:
                continue
            if dur > seg_cfg["max_len"]:
                break
            span = units[i : j + 1]
            hook = _hook_score(units[i], units[i + 1] if i + 1 < len(units) else None, energy, window)
            eng = _engagement_score(span, energy, window, cuts)
            pay = _payoff_score(span, energy, window)
            ln = _length_score(dur, seg_cfg)
            total = 0.35 * hook + 0.35 * eng + 0.15 * pay + 0.15 * ln
            cands.append(
                {
                    "start": round(units[i]["start"], 2),
                    "end": round(units[j]["end"], 2),
                    "duration": round(dur, 2),
                    "hook_line": units[i]["text"],
                    "text": " ".join(u["text"] for u in span),
                    "scores": {"hook": round(hook, 3), "engagement": round(eng, 3), "payoff": round(pay, 3), "length": round(ln, 3)},
                    "heuristic": round(total, 4),
                }
            )
    cands.sort(key=lambda c: c["heuristic"], reverse=True)
    # Greedy non-overlapping selection with a minimum gap.
    chosen: list[dict] = []
    for c in cands:
        if all(c["end"] + seg_cfg["min_gap"] <= k["start"] or c["start"] >= k["end"] + seg_cfg["min_gap"] for k in chosen):
            chosen.append(c)
        if len(chosen) >= seg_cfg["llm_candidates"]:
            break
    return chosen


# ---------------------------------------------------------------- local LLM re-rank


def _ollama_available(cfg: dict) -> bool:
    import requests

    try:
        r = requests.get(cfg["llm"]["host"] + "/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        want = cfg["llm"]["model"]
        ok = any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)
        if not ok:
            log.warning("Ollama is running but model %s is not pulled (have: %s)", want, models)
        return ok
    except Exception:  # noqa: BLE001
        return False


_PROMPT = """You are a senior short-form editor for a top YouTube channel. You are given candidate
clips (transcript excerpts) from one long video. For each candidate, score 0-10:
- hook: do the first ~2 seconds make a cold viewer stop scrolling? (stakes, number, question, bold claim)
- retention: does the clip keep escalating with no dead air until the end?
- standalone: does it make sense with zero context from the rest of the video?
Also write:
- hook_line: a punchy on-screen opening line, max 7 words, derived from the clip's own words.
- title: a YouTube Shorts title, max 55 chars, no clickbait that the clip doesn't pay off.

Video title: {video_title}

Candidates:
{candidates}

Reply with ONLY a JSON object: {{"results": [{{"id": <int>, "hook": <0-10>, "retention": <0-10>, "standalone": <0-10>, "hook_line": "...", "title": "..."}}, ...]}}
Include every candidate id exactly once."""


def llm_rerank(cfg: dict, video_title: str, cands: list[dict]) -> list[dict]:
    import requests

    body = "\n\n".join(f"[id {i}] ({c['duration']:.0f}s) {c['text'][:900]}" for i, c in enumerate(cands))
    prompt = _PROMPT.format(video_title=video_title, candidates=body)
    r = requests.post(
        cfg["llm"]["host"] + "/api/chat",
        json={
            "model": cfg["llm"]["model"],
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        },
        timeout=cfg["llm"]["timeout_sec"],
    )
    r.raise_for_status()
    content = r.json()["message"]["content"]
    data = json.loads(content)
    results = data.get("results") if isinstance(data, dict) else data
    by_id = {}
    for res in results or []:
        try:
            by_id[int(res["id"])] = res
        except (KeyError, TypeError, ValueError):
            continue
    for i, c in enumerate(cands):
        res = by_id.get(i)
        if not res:
            c["llm"] = None
            c["final"] = c["heuristic"]
            continue
        h, rt, st = (float(res.get(k, 5)) / 10 for k in ("hook", "retention", "standalone"))
        llm_score = 0.45 * h + 0.35 * rt + 0.2 * st
        c["llm"] = {"hook": h, "retention": rt, "standalone": st, "hook_line": res.get("hook_line"), "title": res.get("title")}
        c["final"] = round(0.5 * c["heuristic"] + 0.5 * llm_score, 4)
    return cands


def _snap_bounds(c: dict, analysis: dict, transcript: dict) -> None:
    """Lead-in/out padding plus snapping the out-point to a nearby scene cut."""
    c["start"] = round(max(0.0, c["start"] - 0.15), 2)
    end = c["end"] + 0.35
    for cut in analysis["scene_cuts"]:
        if c["end"] + 0.1 <= cut <= c["end"] + 0.8:
            end = cut - 0.04
            break
    c["end"] = round(end, 2)
    c["duration"] = round(c["end"] - c["start"], 2)


def find_segments(cfg: dict, video: dict, transcript: dict, analysis: dict, force: bool = False) -> list[dict]:
    out = work_path(cfg, "segments", f"{video['id']}.json")
    if out.exists() and not force:
        return read_json(out)
    cands = heuristic_candidates(cfg, transcript, analysis)
    if not cands:
        log.warning("No candidate segments for %s", video["id"])
        write_json(out, [])
        return []
    for c in cands:
        c["final"] = c["heuristic"]
        c["llm"] = None
    if cfg["llm"]["enabled"] and _ollama_available(cfg):
        try:
            log.info("LLM re-ranking %d candidates for %s with %s", len(cands), video["id"], cfg["llm"]["model"])
            cands = llm_rerank(cfg, video.get("title", ""), cands)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM re-rank failed (%s); using heuristic ranking", exc)
    elif cfg["llm"]["enabled"]:
        log.warning("Ollama not reachable; using heuristic ranking only")
    cands.sort(key=lambda c: c["final"], reverse=True)
    keep = cands[: cfg["segments"]["clips_per_video"]]
    for i, c in enumerate(keep, 1):
        _snap_bounds(c, analysis, transcript)
        c["clip_rank"] = i
        c["video_id"] = video["id"]
        c["video_title"] = video.get("title", "")
        c["video_url"] = video.get("url", "")
        c["source"] = video["local_file"]
    write_json(out, keep)
    return keep
