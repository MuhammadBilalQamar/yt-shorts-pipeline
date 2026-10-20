"""ASS subtitle generation: word-highlight captions, hook banner and end CTA."""
from __future__ import annotations

from pathlib import Path

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{base},{base},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1
Style: Hook,{font},{hook_size},{base},{base},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},3,8,70,70,{hook_margin},1
Style: CTA,{font},{cta_size},{base},{base},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,70,70,{cta_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def clip_words(transcript: dict, start: float, end: float) -> list[dict]:
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words") or []:
            if w["e"] <= start or w["s"] >= end:
                continue
            words.append({"w": w["w"], "s": max(0.0, w["s"] - start), "e": min(end, w["e"]) - start})
    return words


def build_ass(cfg: dict, clip: dict, transcript: dict, out: Path) -> Path:
    r = cfg["render"]
    cc = r["captions"]
    cta = r["cta"]
    dur = clip["duration"]
    words = clip_words(transcript, clip["start"], clip["end"])

    lines_out = []
    n = max(1, int(cc["words_per_line"]))
    for i in range(0, len(words), n):
        group = words[i : i + n]
        # Each word gets its own event where it is highlighted; event ends when the next word starts
        # so the line never flickers between words.
        for k, w in enumerate(group):
            s = w["s"]
            e = group[k + 1]["s"] if k + 1 < len(group) else w["e"] + 0.12
            if e - s < 0.04:
                e = s + 0.04
            parts = []
            for j, g in enumerate(group):
                txt = _esc(g["w"]).upper()
                if j == k:
                    parts.append(f"{{\\c{cc['highlight_color']}\\fscx108\\fscy108}}{txt}{{\\c{cc['base_color']}\\fscx100\\fscy100}}")
                else:
                    parts.append(txt)
            lines_out.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Cap,,0,0,0,,{' '.join(parts)}")

    if cc.get("hook_banner", True):
        hook = (clip.get("llm") or {}).get("hook_line") or clip.get("hook_line", "")
        hook = " ".join(hook.split()[:8]).upper()
        if hook:
            lines_out.append(
                f"Dialogue: 1,{_ts(0)},{_ts(min(2.6, dur))},Hook,,0,0,0,,{{\\fad(0,250)}}{_esc(hook)}"
            )

    if cta.get("enabled") and dur > cta["seconds"] + 3:
        text = cta["text"].format(handle=cfg["channel"]["handle"])
        lines_out.append(
            f"Dialogue: 1,{_ts(dur - cta['seconds'])},{_ts(dur)},CTA,,0,0,0,,{{\\fad(200,0)}}{_esc(text)}"
        )

    header = HEADER.format(
        w=r["width"],
        h=r["height"],
        font=cc["font"],
        size=cc["size"],
        hook_size=int(cc["size"] * 0.9),
        cta_size=int(cc["size"] * 0.7),
        base=cc["base_color"],
        outline=cc["outline"],
        margin_v=cc["margin_v"],
        hook_margin=260,
        cta_margin=260,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(lines_out) + "\n", encoding="utf-8")
    return out
