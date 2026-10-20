"""Caption overlay rendering (word-highlight captions, hook banner, end CTA).

Renders RGBA frames with Pillow and streams them to ffmpeg as a second input, so there is
no dependency on libass/freetype being compiled into ffmpeg. Frames are cached per unique
visual state (~1 per word), so a 45s clip costs ~150 renders, not 1350.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageDraw, ImageFont

from .common import log

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def resolve_font(cfg_font: str | None) -> str:
    if cfg_font and Path(cfg_font).exists():
        return cfg_font
    for c in FONT_CANDIDATES:
        if Path(c).exists():
            if cfg_font:
                log.warning("Caption font %s not found; using %s", cfg_font, c)
            return c
    raise SystemExit("No usable caption font found; set render.captions.font_file in config.yaml")


def _hex_rgb(s: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    s = (s or "").lstrip("#")
    if len(s) != 6:
        return default
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass(frozen=True)
class State:
    line: tuple[str, ...]
    hi: int  # highlighted word index, -1 for none
    hook: str | None
    cta: str | None


def clip_words(transcript: dict, start: float, end: float) -> list[dict]:
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words") or []:
            if w["e"] <= start or w["s"] >= end:
                continue
            words.append({"w": w["w"], "s": max(0.0, w["s"] - start), "e": min(end, w["e"]) - start})
    return words


def build_timeline(cfg: dict, clip: dict, transcript: dict) -> list[tuple[float, float, State]]:
    """Ordered (start, end, State) spans covering the whole clip (gaps get an empty state)."""
    r = cfg["render"]
    cc = r["captions"]
    cta_cfg = r["cta"]
    dur = clip["duration"]
    words = clip_words(transcript, clip["start"], clip["end"])
    n = max(1, int(cc["words_per_line"]))

    hook_until = 0.0
    hook_text = None
    if cc.get("hook_banner", True):
        hook = (clip.get("llm") or {}).get("hook_line") or clip.get("hook_line", "")
        hook = " ".join(str(hook).split()[:8]).upper().strip()
        if hook:
            hook_text, hook_until = hook, min(2.6, dur)
    cta_from = dur + 1
    cta_text = None
    if cta_cfg.get("enabled") and dur > cta_cfg["seconds"] + 3:
        cta_text = cta_cfg["text"].format(handle=cfg["channel"]["handle"])
        cta_from = dur - cta_cfg["seconds"]

    # Word-level spans: (start, end, line, highlighted index)
    spans: list[tuple[float, float, tuple[str, ...], int]] = []
    for i in range(0, len(words), n):
        group = words[i : i + n]
        line = tuple(w["w"].upper() for w in group)
        for k, w in enumerate(group):
            s = w["s"]
            e = group[k + 1]["s"] if k + 1 < len(group) else w["e"] + 0.12
            e = max(e, s + 0.04)
            spans.append((s, min(e, dur), line, k))

    # Split spans on the hook/CTA boundaries so every span has one constant State.
    cuts = sorted({0.0, dur, hook_until, cta_from} | {s for s, _, _, _ in spans} | {e for _, e, _, _ in spans})
    cuts = [c for c in cuts if 0.0 <= c <= dur]
    timeline: list[tuple[float, float, State]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-4:
            continue
        mid = (a + b) / 2
        line, hi = (), -1
        for s, e, ln, k in spans:
            if s <= mid < e:
                line, hi = ln, k
                break
        state = State(line, hi, hook_text if mid < hook_until else None, cta_text if mid >= cta_from else None)
        if timeline and timeline[-1][2] == state:
            timeline[-1] = (timeline[-1][0], b, state)
        else:
            timeline.append((a, b, state))
    return timeline


class CaptionRenderer:
    def __init__(self, cfg: dict):
        r = cfg["render"]
        cc = r["captions"]
        self.W, self.H = r["width"], r["height"]
        font_path = resolve_font(cc.get("font_file"))
        self.size = int(cc["size"])
        self.font = ImageFont.truetype(font_path, self.size)
        self.font_hi = ImageFont.truetype(font_path, int(self.size * 1.1))
        self.font_hook = ImageFont.truetype(font_path, int(self.size * 0.82))
        self.font_cta = ImageFont.truetype(font_path, int(self.size * 0.6))
        self.base = _hex_rgb(cc.get("base_color_hex"), (255, 255, 255))
        self.hi = _hex_rgb(cc.get("highlight_color_hex"), (255, 221, 0))
        self.outline = int(cc["outline"])
        self.margin_v = int(cc["margin_v"])
        self.blank = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0)).tobytes()

    # -- drawing helpers -------------------------------------------------
    def _text(self, d: ImageDraw.ImageDraw, xy, text, font, fill):
        x, y = xy
        d.text((x + 4, y + 6), text, font=font, fill=(0, 0, 0, 160), stroke_width=self.outline, stroke_fill=(0, 0, 0, 160))
        d.text((x, y), text, font=font, fill=fill + (255,), stroke_width=self.outline, stroke_fill=(0, 0, 0, 255))

    def _wrap(self, text: str, font, max_w: int) -> list[str]:
        lines, cur = [], ""
        for w in text.split():
            t = (cur + " " + w).strip()
            if font.getlength(t) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _caption_line(self, d, line: tuple[str, ...], hi: int):
        gap = self.size * 0.32
        pieces = []
        total = 0.0
        for i, w in enumerate(line):
            f = self.font_hi if i == hi else self.font
            wl = f.getlength(w)
            pieces.append((w, f, wl))
            total += wl + (gap if i else 0)
        max_w = self.W - 120
        if total > max_w:  # long words: shrink everything for this line
            scale = max_w / total
            pieces = [(w, ImageFont.truetype(f.path, int(f.size * scale)), f.getlength(w) * scale) for w, f, _ in pieces]
            total = max_w
        x = (self.W - total) / 2
        base_y = self.H - self.margin_v - self.size
        for i, (w, f, wl) in enumerate(pieces):
            y = base_y - (f.size - self.size) / 2
            self._text(d, (x, y), w, f, self.hi if i == hi else self.base)
            x += wl + gap

    def _banner(self, d, text: str, font, y_top: int, pill: bool):
        lines = self._wrap(text, font, self.W - 160)
        lh = int(font.size * 1.18)
        h = lh * len(lines) + 40
        if pill:
            w = max(font.getlength(ln) for ln in lines) + 90
            x0 = (self.W - w) / 2
            d.rounded_rectangle((x0, y_top, x0 + w, y_top + h), radius=34, fill=(0, 0, 0, 150))
        y = y_top + 20
        for ln in lines:
            x = (self.W - font.getlength(ln)) / 2
            self._text(d, (x, y), ln, font, self.base)
            y += lh

    # -- public --------------------------------------------------------------
    @lru_cache(maxsize=512)
    def frame(self, state: State) -> bytes:
        if not state.line and not state.hook and not state.cta:
            return self.blank
        img = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if state.hook:
            self._banner(d, state.hook, self.font_hook, 250, pill=True)
        if state.cta:
            self._banner(d, state.cta, self.font_cta, 250, pill=True)
        if state.line:
            self._caption_line(d, state.line, state.hi)
        return img.tobytes()

    def stream(self, timeline: list[tuple[float, float, State]], fps: int, duration: float, out: BinaryIO) -> int:
        n_frames = int(round(duration * fps))
        idx = 0
        written = 0
        for i in range(n_frames):
            t = (i + 0.5) / fps
            while idx < len(timeline) - 1 and t >= timeline[idx][1]:
                idx += 1
            state = timeline[idx][2] if timeline and timeline[idx][0] <= t < timeline[idx][1] else State((), -1, None, None)
            out.write(self.frame(state))
            written += 1
        return written
