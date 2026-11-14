"""연도별 타임라인, '올해의 목소리' 하이라이트, 가져온 구간 목록."""
from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import audio
from .audio import AudioCache
from .speakers import Clip

GAP_SEC = 0.45   # 하이라이트 조각 사이 쉼
FADE_SEC = 0.03


@dataclass
class YearSummary:
    year: int | None
    clips: list[Clip]
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    date_sources: Counter = field(default_factory=Counter)

    @property
    def label(self) -> str:
        return f"{self.year}년" if self.year else "날짜 모름"


def build_timeline(clips: list[Clip]) -> list[YearSummary]:
    by_year: dict[int | None, list[Clip]] = {}
    for c in clips:
        by_year.setdefault(c.year, []).append(c)
    years = []
    for y, cs in by_year.items():
        cs.sort(key=lambda c: (c.date or datetime.min.date(), c.path, c.start))
        years.append(YearSummary(
            year=y, clips=cs, seconds=float(sum(c.duration for c in cs)),
            files=sorted({c.path for c in cs}),
            date_sources=Counter(c.date_source for c in {c.path: c for c in cs}.values()),
        ))
    years.sort(key=lambda s: (s.year is None, s.year or 0))
    return years


@dataclass
class Selection:
    clip: Clip
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def select_highlights(clips: list[Clip], max_total: float = 30.0,
                      max_each: float = 8.0, min_each: float = 1.5) -> list[Selection]:
    """가장 확실한 구간부터 골라 30초 안팎으로 모으고, 시간 순서로 다시 놓는다."""
    pool = sorted((c for c in clips if c.duration >= min_each),
                  key=lambda c: (round(c.score, 2), c.duration), reverse=True)
    picked: list[Selection] = []
    total = 0.0
    for c in pool:
        if total >= max_total:
            break
        take = min(c.duration, max_each, max_total - total + 1.0)
        mid = (c.start + c.end) / 2
        s = max(c.start, mid - take / 2)
        picked.append(Selection(c, s, s + take))
        total += take
    picked.sort(key=lambda p: (p.clip.date or datetime.min.date(), p.clip.path, p.start))
    return picked


def all_selections(clips: list[Clip]) -> list[Selection]:
    """찾은 구간 전부를 시간 순서로(한 해 전부 듣기, 날짜별 듣기)."""
    ordered = sorted(clips, key=lambda c: (c.date or datetime.min.date(), c.path, c.start))
    return [Selection(c, c.start, c.end) for c in ordered]


def _fade(x: np.ndarray) -> np.ndarray:
    n = min(int(FADE_SEC * audio.SR), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def compile_highlight(selections: list[Selection], cache: AudioCache,
                      target_rms_db: float = -20.0) -> np.ndarray:
    if not selections:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(GAP_SEC * audio.SR), dtype=np.float32)
    parts = []
    for i, sel in enumerate(selections):
        seg = cache.slice(sel.clip.path, sel.start, sel.end)
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
        seg = seg * (10 ** (target_rms_db / 20) / rms)  # 조각마다 크기를 맞춘다
        parts.append(_fade(seg.astype(np.float32)))
        if i < len(selections) - 1:
            parts.append(gap)
    out = np.concatenate(parts)
    peak = np.abs(out).max()
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def source_rows(year: YearSummary, selections: list[Selection],
                library_root: str | Path | None = None) -> list[dict]:
    """하이라이트의 각 조각이 어느 녹음의 몇 초에서 왔는지 적은 목록(원본을 다시 찾아 들을 때 쓴다)."""
    root = Path(library_root).resolve() if library_root else None
    rows = []
    for i, s in enumerate(selections, 1):
        p = Path(s.clip.path).resolve()
        rows.append({
            "해": year.label,
            "순서": i,
            "녹음 파일": str(p.relative_to(root)) if root and p.is_relative_to(root) else p.name,
            "녹음 날짜": s.clip.date.isoformat() if s.clip.date else "",
            "날짜 근거": audio.DATE_SOURCE_LABEL[s.clip.date_source],
            "시작(초)": round(s.start, 1),
            "끝(초)": round(s.end, 1),
        })
    return rows


def sources_csv(rows: list[dict]) -> bytes:
    """엑셀에서 한글이 깨지지 않도록 BOM을 붙인 CSV."""
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def export_year(year: YearSummary, selections: list[Selection], cache: AudioCache,
                person: str, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{person}_{year.year or '날짜모름'}_올해의목소리"
    wav = compile_highlight(selections, cache)
    paths = {"wav": out_dir / f"{stem}.wav", "m4a": out_dir / f"{stem}.m4a"}
    audio.write_wav(paths["wav"], wav)
    audio.encode_m4a(wav, paths["m4a"], metadata={"title": f"{person} {year.label}의 목소리"})
    return paths
