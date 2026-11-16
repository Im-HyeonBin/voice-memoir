"""화면용 그림과 스타일.

목소리 파형 띠가 이 화면의 그림 언어다. 회색 막대는 다른 사람, 초록 막대는 찾는 사람.
맨 위 예시 띠에서 이 규칙을 먼저 보여 주고, 결과에서는 실제 녹음 파형 위에 같은 규칙으로
찾은 구간을 칠한다.
"""
from __future__ import annotations

import html

import numpy as np

MOSS = "#2f5d50"
OTHER = "#cfcbbf"
SAMPLE_IDLE = "#9aa39d"


def envelope(wav: np.ndarray, bars: int) -> np.ndarray:
    """막대 개수만큼 구간을 나눠 소리 크기(0~1)를 구한다."""
    if len(wav) == 0:
        return np.zeros(bars)
    edges = np.linspace(0, len(wav), bars + 1).astype(int)
    rms = np.array([np.sqrt(np.mean(wav[a:b] ** 2)) if b > a else 0.0
                    for a, b in zip(edges[:-1], edges[1:])])
    ref = np.percentile(rms, 98) or 1.0
    return np.clip(np.sqrt(rms / ref), 0.0, 1.0)


def strip_svg(env: np.ndarray, spans: list[tuple[float, float]] = (), height: int = 48,
              on: str = MOSS, off: str = OTHER, label: str = "") -> str:
    """파형 띠. spans는 (시작 비율, 끝 비율) 목록이며 그 안의 막대를 on 색으로 칠한다."""
    n = len(env)
    rects = []
    for i, v in enumerate(env):
        center = (i + 0.5) / n
        color = on if any(a <= center <= b for a, b in spans) else off
        h = max(2.0, float(v) * (height - 4))
        rects.append(f'<rect x="{i * 10 + 2}" y="{(height - h) / 2:.1f}" width="6" '
                     f'height="{h:.1f}" rx="3" fill="{color}"/>')
    return (f'<svg class="vm-strip" viewBox="0 0 {n * 10} {height}" preserveAspectRatio="none" '
            f'style="height:{height}px" role="img" aria-label="{html.escape(label)}">'
            + "".join(rects) + "</svg>")


def hero_strip(bars: int = 150, seed: int = 4) -> str:
    """여러 사람이 번갈아 말하는 녹음 모양의 예시 띠. 몇몇 말 덩어리만 찾는 사람으로 칠한다."""
    rng = np.random.default_rng(seed)
    env = np.zeros(bars)
    spans = []
    i, turn = 1, 0
    while i < bars - 6:
        n = min(int(rng.integers(9, 20)), bars - 1 - i)
        shape = np.sin(np.linspace(0.15, np.pi - 0.15, n)) ** 0.6
        env[i:i + n] = np.clip(shape * (0.45 + 0.55 * rng.random(n)), 0.08, 1.0)
        if turn in (1, 4, 5):
            spans.append((i / bars, (i + n) / bars))
        i += n + int(rng.integers(2, 5))
        turn += 1
    env = np.maximum(env, 0.04)
    return strip_svg(env, spans, height=72,
                     label="여러 사람이 섞인 녹음에서 찾는 사람의 목소리만 초록색으로 칠한 예시")


CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');
@import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&display=swap');

:root {
  --moss: #2f5d50; --moss-tint: #e7efea; --ink: #1f2328; --stone: #5b6168;
  --line: #e3e0d6; --paper: #fbfaf7; --sand: #f1efe8; --card: #fffefb;
  --serif: "Gowun Batang", "AppleMyungjo", "Nanum Myeongjo", serif;
  --sans: "Pretendard Variable", Pretendard, "Apple SD Gothic Neo", sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] { font-family: var(--sans); }
.stApp [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
header[data-testid="stHeader"] { background: transparent; }
/* 한국어는 단어 중간에서 줄을 바꾸지 않는다 */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMarkdownContainer"] div, .stApp [data-testid="stMarkdownContainer"] span {
  word-break: keep-all; overflow-wrap: break-word; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 940px; padding-top: 3rem; padding-bottom: 6rem; }

/* 머리말 */
.stApp .vm-hero h1 { font-family: var(--serif); font-weight: 700; font-size: 3rem; line-height: 1.15;
  letter-spacing: -0.01em; color: var(--ink); margin: 0 0 .7rem; padding: 0; }
.stApp .vm-hero p { font-size: 1.07rem; line-height: 1.8; color: var(--stone); max-width: 34em; margin: 0 0 1.8rem; }
.stApp .vm-hero figure { margin: 0; }
.stApp .vm-strip { width: 100%; display: block; }
.stApp .vm-legend { display: flex; gap: 1.4rem; font-size: .85rem; color: var(--stone); margin-top: .7rem; }
.stApp .vm-legend i { display: inline-block; width: .75rem; height: .75rem; border-radius: 3px;
  margin-right: .45rem; vertical-align: -1px; }

/* 단계 */
.stApp .vm-step { display: flex; align-items: center; gap: .85rem; margin: 3.4rem 0 .5rem; }
.stApp .vm-step-n { font-family: var(--serif); font-weight: 700; font-size: 1rem; width: 2rem; height: 2rem;
  border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
  border: 1.5px solid var(--moss); color: var(--moss); flex: none; }
.stApp .vm-step.done .vm-step-n { background: var(--moss); color: var(--paper); }
.stApp .vm-step.later .vm-step-n { border-color: #c4c0b4; color: #a19d92; }
.stApp .vm-step h2 { font-family: var(--sans); font-weight: 700; font-size: 1.4rem; letter-spacing: -0.01em;
  margin: 0; padding: 0; color: var(--ink); }
.stApp .vm-step.later h2 { color: #a19d92; }
.stApp .vm-lede { color: var(--stone); font-size: .95rem; line-height: 1.75; margin: 0 0 1.1rem 2.85rem; max-width: 38em; }
.stApp .vm-text { color: var(--stone); font-size: .95rem; line-height: 1.75; margin: .2rem 0 .6rem; max-width: 38em; }
[class*="st-key-body-"] { padding-left: 2.85rem; }
.stApp .vm-step.later + .vm-lede { color: #a19d92; }

/* 찾는 사람 고르기 */
[class*="st-key-voice-"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 1rem 1rem .35rem; gap: .5rem; }
.stApp .vm-voice-name { font-family: var(--serif); font-weight: 700; font-size: 1.2rem; color: var(--ink); }
.stApp .vm-voice-meta { font-size: .85rem; color: var(--stone); line-height: 1.55; margin-top: .1rem; }
.stApp .vm-chip { display: inline-block; font-size: .75rem; line-height: 1.6; padding: 0 .6rem; border-radius: 999px;
  background: var(--sand); color: var(--stone); }
.stApp .vm-chip.moss { background: var(--moss-tint); color: var(--moss); }

/* 해마다 */
.stApp .vm-summary { font-size: 1rem; line-height: 1.75; color: var(--ink); margin: 0 0 1.6rem 2.85rem; }
[class*="st-key-year-"] { position: relative; border-left: 2px solid var(--line); margin-left: calc(1rem - 1px);
  padding: 0 0 2.6rem calc(1.85rem - 1px); gap: .65rem; }
[class*="st-key-year-"]::before { content: ""; position: absolute; left: -8px; top: 1rem; width: 14px; height: 14px;
  border-radius: 50%; background: var(--paper); border: 2.5px solid var(--moss); }
.stApp .vm-year { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.stApp .vm-year b { font-family: var(--serif); font-weight: 700; font-size: 2.6rem; line-height: 1.2; color: var(--ink); }
.stApp .vm-year span { color: var(--stone); font-size: .95rem; }
.stApp .vm-chips { display: flex; gap: .45rem; flex-wrap: wrap; }
.stApp .vm-quote { font-family: var(--serif); font-size: 1.12rem; line-height: 1.9; color: var(--ink);
  margin: .2rem 0; max-width: 34em; }
.stApp .vm-quote::before { content: "“"; color: var(--moss); font-size: 1.6rem; line-height: 0; margin-right: .2rem;
  vertical-align: -.35rem; }
.stApp .vm-files-label { font-size: .85rem; color: var(--stone); margin: 1rem 0 .4rem; }
.stApp .vm-file { margin: .1rem 0 .5rem; }
.stApp .vm-file-name { font-size: .9rem; color: var(--ink); }
.stApp .vm-file-meta { font-size: .8rem; color: var(--stone); margin-bottom: .3rem; }

/* 가져온 구간 */
.stApp .vm-sources { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.4rem 1.6rem 1.1rem; margin: 0 0 1.2rem 2.85rem; }
.stApp .vm-sources h3 { font-family: var(--sans); font-weight: 700; font-size: 1.05rem; margin: 0 0 .4rem; padding: 0; color: var(--ink); }
.stApp .vm-sources p { font-size: .9rem; line-height: 1.75; color: var(--stone); max-width: 36em; margin: 0; }
.stApp .vm-sources table { width: 100%; border-collapse: collapse; font-size: .84rem; margin-top: 1rem; border: 0; }
.stApp .vm-sources tr, .stApp .vm-sources th, .stApp .vm-sources td { border: 0; background: none; }
.stApp .vm-sources th { text-align: left; font-weight: 600; color: var(--stone); border-bottom: 1px solid var(--line);
  padding: .45rem .8rem .45rem 0; white-space: nowrap; }
.stApp .vm-sources td { padding: .5rem .8rem .5rem 0; border-bottom: 1px solid #efece4; color: var(--ink); vertical-align: top; }
.stApp .vm-sources td:nth-child(1), .stApp .vm-sources td:nth-child(3), .stApp .vm-sources td:nth-child(4) { white-space: nowrap; }
.stApp .vm-note { font-size: .8rem; color: var(--stone); margin: 0; }
.stApp .vm-review-title { font-weight: 700; font-size: 1.02rem; color: var(--ink); margin-bottom: .2rem; }
.st-key-review-box, .st-key-unsure-box, .st-key-followed-box { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 1.1rem 1.2rem .8rem; margin-bottom: 1.2rem; gap: .5rem; }
.stApp [class*="st-key-review-box"] button p, .stApp [class*="st-key-unsure-box"] button p, .stApp [class*="st-key-followed-box"] button p,
.stApp [class*="st-key-year-"] button p { white-space: nowrap; }
.stApp .vm-year.vm-date b { font-size: 1.9rem; }
.st-key-download-all { margin-left: 2.85rem; }

/* 사이드바 */
[data-testid="stSidebar"] .vm-side-note { font-size: .82rem; line-height: 1.7; color: var(--stone); }

@media (max-width: 640px) {
  .stApp .vm-hero h1 { font-size: 2.3rem; }
  .stApp .vm-lede, .stApp .vm-summary, .stApp .vm-sources, .st-key-download-all { margin-left: 0; }
  [class*="st-key-body-"] { padding-left: 0; }
  [class*="st-key-year-"] { margin-left: .4rem; padding-left: 1.3rem; }
  .stApp .vm-sources td:nth-child(3), .stApp .vm-sources th:nth-child(3) { display: none; }
}
</style>
"""
