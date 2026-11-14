"""겹쳐 말하기 시험 세트(bench/overlap): 다른 사람과 겹친 찾을 사람 목소리를 앱 전체가 얼마나 찾는지.

Zeroth-Korean 테스트 화자 10명(화자 지문·대상 화자 추출 학습 모두에 안 씀)을 한 명씩 찾을 사람으로 두고,
사람마다 폴더 하나에 녹음 4개를 만든다.
- 1 조용한 대화: 찾을 사람과 다른 두 사람이 겹치지 않게 번갈아 말함(폴더 속 목소리를 묶을 깨끗한 말소리)
- 2 겹친 대화: 찾을 사람이 말할 때마다 다른 사람이 절반쯤 겹치거나 완전히 동시에 말함(같은 세기)
- 3 작게 겹친 대화: 2와 같되 찾을 사람이 5dB 작음
- 4 함정 겹침: 찾을 사람 없이 다른 두 사람이 겹쳐 말함
참고 녹음(사람NN_참고.wav)은 폴더 안 녹음과 다른 문장이다. 정답은 bench/overlap_truth.json.

    python tools/make_overlap_benchmark.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from korean_sv import load  # noqa: E402
from make_cond_benchmark import encode  # noqa: E402
from vm import audio  # noqa: E402

SR = audio.SR
OUT = ROOT / "bench" / "overlap"
TARGET = "찾을사람"
FILES = ["1 조용한 대화.m4a", "2 겹친 대화.m4a", "3 작게 겹친 대화.m4a", "4 함정 겹침.m4a"]


def scene(events: list[tuple[str, np.ndarray, float, float]]) -> tuple[np.ndarray, list[dict]]:
    """events: (누구, 소리, 시작초, 크기dB)."""
    end = max(t + len(x) / SR for _, x, t, _ in events) + 0.5
    out = np.zeros(int(end * SR), np.float32)
    spans = []
    for who, x, t, g in events:
        x = C.level(x, 0.05 * 10 ** (g / 20))
        a = int(t * SR)
        out[a:a + len(x)] += x
        spans.append({"speaker": who, "start": round(t, 3), "end": round(t + len(x) / SR, 3)})
    return out, spans


def main() -> None:
    utts, spk = load()
    speakers = sorted(set(spk.tolist()))
    by = {s: [utts[i] for i in np.where(spk == s)[0]] for s in speakers}
    if OUT.exists():
        shutil.rmtree(OUT)
    truth = {}
    for k, s in enumerate(speakers):
        rng = np.random.default_rng(500 + k)
        a, b = speakers[(k + 3) % len(speakers)], speakers[(k + 4) % len(speakers)]
        pools = {TARGET: list(by[s]), "A": list(by[a]), "B": list(by[b])}
        for v in pools.values():
            rng.shuffle(v)
        take = lambda who: pools[who].pop() if pools[who] else by[{"A": a, "B": b, TARGET: s}[who]][0]  # noqa: E731
        folder = OUT / f"사람{k + 1:02d}"
        folder.mkdir(parents=True)
        ref = []
        while sum(len(x) for x in ref) < 7 * SR:
            ref.append(take(TARGET))
        audio.write_wav(OUT / f"사람{k + 1:02d}_참고.wav", C.level(np.concatenate(ref)))
        truth[folder.name] = {}

        # 1 조용한 대화: 번갈아 말하기
        ev, t = [], 0.5
        for who in (TARGET, "A", TARGET, "B", TARGET, "A"):
            x = take(who)
            ev.append((who, x, t, rng.uniform(-2, 2)))
            t += len(x) / SR + rng.uniform(0.3, 0.8)
        files = {FILES[0]: scene(ev)}

        # 2·3 겹친 대화: 다른 사람 말 중간에 찾을 사람이 끼어들거나, 완전히 동시에 말함
        for name, gain in ((FILES[1], 0.0), (FILES[2], -5.0)):
            ev, t = [], 0.5
            for j, other in enumerate(("A", "B", "A", "B")):
                o, x = take(other), take(TARGET)
                ev.append((other, o, t, 0.0))
                start = t if j % 2 else t + 0.5 * len(o) / SR  # 홀수 번째는 동시에, 짝수 번째는 절반쯤 겹침
                ev.append((TARGET, x, start, gain))
                t = max(t + len(o) / SR, start + len(x) / SR) + rng.uniform(0.3, 0.8)
            files[name] = scene(ev)

        # 4 함정: 찾을 사람 없이 둘이 겹쳐 말함
        ev, t = [], 0.5
        for _ in range(3):
            o1, o2 = take("A"), take("B")
            ev += [("A", o1, t, 0.0), ("B", o2, t + 0.5 * len(o1) / SR, 0.0)]
            t += max(len(o1) / SR, 0.5 * len(o1) / SR + len(o2) / SR) + 0.6
        files[FILES[3]] = scene(ev)

        for name, (x, spans) in files.items():
            encode(x, folder / name)
            truth[folder.name][name] = spans
        print(f"{folder.name}: 찾을 사람 {s}, 다른 사람 {a}·{b}", flush=True)
    (ROOT / "bench" / "overlap_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
