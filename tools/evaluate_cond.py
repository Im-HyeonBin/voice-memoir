"""조건별 시험 세트(bench/cond)로 앱 전체 정확도를 잰다.

사람 10명 폴더마다 앱과 똑같이 스캔 → 목소리 자동 발견 → 참고 녹음 또는 자동 발견으로 찾기를 하고,
녹음 조건(파일)별로 10명 결과를 더한다.
- 찾은 비율: 찾을 사람이 말한 시간(말소리) 중 찾아낸 시간
- 잘못 담은 시간: 찾을 사람이 말하지 않는 곳을 담은 시간(10명 합)

    python tools/evaluate_cond.py voxblink2_samresnet34_ft ecapa
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import TARGET, Scorer, owner_share  # noqa: E402
from make_cond_benchmark import FILES, OUT  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import discover_speakers, search, target_from_clusters, target_from_reference  # noqa: E402


def run(model: str, device: str = "auto", level: int = 2) -> dict:
    eng = VoiceEngine(ROOT / "models", device, model)
    thr = eng.cal.levels[level]
    truth_all = json.loads((ROOT / "bench" / "cond_truth.json").read_text())
    tot = defaultdict(lambda: defaultdict(float))
    split, asked = 0, {}
    for folder, truth in truth_all.items():
        records = Library(eng, ROOT / ".cache").scan(OUT / folder)
        cache = audio.AudioCache(max_files=8)
        scorer = Scorer(records, truth)
        clusters = discover_speakers(records, eng, cache)
        chosen = [c for c in clusters
                  if (lambda m: m[TARGET] / (sum(m.values()) or 1) > 0.5)(owner_share(records, truth, c, clusters))]
        split += len(chosen)
        ref = audio.decode(OUT / f"{folder}_참고.wav")
        targets = {"참고": target_from_reference(eng, ref, clusters)}
        # 참고 + 확인: 애매한 목소리를 사람이 들어 보고 맞게 답했다고 가정
        unsure = targets["참고"].info["unsure"]
        same = {c.index for c in chosen if c.index in unsure}
        targets["참고+확인"] = target_from_reference(eng, ref, clusters, same=same)
        asked[folder] = len(unsure)
        if chosen:
            targets["자동"] = target_from_clusters(chosen, clusters, eng, cache)
        for mode, tgt in targets.items():
            s = scorer.score(search(records, tgt, eng, cache, thr).clips)
            for fname, row in s["per_file"].items():
                for k in ("target_speech", "found_speech", "wrong"):
                    tot[(mode, fname)][k] += row[k]
    return {"rows": tot, "split": split / len(truth_all), "asked": sum(asked.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--level", type=int, default=2, help="담는 기준 단계 0~4(넉넉하게…엄격하게), 2가 보통")
    args = ap.parse_args()
    results = {m: run(m, args.device, args.level) for m in args.models}
    print(f"기준 단계 {args.level} (0 넉넉하게 … 2 보통 … 4 엄격하게)")
    names = [f[0] for f in FILES]
    for mode in ("참고", "참고+확인", "자동"):
        print(f"\n### {mode} 방식 — 찾은 비율(말소리) / 잘못 담은 시간 (사람 10명 합)")
        print(f"{'조건':14s} " + " ".join(f"{m[:24]:>24s}" for m in args.models))
        for fname in names:
            cells = []
            for m in args.models:
                r = results[m]["rows"][(mode, fname)]
                rate = f"{r['found_speech'] / r['target_speech']:4.0%}" if r["target_speech"] else "   -"
                cells.append(f"{rate} {r['wrong']:5.1f}초")
            print(f"{fname[:-4]:14s} " + " ".join(f"{c:>24s}" for c in cells))
        cells = []
        for m in args.models:
            rows = [results[m]["rows"][(mode, f)] for f in names]
            ts, fs, w = (sum(r[k] for r in rows) for k in ("target_speech", "found_speech", "wrong"))
            cells.append(f"{fs / ts:4.0%} {w:5.1f}초")
        print(f"{'합계':14s} " + " ".join(f"{c:>24s}" for c in cells))
    print("\n찾을 사람이 나뉜 목소리 수(평균): " + ", ".join(f"{m} {r['split']:.1f}개" for m, r in results.items()))
    print("참고 방식에서 '같은 사람인가요?'로 물은 목소리(10명 합): " + ", ".join(f"{m} {r['asked']}개" for m, r in results.items()))


if __name__ == "__main__":
    main()
