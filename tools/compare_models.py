"""화자 지문 모델을 앱의 실제 처리 방식에 넣어 비교한다(데모 세트 + 어려운 세트).

모델마다 점수 크기가 달라 기준값은 tools/calibrate.py와 같은 방법으로 ECAPA 기준값에서 환산한다.
기준은 ECAPA 눈금 0.40/0.45/0.50에 해당하는 값으로 맞춰 비교한다.

    python tools/compare_models.py voxblink2_samresnet34_ft wespeaker_vox_resnet293
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import vm.speakers as S  # noqa: E402
from calibrate import calibrate, korean_crops, mapper, negatives  # noqa: E402
from evaluate import SETS, TARGET, Scorer, owner_share  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import ECAPA_CAL, VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402

SWEEP = [0.40, 0.45, 0.50]  # ECAPA 눈금 기준값 → 모델마다 환산


def run(name: str, m) -> dict:
    eng = VoiceEngine(ROOT / "models", "auto", name)
    eng.cal = calibrate(m)
    base = eng.cal
    ref = audio.decode(ROOT / "demo_sources" / "참고음성_본인_7초.wav")
    out = {}
    for set_name, (library, truth_path) in SETS.items():
        truth = json.loads(truth_path.read_text())
        records = Library(eng, ROOT / ".cache").scan(library)
        cache = audio.AudioCache(max_files=16)
        scorer = Scorer(records, truth)
        clusters = S.discover_speakers(records, eng, cache)
        chosen = [c for c in clusters
                  if (lambda mix: mix[TARGET] / (sum(mix.values()) or 1) > 0.5)(owner_share(records, truth, c, clusters))]
        targets = {"참고": S.target_from_reference(eng, ref, clusters)}
        if chosen:
            targets["자동"] = S.target_from_clusters(chosen, clusters, eng, cache)
        for mode, tgt in targets.items():
            for t in SWEEP:
                thr = m(t)
                eng.cal = replace(base, candidate_margin=thr - m(t - ECAPA_CAL.candidate_margin),
                                  maybe_band=thr - m(t - ECAPA_CAL.maybe_band))
                s = scorer.score(S.search(records, tgt, eng, cache, thr).clips)
                out[(set_name, mode, t)] = (s["found_speech"], s["wrong_sec"])
        eng.cal = base
        out[(set_name, "split")] = len(chosen)
    return out


def main() -> None:
    names = sys.argv[1:]
    crops, bad, spk = korean_crops()
    neg_ref = negatives(VoiceEngine(ROOT / "models", "auto", "ecapa"), crops, bad, spk)
    results = {}
    for name in ["ecapa", *names]:
        neg = neg_ref if name == "ecapa" else negatives(VoiceEngine(ROOT / "models", "auto", name), crops, bad, spk)
        m = mapper(neg_ref, neg)
        print(f"{name}: 기준 0.45 → {m(0.45):.3f}", file=sys.stderr, flush=True)
        results[name] = run(name, m)

    for mode in ("참고", "자동"):
        print(f"\n### {mode} 방식 — 찾은 비율(말소리) / 잘못 담은 시간 (기준은 ECAPA 눈금으로 환산)")
        print(f"{'모델':28s} {'기준':>5s} | {'데모':>14s} | {'어려운 세트':>14s}")
        for name, r in results.items():
            for t in SWEEP:
                d, h = r.get(("demo", mode, t)), r.get(("hard", mode, t))
                if d and h:
                    print(f"{name:28s} {t:5.2f} | {d[0]:5.0%} {d[1]:5.1f}초 | {h[0]:5.0%} {h[1]:5.1f}초")
        print()
    print("찾을 사람이 나뉜 목소리 수(어려운 세트): " +
          ", ".join(f"{n} {r[('hard', 'split')]}개" for n, r in results.items()))


if __name__ == "__main__":
    main()
