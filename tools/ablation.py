"""엔진 설정끼리 정확도를 나란히 비교한다(데모 세트 + 어려운 세트).

    python tools/ablation.py [모델 이름]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import SETS, TARGET, Scorer, owner_share  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import DEFAULT_MODEL, VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import (discover_speakers, find_clips, target_from_clusters,  # noqa: E402
                         target_from_reference)

CONFIGS = [  # 이름, 조건 지문, 경쟁자, 경쟁자 여유(모델 기본값의 몇 배)
    ("기존", False, False, 0.0),
    ("조건 지문", True, False, 0.0),
    ("경쟁자", False, True, 1.0),
    ("둘 다 여유0", True, True, 0.0),
    ("둘 다 여유1", True, True, 1.0),
    ("둘 다 여유2", True, True, 2.0),
]


def main() -> None:
    engine = VoiceEngine(ROOT / "models", device="auto", model=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    thresholds = list(engine.cal.levels)
    ref_wav = audio.decode(ROOT / "demo_sources" / "참고음성_본인_7초.wav")
    cache = audio.AudioCache(max_files=16)
    results = {}
    for set_name, (library, truth_path) in SETS.items():
        truth = json.loads(truth_path.read_text())
        records = Library(engine, ROOT / ".cache").scan(library)
        scorer = Scorer(records, truth)
        clusters = discover_speakers(records, engine, cache)
        chosen = []
        for c in clusters:
            mix = owner_share(records, truth, c, clusters)
            if mix[TARGET] / (sum(mix.values()) or 1) > 0.5:
                chosen.append(c)
        for mode in ("참고", "자동"):
            for name, aug, riv, scale in CONFIGS:
                if mode == "참고":
                    tgt = target_from_reference(engine, ref_wav, clusters if riv else None, augment=aug)
                else:
                    tgt = target_from_clusters(chosen, clusters if riv else None, engine, cache, augment=aug)
                tgt.margin = engine.cal.rival_margin * scale
                for thr in thresholds:
                    s = scorer.score(find_clips(records, tgt, engine, cache, thr))
                    results[(set_name, mode, name, thr)] = (s["found_speech"], s["wrong_sec"])
                print(f"  {set_name} {mode} {name} 완료", file=sys.stderr)

    for mode in ("참고", "자동"):
        print(f"\n### {mode} 방식 — 찾은 비율(말소리) / 잘못 담은 시간")
        print(f"{'설정':12s} {'기준':>5s} | {'데모':>14s} | {'어려운 세트':>14s}")
        for name, *_ in CONFIGS:
            for thr in thresholds:
                d = results[("demo", mode, name, thr)]
                h = results[("hard", mode, name, thr)]
                print(f"{name:12s} {thr:5.2f} | {d[0]:5.0%} {d[1]:5.1f}초 | {h[0]:5.0%} {h[1]:5.1f}초")
            print()


if __name__ == "__main__":
    main()
