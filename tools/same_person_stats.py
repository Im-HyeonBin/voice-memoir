"""참고 녹음과 폴더 속 목소리(군집)의 닮은 정도를, 실제로 같은 사람인지 다른 사람인지 나눠 본다.

참고 녹음 방식에서 '확실히 같은 사람'으로 볼 기준(same_person_sure)을 정할 때 쓴다.
다른 사람 군집이 이 기준을 넘으면 그 사람 목소리가 담기고, 찾을 사람 군집이 이 기준에 못 미치면
화면에서 '같은 사람인가요?'로 물어본다.

    python tools/same_person_stats.py voxblink2_samresnet34_ft ecapa
    python tools/same_person_stats.py voxblink2_samresnet34_ko --write   # 학습한 모델의 calibration.json에 기준을 남긴다
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import SETS, TARGET, owner_share  # noqa: E402
from make_cond_benchmark import OUT  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import _sim, discover_speakers, target_from_reference  # noqa: E402


def libraries():
    cond = json.loads((ROOT / "bench" / "cond_truth.json").read_text())
    for folder, truth in cond.items():
        yield OUT / folder, truth, OUT / f"{folder}_참고.wav"
    for lib, tp in SETS.values():
        yield lib, json.loads(tp.read_text()), ROOT / "demo_sources" / "참고음성_본인_7초.wav"


def main() -> None:
    write = "--write" in sys.argv
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        eng = VoiceEngine(ROOT / "models", "auto", name)
        same, other = [], []
        for lib, truth, ref in libraries():
            records = Library(eng, ROOT / ".cache").scan(lib)
            cache = audio.AudioCache(max_files=8)
            clusters = discover_speakers(records, eng, cache)
            t = target_from_reference(eng, audio.decode(ref))
            protos = np.vstack([t.wide, t.narrow])
            for c in clusters:
                mix = owner_share(records, truth, c, clusters)
                share = mix[TARGET] / (sum(mix.values()) or 1)
                s = float(_sim(c.centroid[None], protos)[0])
                (same if share > 0.5 else other).append((round(s, 3), f"{lib.name}#{c.index}"))
        same.sort()
        other.sort(reverse=True)
        cal = eng.cal
        print(f"\n== {name}  (지금 기준: 다른 사람 < {cal.same_person:.3f} ≤ 애매 < {cal.same_person_sure:.3f} ≤ 같은 사람)")
        print(f"찾을 사람 군집 {len(same)}개, 낮은 쪽부터: " + ", ".join(f"{s:.2f}" for s, _ in same[:12]))
        print(f"다른 사람 군집 {len(other)}개, 높은 쪽부터: " + ", ".join(f"{s:.2f}({w})" for s, w in other[:8]))
        top_other = other[0][0] if other else 0.0
        print(f"다른 사람 군집의 최댓값 {top_other:.3f} → 이보다 높은 찾을 사람 군집 "
              f"{sum(s > top_other for s, _ in same)}/{len(same)}개")
        cal_file = ROOT / "models" / "alt" / name / "calibration.json"
        if write and cal_file.exists():  # 다른 사람 최댓값보다 조금 위를 '확실히 같은 사람' 기준으로
            d = json.loads(cal_file.read_text())
            d["same_person_sure"] = round(top_other + 0.02, 2)
            cal_file.write_text(json.dumps(d, indent=1))
            print(f"기준 저장: same_person_sure = {d['same_person_sure']}")


if __name__ == "__main__":
    main()
