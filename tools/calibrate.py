"""새 화자 지문 모델의 기준값을 ECAPA 기준값에서 환산한다.

모델마다 '닮은 정도' 숫자의 크기가 달라 ECAPA 기준값(0.45 등)을 그대로 쓸 수 없다.
한국어 문장 2초 조각(깨끗한 것끼리, 깨끗한 것과 나쁜 음질끼리)에서 다른 사람끼리
ECAPA로 t 이상이 나오는 비율을 구하고, 새 모델에서 같은 비율이 되는 점수를 t의 짝으로 삼는다.

    python tools/calibrate.py voxblink2_samresnet34_ft   # vm/engine.py MODELS에 붙여 넣을 값을 출력
    python tools/calibrate.py voxblink2_samresnet34_ko_l3   # 학습한 모델은 models/alt/<이름>/calibration.json에도 남긴다
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from korean_sv import CROP, degrade, load  # noqa: E402
from vm.audio import SR  # noqa: E402
from vm.engine import ECAPA_CAL, Calibration, VoiceEngine  # noqa: E402


def korean_crops() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    utts, spk = load()
    crops = [x[(len(x) - int(CROP * SR)) // 2:][: int(CROP * SR)] for x in utts]
    return crops, [degrade(c, i) for i, c in enumerate(crops)], spk


def negatives(eng: VoiceEngine, crops, bad, spk) -> np.ndarray:
    """다른 사람끼리의 닮은 정도 점수들."""
    a = np.concatenate([eng.embed(x, np.array([[0, CROP]], np.float32)) for x in crops])
    b = np.concatenate([eng.embed(x, np.array([[0, CROP]], np.float32)) for x in bad])
    diff = spk[:, None] != spk[None, :]
    return np.concatenate([(a @ a.T)[diff], (a @ b.T)[diff]])


def mapper(neg_ref: np.ndarray, neg_model: np.ndarray):
    """ECAPA 점수 t → 다른 사람끼리 t 이상이 나오는 비율이 같은 모델 점수."""
    def m(t: float) -> float:
        rate = float((neg_ref >= t).mean())
        return float(np.quantile(neg_model, 1 - rate)) if rate > 0 else float(neg_model.max())
    return m


def calibrate(m) -> Calibration:
    c, thr = ECAPA_CAL, ECAPA_CAL.threshold
    return Calibration(
        levels=tuple(round(m(t), 3) for t in c.levels),
        cluster_distance=round(1 - m(1 - c.cluster_distance), 3),
        link=round(m(c.link), 3),
        same_person=round(m(c.same_person), 3),
        candidate_margin=round(m(thr) - m(thr - c.candidate_margin), 3),
        candidate_floor=round(m(c.candidate_floor), 3),
        maybe_band=round(m(thr) - m(thr - c.maybe_band), 3),
        rival_margin=round(m(thr) - m(thr - c.rival_margin), 3),
    )


def main() -> None:
    import json
    from dataclasses import asdict, replace
    crops, bad, spk = korean_crops()
    neg_ref = negatives(VoiceEngine(ROOT / "models", "auto", "ecapa"), crops, bad, spk)
    for name in sys.argv[1:]:
        eng = VoiceEngine(ROOT / "models", "auto", name, require_cal=False)  # 기준값은 지금 만든다
        cal = calibrate(mapper(neg_ref, negatives(eng, crops, bad, spk)))
        if name.startswith("ecapa"):
            # ECAPA 계열은 묶음 합치기·확실히 같은 사람 따라가기를 켜면 합성 세트에서 다른 사람까지 이어 붙었다
            # (bench/server/ecapa_ko_merge_chain_on_*.txt). 기본 모델에서만 켠다.
            cal = replace(cal, merge=False, chain=False)
        print(name, cal, flush=True)
        folder = ROOT / "models" / "alt" / name
        if folder.exists():  # 앱·시험 도구가 이 모델을 쓸 때 읽는다
            (folder / "calibration.json").write_text(json.dumps(asdict(cal), indent=1))


if __name__ == "__main__":
    main()
