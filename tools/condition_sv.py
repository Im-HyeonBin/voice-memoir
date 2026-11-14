"""시끄럽거나 멀리서 울리는 녹음에서 화자 지문 모델이 얼마나 헷갈리는지 잰다(학습에 안 쓴 소리로).

깨끗한 2초 조각 하나와, 다른 문장의 2초 조각에 조건을 입힌 것을 짝지어 같은 사람·다른 사람을
헷갈린 비율(EER, 낮을수록 좋음)을 조건별로 구한다. 소음은 DEMAND 실제 녹음, 울림은 실제로 잰 방 울림이다.
화자: Zeroth-Korean 테스트 10명(낭독), Pansori-TEDxKR 41명(실제 강연, 다른 녹음 환경).

    python tools/condition_sv.py voxblink2_samresnet34_ft ecapa
    python tools/condition_sv.py voxblink2_samresnet34_ft --wpe     # 울림 걷어내기를 먼저 하고 재기
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from korean_sv import CROP, eer, load  # noqa: E402
from vm.audio import SR  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402

CONDS = [  # 이름, 울림, 소음 환경, SNR
    ("카페 5dB", False, "카페", 5),
    ("식당 0dB", False, "식당", 0),
    ("거실 5dB", False, "거실", 5),
    ("버스 5dB", False, "버스", 5),
    ("멀리서", True, None, None),
    ("멀리서+광장 5dB", True, "광장", 5),
]


def center(x: np.ndarray) -> tuple[int, int]:
    a = (len(x) - int(CROP * SR)) // 2
    return a, a + int(CROP * SR)


def degrade(x: np.ndarray, i: int, reverb: bool, env: str | None, snr: float | None, wpe: bool) -> np.ndarray:
    rng = np.random.default_rng(1000 + i)
    y = x
    if reverb:
        rirs = C.rir_paths("test")
        y = C.apply_rir(x, C.load_rir(rirs[int(rng.integers(len(rirs)))]))
    if env:
        chans = C.demand(env)
        y = C.add_noise(y, chans[int(rng.integers(len(chans)))], snr, rng)
    if wpe and reverb:
        from vm.enhance import dereverb
        y = dereverb(y)
    a, b = center(x)
    return y[a:b]


def pansori() -> tuple[list[np.ndarray], np.ndarray]:
    from pansori import load as load_pansori
    return load_pansori()


def run(name: str, sets: dict, wpe: bool) -> dict:
    eng = VoiceEngine(ROOT / "models", "auto", name)
    win = np.array([[0, CROP]], np.float32)
    emb = lambda xs: np.concatenate([eng.embed(x, win) for x in xs])  # noqa: E731
    out = {}
    for set_name, (utts, spk) in sets.items():
        clean = emb([x[slice(*center(x))] for x in utts])
        same, mask = spk[:, None] == spk[None, :], np.triu(np.ones((len(spk),) * 2, bool), 1)
        out[(set_name, "깨끗함")] = eer((clean @ clean.T)[same & mask], (clean @ clean.T)[~same & mask])
        off = ~np.eye(len(spk), dtype=bool)
        for cname, rev, env, snr in CONDS:
            deg = emb([degrade(x, i, rev, env, snr, wpe) for i, x in enumerate(utts)])
            s = clean @ deg.T
            out[(set_name, cname)] = eer(s[same & off], s[~same & off])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--wpe", action="store_true")
    ap.add_argument("--no-pansori", action="store_true")
    args = ap.parse_args()
    sets = {"Zeroth": load()}
    if not args.no_pansori:
        sets["Pansori"] = pansori()
    for k, (u, s) in sets.items():
        print(f"{k}: {len(set(s.tolist()))}명, 문장 {len(u)}개")
    names = ["깨끗함"] + [c[0] for c in CONDS]
    print("\n같은 사람·다른 사람을 헷갈린 비율(EER, 낮을수록 좋음). 깨끗한 2초 조각 ↔ 조건을 입힌 2초 조각"
          + (" · 울림 걷어내기 적용" if args.wpe else ""))
    print(f"{'모델':28s} {'세트':8s} " + " ".join(f"{n:>13s}" for n in names) + "  처리 시간")
    for m in args.models:
        t = time.time()
        r = run(m, sets, args.wpe)
        for set_name in sets:
            print(f"{m:28s} {set_name:8s} " + " ".join(f"{r[(set_name, n)]:13.1%}" for n in names)
                  + f"  {time.time() - t:5.0f}s", flush=True)


if __name__ == "__main__":
    main()
