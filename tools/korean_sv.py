"""한국어 화자 구분 시험: 화자 지문 모델끼리 비교한다(학습 없음).

Zeroth-Korean 테스트 세트(CC BY 4.0)의 실제 한국인 10명, 문장 457개를 모든 쌍으로 짝지어
같은 사람과 다른 사람을 얼마나 헷갈리는지(EER, 낮을수록 좋음) 잰다. EER은 모델마다 다른
점수 척도와 상관없이 비교할 수 있다.

조건
- 문장 전체: 깨끗한 문장끼리
- 2초 조각: 앱이 실제로 판단하는 길이로 문장 가운데 2초만
- 2초 조각 + 나쁜 음질: 한쪽 조각에 통화 음질·잡음·울림·심한 압축 중 하나를 입혀서

    python tools/korean_sv.py                 # 모든 모델
    python tools/korean_sv.py ecapa voxblink2_samresnet34_ft
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vm import augment  # noqa: E402
from vm.audio import SR, telephone_band  # noqa: E402
from vm.engine import BENCH_MODELS, MODELS, VoiceEngine  # noqa: E402

DATA = ROOT / "demo_sources" / "others" / "zeroth_ko_test_with_speakers.parquet"
CROP = 2.0


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    scores = np.concatenate([pos, neg])
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))][np.argsort(-scores)]
    fnr = 1 - np.cumsum(labels) / len(pos)
    fpr = np.cumsum(1 - labels) / len(neg)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def degrade(x: np.ndarray, i: int) -> np.ndarray:
    kind = i % 4
    if kind == 0:
        return augment.lowbit(telephone_band(x), kbps=8)
    if kind == 1:
        return augment.noisy(x, snr_db=5, seed=i)
    if kind == 2:
        return augment.reverb(x, rt60=0.8, seed=i)
    return augment.lowbit(x, kbps=8)


def load() -> tuple[list[np.ndarray], np.ndarray]:
    rows = pq.read_table(DATA, columns=["speaker_id", "audio"]).to_pylist()
    utts, spk = [], []
    for r in rows:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr == SR and len(x) >= int(CROP * SR) + SR // 2:
            utts.append(x)
            spk.append(r["speaker_id"])
    return utts, np.array(spk)


def pair_eers(clean: np.ndarray, other: np.ndarray, spk: np.ndarray, skip_diag: bool) -> float:
    sims = clean @ other.T
    same = spk[:, None] == spk[None, :]
    mask = ~np.eye(len(spk), dtype=bool) if skip_diag else np.triu(np.ones_like(same), 1)
    return eer(sims[same & mask], sims[~same & mask])


def main() -> None:
    names = sys.argv[1:] or [*MODELS, *BENCH_MODELS]
    utts, spk = load()
    crops = [x[(len(x) - int(CROP * SR)) // 2:][: int(CROP * SR)] for x in utts]
    bad = [degrade(c, i) for i, c in enumerate(crops)]
    print(f"한국인 {len(set(spk))}명, 문장 {len(utts)}개 "
          f"(같은 사람 쌍 {int(sum((spk == s).sum() * ((spk == s).sum() - 1) / 2 for s in set(spk)))}개)\n")
    print(f"{'모델':28s} {'문장 전체':>9s} {'2초 조각':>9s} {'2초+나쁜 음질':>12s} {'처리 시간':>9s}")
    for name in names:
        eng = VoiceEngine(ROOT / "models", "auto", name)
        t = time.time()
        full = np.concatenate([eng.embed(x, np.array([[0, len(x) / SR]], np.float32)) for x in utts])
        two = np.concatenate([eng.embed(x, np.array([[0, CROP]], np.float32)) for x in crops])
        two_bad = np.concatenate([eng.embed(x, np.array([[0, CROP]], np.float32)) for x in bad])
        dt = time.time() - t
        print(f"{name:28s} {pair_eers(full, full, spk, False):9.1%} {pair_eers(two, two, spk, False):9.1%} "
              f"{pair_eers(two, two_bad, spk, True):12.1%} {dt:8.0f}s", flush=True)


if __name__ == "__main__":
    main()
