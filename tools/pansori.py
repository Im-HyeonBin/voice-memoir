"""Pansori-TEDxKR(한국어 TEDx 강연 41명, CC BY-NC-ND 4.0)을 시험용으로 불러온다. 학습에는 쓰지 않는다.

낭독체인 Zeroth와 달리 실제 강연장에서 녹음한 말소리라, 다른 녹음 환경에서도 결과가 같은지 볼 때 쓴다.
폴더 구조: <화자>/<영상 ID>/<화자>-<영상 ID>-<번호>.flac
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "pansori" / "pansori-tedxkr-corpus-1.0"


def load(per_speaker: int = 15, min_sec: float = 2.5) -> tuple[list[np.ndarray], np.ndarray]:
    utts, spk = [], []
    for k, d in enumerate(sorted(p for p in BASE.iterdir() if p.is_dir())):
        mine = []
        for f in sorted(d.rglob("*.flac")):
            x, sr = sf.read(f, dtype="float32")
            if sr == 16000 and x.ndim == 1 and len(x) >= min_sec * sr:
                mine.append(x)
            if len(mine) >= per_speaker:
                break
        if len(mine) >= 5:  # 문장이 너무 적은 화자는 뺀다
            utts += mine
            spk += [k] * len(mine)
    return utts, np.array(spk)
