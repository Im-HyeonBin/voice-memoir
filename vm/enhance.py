"""울림 걷어내기(WPE).

멀리서 녹음하면 벽에 반사된 소리가 늦게 따라와 말소리를 덮는다. WPE는 지난 몇 프레임으로
지금 프레임의 늦은 울림을 예측해 빼는 방법으로, 학습이 필요 없다(nara_wpe, MIT).
"""
from __future__ import annotations

import numpy as np

SIZE, SHIFT = 512, 128


def dereverb(x: np.ndarray, taps: int = 10, delay: int = 3, iterations: int = 3) -> np.ndarray:
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe
    if len(x) < SIZE * 4:
        return x
    Y = stft(x[None].astype(np.float64), size=SIZE, shift=SHIFT).transpose(2, 0, 1)  # (주파수, 채널, 프레임)
    Z = wpe(Y, taps=taps, delay=delay, iterations=iterations).transpose(1, 2, 0)
    z = istft(Z, size=SIZE, shift=SHIFT)[0][: len(x)]
    out = np.zeros(len(x), np.float32)
    out[: len(z)] = z
    return out
