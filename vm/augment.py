"""찾을 사람 목소리에 녹음 조건을 입혀 기준 지문을 여러 개 만든다.

멀리서 울리는 소리, 잡음, 심한 압축은 같은 사람이라도 지문을 크게 바꾼다.
기준 쪽에도 같은 조건을 입혀 두면, 그런 녹음과 비교할 때 조건이 맞는 기준이 생긴다.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import fftconvolve

from .audio import FFMPEG, SR, telephone_band


def reverb(x: np.ndarray, rt60: float = 0.8, wet: float = 0.75, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / SR / rt60)
    h[0] = 4.0
    y = fftconvolve(x, h)[: len(x)]
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-9)
    return ((1 - wet) * x + wet * y).astype(np.float32)


def noisy(x: np.ndarray, snr_db: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = np.cumsum(rng.standard_normal(len(x)))
    n -= np.convolve(n, np.ones(200) / 200, mode="same")
    n = n / (np.sqrt(np.mean(n ** 2)) + 1e-9) * np.sqrt(np.mean(x ** 2)) * 10 ** (-snr_db / 20)
    return (x + n).astype(np.float32)


def lowbit(x: np.ndarray, kbps: int = 10) -> np.ndarray:
    """Opus로 심하게 압축했다가 다시 푼다(메신저 음성메시지 조건)."""
    enc = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg", "-"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    dec = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-i", "-", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, np.float32)
    out = np.zeros(len(x), np.float32)
    out[: min(len(x), len(y))] = y[: len(x)]
    return out


def wide_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    return {"울림": reverb(x), "잡음": noisy(x), "압축": lowbit(x)}


def narrow_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    return {"통화 압축": lowbit(telephone_band(x), kbps=8)}
