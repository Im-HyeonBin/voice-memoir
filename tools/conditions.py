"""학습용·시험용 녹음 조건(소음·울림)을 만든다.

학습과 시험에 쓰는 소리는 출처부터 나눠, 학습한 소리로 시험해서 점수가 부풀지 않게 한다.
- 학습: OpenSLR 28의 시뮬레이션 방 울림(simulated_rirs)과 점 소음(pointsource_noises),
  Zeroth 학습 화자 여럿을 겹친 웅성거림
- 시험: OpenSLR 28의 실측 방 울림(real_rirs_isotropic_noises의 RWCP·AIR·REVERB 2014),
  DEMAND 실제 소음(카페·식당·거실·광장·버스), 다른 말뭉치(Pansori TEDx) 말소리를 TV 소리로

받은 압축 파일은 data/ 아래에 두고, 처음 쓸 때 풀어 둔다.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SR = 16000
DEMAND_ENVS = {"카페": "PCAFETER", "식당": "PRESTO", "거실": "DLIVING", "광장": "SPSQUARE", "버스": "TBUS"}


def _unzip(zpath: Path, dest: Path, marker: str) -> Path:
    if not (dest / marker).exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    return dest / marker


def _mono(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return x[:, 0]


# ── 방 울림 ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def rir_paths(split: str) -> tuple[Path, ...]:
    base = _unzip(DATA / "rirs" / "rirs_noises.zip", DATA / "rirs", "RIRS_NOISES")
    if split == "train":
        return tuple(sorted((base / "simulated_rirs").rglob("*.wav")))
    real = base / "real_rirs_isotropic_noises"
    # 실측 울림 325개: REVERB 2014 36개, RWCP 182개, AIR 107개
    return tuple(sorted(p for p in real.glob("*.wav")
                        if ("_rir_" in p.name or p.name.startswith("air_")) and "noise" not in p.name))


def load_rir(path: Path) -> np.ndarray:
    h = _mono(path)
    peak = int(np.argmax(np.abs(h)))
    h = h[max(0, peak - int(0.002 * SR)):]  # 직접음 직전부터: 울림을 입혀도 말 시작 시각이 그대로다
    return (h / (np.sqrt(np.sum(h ** 2)) + 1e-9)).astype(np.float32)


def apply_rir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = fftconvolve(x, h)[: len(x)]
    return (y * np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9)).astype(np.float32)


# ── 소음 ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def train_noise_paths() -> tuple[Path, ...]:
    base = _unzip(DATA / "rirs" / "rirs_noises.zip", DATA / "rirs", "RIRS_NOISES")
    return tuple(sorted((base / "pointsource_noises").glob("*.wav")))


@lru_cache(maxsize=8)
def demand(env: str) -> tuple[np.ndarray, ...]:
    """DEMAND 환경 하나의 16채널 녹음(각 5분). 채널마다 마이크 위치가 달라 서로 다른 소음으로 쓴다."""
    name = DEMAND_ENVS.get(env, env)
    base = _unzip(DATA / "demand" / f"{name}_16k.zip", DATA / "demand", name)
    return tuple(_mono(p) for p in sorted(base.glob("ch*.wav")))


def crop_noise(n: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(n) < length:
        n = np.tile(n, length // max(1, len(n)) + 1)
    a = int(rng.integers(0, len(n) - length + 1))
    return n[a:a + length]


def add_noise(x: np.ndarray, n: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """말소리 세기 대비 snr_db만큼 작은 소음을 섞는다."""
    n = crop_noise(n, len(x), rng)
    px = np.mean(x ** 2) + 1e-12
    pn = np.mean(n ** 2) + 1e-12
    return (x + n * np.sqrt(px / pn * 10 ** (-snr_db / 10))).astype(np.float32)


def level(x: np.ndarray, rms: float = 0.05) -> np.ndarray:
    return (x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)
