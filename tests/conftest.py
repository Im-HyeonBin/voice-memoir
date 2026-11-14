"""시험 공용 도구.

모델 없이 찾기 흐름 전체를 돌릴 수 있게, 소리 값으로 '누구 목소리인지'를 정하는 가짜 엔진을 둔다.
- +0.5 근처 표본 = 찾을 사람, -0.5 근처 표본 = 다른 사람
- 지문 = [찾을 사람 표본 비율, 다른 사람 표본 비율, 0.1]을 길이 1로 맞춘 것
그래서 두 사람이 섞인 구간은 지문도 섞이고, 다른 사람 표본을 지우면(가짜 추출기) 찾을 사람 쪽으로 돌아온다.
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm.audio import SR  # noqa: E402
from vm.engine import ECAPA_CAL, VoiceEngine  # noqa: E402
from vm.library import FileRecord  # noqa: E402

ME, OTHER = 0.5, -0.5
DIM = 3

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None and not Path("/opt/homebrew/bin/ffmpeg").exists(),
                                  reason="ffmpeg가 없어요")


def fake_embedding(x: np.ndarray) -> np.ndarray:
    v = np.array([(x > 0.25).mean() if len(x) else 0.0, (x < -0.25).mean() if len(x) else 0.0, 0.1], np.float32)
    return v / np.linalg.norm(v)


class FakeEngine:
    """VoiceEngine과 같은 꼴(cal·dim·model·embed·analyze_wave)만 갖춘 가짜."""

    def __init__(self, cal=ECAPA_CAL):
        self.cal = cal
        self.dim = DIM
        self.model = "fake"
        self.models_dir = ROOT / "models"
        self.analyzed = 0  # analyze_wave를 부른 횟수(캐시가 쓰였는지 확인용)

    def embed(self, wav: np.ndarray, windows: np.ndarray) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros((0, DIM), np.float32)
        return np.stack([fake_embedding(wav[int(s * SR):int(e * SR)]) for s, e in windows])

    def analyze_wave(self, wav: np.ndarray):
        self.analyzed += 1
        win = VoiceEngine.windows([(0.0, len(wav) / SR)])
        return win, self.embed(wav, win)


class FakeCache:
    """AudioCache 대신 메모리에 든 소리를 돌려준다."""

    def __init__(self, waves: dict[str, np.ndarray]):
        self.waves = waves

    def get(self, path: str) -> np.ndarray:
        return self.waves[path]

    def slice(self, path: str, start: float, end: float) -> np.ndarray:
        return self.waves[path][int(start * SR):int(end * SR)]


def compose(*parts: tuple[str, float]) -> np.ndarray:
    """('me'|'other'|'mix'|'quiet', 초) 조각을 이어 붙인다. mix는 3표본 중 1개만 찾을 사람."""
    out = []
    for kind, sec in parts:
        n = int(sec * SR)
        if kind == "me":
            out.append(np.full(n, ME, np.float32))
        elif kind == "other":
            out.append(np.full(n, OTHER, np.float32))
        elif kind == "mix":
            x = np.full(n, OTHER, np.float32)
            x[::3] = ME
            out.append(x)
        else:
            out.append(np.zeros(n, np.float32))
    return np.concatenate(out)


def record(engine: FakeEngine, path: str, wav: np.ndarray, when: date | None = date(2020, 5, 1),
           narrowband: bool = False) -> FileRecord:
    win, emb = engine.analyze_wave(wav)
    return FileRecord(path, len(wav) / SR, when, "filename" if when else "unknown", win, emb, narrowband)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()
