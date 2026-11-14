"""대상 화자 추출 감싸개(vm/tse.py): 가중치 읽기, 출력 꼴, 같은 구간 다시 뽑지 않기.

학습한 가중치 대신 무작위 가중치를 저장해 쓴다(모양과 캐시 동작만 본다).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vm.audio import SR
from vm.tse import TDSpeakerBeam, Extractor


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(0)
    net = TDSpeakerBeam()
    state = {**net.state_dict(), "loss.weight": torch.zeros(1)}  # ESPnet 저장본처럼 다른 키가 섞여 있어도 된다
    path = tmp_path_factory.mktemp("tse") / "random.pth"
    torch.save(state, path)
    return path


def test_output_shape_and_level(weights):
    ext = Extractor(weights, "cpu")
    rng = np.random.default_rng(0)
    mix = rng.standard_normal(SR).astype(np.float32) * 0.1
    out = ext(mix, rng.standard_normal(SR // 2).astype(np.float32) * 0.1)
    assert out.shape == mix.shape and out.dtype == np.float32
    assert np.abs(out).max() == pytest.approx(0.9, abs=1e-4)  # 끝값을 0.9로 맞춘다


def test_cache_hits_and_lru(weights, monkeypatch):
    ext = Extractor(weights, "cpu")
    monkeypatch.setattr(Extractor, "CACHE", 2)
    runs = []
    real = ext._run
    ext._run = lambda m, e: runs.append(1) or real(m, e)
    rng = np.random.default_rng(1)
    mix = rng.standard_normal(SR // 2).astype(np.float32) * 0.1
    enroll = rng.standard_normal(SR // 2).astype(np.float32) * 0.1
    first = ext(mix, enroll, key=("a.wav", 1.0, 1.5))
    again = ext(mix, enroll, key=("a.wav", 1.0, 1.5))
    assert len(runs) == 1 and ext.hits == 1
    assert np.allclose(first, again, atol=1e-3)  # 절반 정밀도로 둔다

    ext(mix, enroll * 0.5, key=("a.wav", 1.0, 1.5))  # 참고 목소리가 바뀌면 다시 뽑는다
    assert len(runs) == 2
    ext(mix, enroll, key=("b.wav", 0.0, 0.5))
    ext(mix, enroll, key=("c.wav", 0.0, 0.5))  # 가장 오래된 것이 밀려난다
    ext(mix, enroll, key=("a.wav", 1.0, 1.5))
    assert len(runs) == 5
    ext(mix, enroll)  # 열쇠가 없으면 저장하지 않는다
    assert len(ext._cache) == 2
