"""모델 없이 도는 작은 부품들: 조사, 날짜 추정, 전화 음질 판별, 창 나누기, 기준값."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date

import numpy as np
import pytest

from vm import audio
from vm.audio import SR
from vm.engine import ECAPA_CAL, HOP, KO_MODEL, MODELS, WIN, Calibration, VoiceEngine, _load_cal
from vm.text import josa


@pytest.mark.parametrize("word,expected", [
    ("형", "형과"), ("아버지", "아버지와"), ("엄마", "엄마와"), ("할아버지", "할아버지와"),
    ("선생님", "선생님과"), ("Tom", "Tom와"), ("", "와"),
])
def test_josa(word, expected):
    assert josa(word, "과", "와") == expected


# ── 녹음 날짜 ──────────────────────────────

def test_date_from_metadata_wins_over_filename(tmp_path):
    f = tmp_path / "2015-01-02 가족.m4a"
    f.write_bytes(b"x")
    info = {"format": {"tags": {"creation_time": "2019-07-08T10:00:00.000000Z"}}}
    d = audio.recorded_date(f, info)
    assert (d.value, d.source) == (date(2019, 7, 8), "metadata")


@pytest.mark.parametrize("name,expected", [
    ("VID_20170815_143000.mp4", date(2017, 8, 15)),
    ("통화녹음 2024-07-21.mp3", date(2024, 7, 21)),
    ("카톡 음성메시지 2020.11.03.opus", date(2020, 11, 3)),
])
def test_date_from_filename(tmp_path, name, expected):
    f = tmp_path / name
    f.write_bytes(b"x")
    d = audio.recorded_date(f, {})
    assert (d.value, d.source) == (expected, "filename")


def test_implausible_dates_fall_back_to_file_time(tmp_path):
    # 메타데이터 1904년(빈 값을 기기가 채운 것), 파일 이름의 숫자는 날짜가 아님
    f = tmp_path / "IMG_20991345.mp4"
    f.write_bytes(b"x")
    os.utime(f, (1_600_000_000, 1_600_000_000))  # 2020-09
    d = audio.recorded_date(f, {"format": {"tags": {"creation_time": "1904-01-01T00:00:00Z"}}})
    assert d.source == "file"
    assert d.value.year in (2020, date.today().year)  # 파일 생성 시각이 더 이를 수 있어 둘 중 하나


# ── 전화 음질 ──────────────────────────────

def test_narrowband_detection():
    rng = np.random.default_rng(0)
    wide = rng.standard_normal(SR * 3).astype(np.float32) * 0.1
    assert not audio.is_narrowband(wide)
    assert audio.is_narrowband(audio.telephone_band(wide))


def test_wav_bytes_roundtrip():
    import io

    import soundfile as sf
    x = (np.sin(np.linspace(0, 200, SR)) * 0.5).astype(np.float32)
    y, sr = sf.read(io.BytesIO(audio.wav_bytes(x)), dtype="float32")
    assert sr == SR and len(y) == len(x)
    assert np.abs(y - x).max() < 1e-3  # 16비트로 저장해 생기는 오차만


# ── 창 나누기 ──────────────────────────────

def test_windows_cover_region_with_hop():
    w = VoiceEngine.windows([(0.0, 5.0)])
    assert w[0, 0] == 0.0 and np.allclose(w[:, 1] - w[:, 0], WIN)
    assert np.allclose(np.diff(w[:-1, 0]), HOP)
    assert w[-1, 1] == pytest.approx(5.0)  # 꼬리까지 덮는다


def test_windows_short_regions():
    w = VoiceEngine.windows([(0.0, 0.5), (1.0, 2.2)])  # 0.5초는 너무 짧아 버리고, 1.2초는 창 하나
    assert w.tolist() == [[1.0, pytest.approx(2.2)]]
    assert VoiceEngine.windows([]).shape == (0, 2)


# ── 기준값 ──────────────────────────────

def test_calibration_threshold_is_middle_level_and_sure_defaults():
    c = Calibration(levels=(1, 2, 3, 4, 5), cluster_distance=0.5, link=0.6, same_person=0.5,
                    candidate_margin=0.1, candidate_floor=0.2, maybe_band=0.1, rival_margin=0.05)
    assert c.threshold == 3
    assert c.same_person_sure == c.same_person  # 따로 안 정하면 같은 값


@pytest.mark.parametrize("name", [m for m, v in MODELS.items() if v[3] is not None])
def test_built_in_calibrations_are_ordered(name):
    c = MODELS[name][3]
    assert list(c.levels) == sorted(c.levels)  # 넉넉하게 → 엄격하게
    assert c.candidate_floor < c.threshold
    assert c.same_person <= c.same_person_sure


def test_ko_model_is_stricter_than_ecapa_scale():
    assert MODELS[KO_MODEL][3].threshold > ECAPA_CAL.threshold


def test_load_cal_roundtrip(tmp_path):
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(asdict(ECAPA_CAL)))
    assert _load_cal(f) == ECAPA_CAL
    with pytest.raises(FileNotFoundError):
        _load_cal(tmp_path / "없음.json")
