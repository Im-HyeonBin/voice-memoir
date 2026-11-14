"""실제 녹음 채점 도구(tools/evaluate_real.py)의 정답 파일 읽기와 채점."""
from __future__ import annotations

import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from evaluate_real import read_labels, score, truth_file  # noqa: E402


def test_read_audacity_labels(tmp_path):
    f = tmp_path / "설날.txt"
    f.write_text("1.000000\t3.500000\t아버지\n\\\t200.0\t3000.0\n5.0\t6.0\t엄마\n10.0\t12.0\t아버지\n",
                 encoding="utf-8")
    assert read_labels(f) == [(1.0, 3.5), (5.0, 6.0), (10.0, 12.0)]  # 이름을 안 주면 모든 레이블
    assert read_labels(f, label="아버지") == [(1.0, 3.5), (10.0, 12.0)]


def test_read_rttm_and_empty_file(tmp_path):
    f = tmp_path / "통화.rttm"
    f.write_text("SPEAKER 통화 1 0.50 2.00 <NA> <NA> dad <NA> <NA>\n"
                 "SPEAKER 통화 1 3.00 1.00 <NA> <NA> mom <NA> <NA>\n")
    assert read_labels(f, speaker="dad") == [(0.5, 2.5)]
    empty = tmp_path / "함정.txt"
    empty.write_text("")
    assert read_labels(empty) == []


def test_truth_file_prefers_rttm(tmp_path):
    a = tmp_path / "a.m4a"
    a.write_bytes(b"")
    assert truth_file(a) is None
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "a.rttm").write_text("")
    assert truth_file(a).suffix == ".rttm"


def test_score_with_collar():
    # 정답 2~6초, 담은 것 1~5초: 경계 앞뒤 0.25초는 빼고 센다
    r = score([(2.0, 6.0)], [(1.0, 5.0)], duration=10.0)
    assert r["target"] == pytest.approx(3.5)   # 2.25~5.75
    assert r["found"] == pytest.approx(2.75)   # 2.25~5.0
    assert r["wrong"] == pytest.approx(0.75)   # 1.0~1.75
    assert score([], [(0.0, 3.0)], duration=5.0)["wrong"] == pytest.approx(3.0)  # 함정 녹음
