"""해마다 묶기, '올해의 목소리' 고르기와 이어 붙이기, 구간 목록 CSV."""
from __future__ import annotations

import csv
import io
from datetime import date

import numpy as np
import pytest

from vm.audio import SR
from vm.speakers import Clip
from vm.timeline import (GAP_SEC, all_selections, build_timeline, compile_highlight, select_highlights,
                         source_rows, sources_csv)

from conftest import FakeCache


def clip(path, d, s, e, score=0.7, src="filename"):
    return Clip(path, d, src, s, e, score, score)


def test_build_timeline_groups_by_year_unknown_last():
    clips = [clip("b.wav", date(2021, 3, 1), 0, 4), clip("a.wav", date(2019, 1, 1), 0, 2),
             clip("x.wav", None, 0, 3, src="unknown"), clip("a.wav", date(2019, 1, 1), 5, 6)]
    years = build_timeline(clips)
    assert [y.label for y in years] == ["2019년", "2021년", "날짜 모름"]
    assert years[0].seconds == 3 and years[0].files == ["a.wav"]
    assert years[0].date_sources["filename"] == 1  # 파일마다 한 번 센다


def test_select_highlights_prefers_confident_and_keeps_time_order():
    clips = [clip("a.wav", date(2019, 1, 1), 0, 10, 0.6), clip("a.wav", date(2019, 1, 1), 20, 26, 0.9),
             clip("b.wav", date(2019, 6, 1), 0, 1, 0.99),  # 너무 짧아 빠진다
             clip("c.wav", date(2019, 3, 1), 0, 20, 0.8)]
    sel = select_highlights(clips, max_total=12, max_each=8)
    assert [s.clip.path for s in sel] == ["a.wav", "c.wav"]  # 시간 순서
    assert sum(s.duration for s in sel) <= 12 + 1
    assert all(s.clip.start <= s.start and s.end <= s.clip.end for s in sel)


def test_compile_highlight_levels_and_gaps():
    loud = np.full(3 * SR, 0.9, np.float32)
    quiet = np.full(2 * SR, 0.01, np.float32)
    cache = FakeCache({"loud.wav": loud, "quiet.wav": quiet})
    sels = all_selections([clip("loud.wav", date(2019, 1, 1), 0, 2), clip("quiet.wav", date(2019, 2, 1), 0, 2)])
    out = compile_highlight(sels, cache)
    assert len(out) == 4 * SR + int(GAP_SEC * SR)
    def rms(x):
        return float(np.sqrt((x ** 2).mean()))
    assert rms(out[SR // 2:SR]) == pytest.approx(rms(out[-SR:-SR // 2]), rel=1e-3)  # 조각마다 크기를 맞춘다
    assert np.abs(out).max() <= 0.95 and out[0] == 0  # 앞뒤를 부드럽게 줄인다
    assert compile_highlight([], cache).size == 0


def test_sources_csv_opens_in_excel(tmp_path):
    root = tmp_path / "녹음"
    (root / "2019").mkdir(parents=True)
    f = root / "2019" / "설날.m4a"
    f.write_bytes(b"x")
    years = build_timeline([clip(str(f), date(2019, 2, 5), 1.04, 4.96)])
    rows = source_rows(years[0], all_selections(years[0].clips), library_root=root)
    assert rows[0]["녹음 파일"] == "2019/설날.m4a" and rows[0]["시작(초)"] == 1.0 and rows[0]["끝(초)"] == 5.0
    data = sources_csv(rows)
    assert data.startswith("﻿".encode())  # 엑셀에서 한글이 깨지지 않게
    parsed = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert parsed[0]["해"] == "2019년" and parsed[0]["날짜 근거"] == "파일 이름"
    assert sources_csv([]) == "".encode("utf-8-sig")
