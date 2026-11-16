"""폴더 스캔 캐시: 다시 계산하지 않기, 바뀐 파일만 다시 계산, 쓰다 끊긴 캐시에서 살아나기."""
from __future__ import annotations

import os

import numpy as np
import pytest

from vm import audio
from vm.library import Library, find_media

from conftest import FakeEngine, compose, needs_ffmpeg

pytestmark = needs_ffmpeg


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "녹음"
    d.mkdir()
    audio.write_wav(d / "2019-03-01 거실.wav", compose(("me", 3), ("other", 3)))
    audio.write_wav(d / "2020-05-05 식당.wav", compose(("other", 4)))
    (d / ".숨김").mkdir()
    audio.write_wav(d / ".숨김" / "무시.wav", compose(("me", 2)))
    (d / "메모.txt").write_text("녹음 아님")
    return d


def cache_files(lib: Library, path):
    key = lib._key(path)
    return lib.cache_dir / f"{key}.json", lib.cache_dir / f"{key}.npz"


def test_find_media_skips_hidden_and_non_media(folder):
    assert [p.name for p in find_media(folder)] == ["2019-03-01 거실.wav", "2020-05-05 식당.wav"]


def test_scan_then_rescan_uses_cache(folder, tmp_path):
    eng = FakeEngine()
    lib = Library(eng, tmp_path / "cache")
    recs = lib.scan(folder, devices=["cpu"])
    assert eng.analyzed == 2
    assert recs[0].date.isoformat() == "2019-03-01" and recs[0].date_source == "filename"
    assert recs[0].duration == pytest.approx(6.0, abs=0.01)
    again = lib.scan(folder, devices=["cpu"])
    assert eng.analyzed == 2  # 두 번째는 계산하지 않는다
    assert np.array_equal(again[0].embeddings, recs[0].embeddings)


def test_changed_file_is_recomputed(folder, tmp_path):
    eng = FakeEngine()
    lib = Library(eng, tmp_path / "cache")
    lib.scan(folder, devices=["cpu"])
    f = folder / "2020-05-05 식당.wav"
    audio.write_wav(f, compose(("me", 5)))
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    lib.scan(folder, devices=["cpu"])
    assert eng.analyzed == 3


@pytest.mark.parametrize("damage", ["empty_json", "truncated_npz", "garbage_json"])
def test_broken_cache_is_recomputed(folder, tmp_path, damage):
    """서버가 쓰는 도중 멈춰 반쯤 남은 캐시 때문에 폴더가 영영 안 읽히던 문제."""
    eng = FakeEngine()
    lib = Library(eng, tmp_path / "cache")
    f = find_media(folder)[0]
    good = lib.analyze(f)
    meta, arr = cache_files(lib, f)
    if damage == "empty_json":
        meta.write_text("")
    elif damage == "garbage_json":
        meta.write_text("{\"duration\": ")
    else:
        arr.write_bytes(arr.read_bytes()[:20])
    rec = lib.analyze(f)
    assert eng.analyzed == 2
    assert np.array_equal(rec.embeddings, good.embeddings)
    assert lib._read_cache(f, meta, arr) is not None  # 새로 쓴 캐시는 온전하다


def test_leftover_tmp_files_do_not_count_as_cache(folder, tmp_path):
    eng = FakeEngine()
    lib = Library(eng, tmp_path / "cache")
    f = find_media(folder)[0]
    meta, arr = cache_files(lib, f)
    meta.with_name(meta.name + ".tmp").write_text("{}")
    arr.with_name(arr.name + ".tmp").write_bytes(b"")
    assert not lib.is_cached(f)
    meta.write_text("")  # 빈 파일만 남은 경우도 캐시가 아니다
    arr.write_bytes(b"")
    assert not lib.is_cached(f)
    lib.analyze(f)
    assert lib.is_cached(f)


def test_unreadable_file_is_reported_not_raised(tmp_path):
    d = tmp_path / "녹음"
    d.mkdir()
    (d / "깨진 파일.m4a").write_bytes(b"not audio at all")
    rec = Library(FakeEngine(), tmp_path / "cache").analyze(d / "깨진 파일.m4a")
    assert rec.error and rec.duration == 0 and len(rec.windows) == 0


def test_scan_falls_back_when_workers_die(tmp_path, monkeypatch):
    """작업자가 어떤 이유로든 죽으면 스캔이 실패하지 않고 이 프로세스에서 이어서 계산한다."""
    import contextlib
    import sys
    import types

    import vm.library as L
    from vm import audio

    from conftest import FakeEngine, compose
    bomb = tmp_path / "bomb_app.py"
    bomb.write_text("raise RuntimeError('작업자가 화면 파일을 다시 실행했다')\n")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(bomb)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.setattr(L, "_workers_skip_main", contextlib.nullcontext)  # 일부러 작업자가 죽게
    folder = tmp_path / "녹음"
    folder.mkdir()
    for i in range(3):
        audio.write_wav(folder / f"{i}.wav", compose(("me", 2)) * 0.1)
    eng = FakeEngine()
    progress = []
    records = L.Library(eng, tmp_path / "cache").scan(folder, devices=["cpu", "cpu"],
                                                      progress=lambda i, n, name: progress.append(i))
    assert len(records) == 3 and eng.analyzed == 3 and progress[-1] == 3
