"""실제 모델로 도는 시험. 가중치나 시험 세트가 없으면 건너뛴다(저장소에는 둘 다 넣지 않는다).

- 어려운 세트(bench/hard, tools/make_hard_benchmark.py로 만듦)에서 정확도가 기록보다 크게 떨어지지 않는지
- 화면(app.py)을 처음부터 끝까지 눌러 봐도 오류가 없는지
맥에서는 추론만 한다(학습은 서버에서만).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import ROOT, needs_ffmpeg

KO_WEIGHTS = ROOT / "models" / "alt" / "voxblink2_samresnet34_ko" / "avg_model.pt"
HARD = ROOT / "bench" / "hard"
REF = ROOT / "demo_sources" / "참고음성_본인_7초.wav"
TRAPS = ("2023 뉴스 녹화.m4a", "통화녹음 2025-03-02.mp3")  # 찾을 사람이 없는 녹음(TV 뉴스, 다른 사람 통화)

pytestmark = [
    needs_ffmpeg,
    pytest.mark.skipif(not KO_WEIGHTS.exists(), reason="한국어 학습 모델 가중치가 없어요"),
    pytest.mark.skipif(not HARD.exists() or not REF.exists(), reason="어려운 세트가 없어요(tools/make_hard_benchmark.py)"),
]


@pytest.fixture(scope="module")
def hard():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from evaluate import Scorer

    from vm import audio
    from vm.engine import KO_MODEL, VoiceEngine
    from vm.library import Library
    from vm.speakers import discover_speakers, target_from_reference
    eng = VoiceEngine(ROOT / "models", "auto", KO_MODEL)
    records = Library(eng, ROOT / ".cache").scan(HARD)
    cache = audio.AudioCache(max_files=16)
    clusters = discover_speakers(records, eng, cache)
    target = target_from_reference(eng, audio.decode(REF), clusters)
    scorer = Scorer(records, json.loads((ROOT / "bench" / "hard_truth.json").read_text()))
    return eng, records, cache, target, scorer


def test_hard_set_accuracy_and_traps(hard):
    from vm.speakers import find_clips
    eng, records, cache, target, scorer = hard
    s = scorer.score(find_clips(records, target, eng, cache))
    # 기록(bench/server/eval_hard_ko_l3.txt, 기본 기준): 말소리 찾은 비율 약 70%, 잘못 담은 시간 2초 안팎
    assert s["found_speech"] >= 0.60, s["found_speech"]
    assert s["wrong_sec"] <= 3.0, s["wrong_sec"]
    for trap in TRAPS:
        assert s["per_file"][trap]["wrong"] == 0, trap  # 찾을 사람이 없는 녹음에서는 아무것도 담지 않는다


def test_stricter_level_never_finds_more(hard):
    from vm.speakers import find_clips
    eng, records, cache, target, scorer = hard
    found = [scorer.score(find_clips(records, target, eng, cache, t))["wrong_sec"] for t in eng.cal.levels]
    assert all(a >= b - 1e-6 for a, b in zip(found, found[1:]))  # 엄격할수록 잘못 담는 시간이 늘지 않는다


def test_app_end_to_end():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception
    next(t for t in at.text_input if t.label == "녹음 폴더 위치").set_value(str(HARD))  # 입력칸 순서는 환경마다 달라 이름으로 찾는다
    next(b for b in at.button if b.label == "불러오기").click().run()
    assert not at.exception
    assert {Path(r.path).parent for r in at.session_state["records"]} == {HARD}  # 정말 그 폴더를 불러왔는지
    clusters = at.session_state["clusters"]
    assert clusters, "목소리를 하나도 못 찾았어요"
    biggest = max(clusters, key=lambda c: c.seconds)
    next(cb for cb in at.checkbox if cb.key == f"pick_{biggest.index}").check()
    at.run()
    next(b for b in at.button if b.label == "고른 목소리로 찾기").click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert re.search(r"\d+개 해에서 모두 [\d.]+초", text), "결과 요약 문구가 없어요"
    at.select_slider[0].set_value(at.select_slider[0].options[-1]).run()  # 가장 엄격하게
    assert not at.exception


ECAPA_DIR = ROOT / "models" / "ecapa"


@pytest.mark.skipif(not (ECAPA_DIR / "embedding_model.ckpt").exists(), reason="ECAPA 가중치가 없어요")
def test_worker_processes_do_not_rerun_main(tmp_path, monkeypatch):
    """Streamlit은 app.py를 __main__으로 돌린다. 스캔 작업자(spawn)가 그 파일을 다시 실행하면 죽어서 스캔이 실패했다."""
    import sys
    import types

    from vm import audio
    from vm.engine import VoiceEngine
    from vm.library import Library

    from conftest import compose
    bomb = tmp_path / "bomb_app.py"
    bomb.write_text("raise RuntimeError('작업자가 화면 파일을 다시 실행했다')\n")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(bomb)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    folder = tmp_path / "녹음"
    folder.mkdir()
    for i in range(3):
        audio.write_wav(folder / f"{i}.wav", compose(("me", 2)) * 0.1)
    eng = VoiceEngine(ROOT / "models", "cpu", "ecapa")
    in_parent = []
    real = eng.analyze_wave
    eng.analyze_wave = lambda wav: in_parent.append(1) or real(wav)
    progress = []
    records = Library(eng, tmp_path / "cache").scan(folder, devices=["cpu", "cpu"],
                                                    progress=lambda i, n, name: progress.append(i))
    assert len(records) == 3 and not any(r.error for r in records)
    assert in_parent == [], "작업자가 죽어 이 프로세스가 대신 계산했어요"
    assert progress[-1] == 3
    assert sys.modules["__main__"] is fake_main  # 끝나면 되돌린다


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
