"""온라인 서비스의 '예시 녹음'을 공개 음성만으로 만든다(deploy/demo/).

데모 폴더(tools/make_demo_library.py)와 같은 구성이지만 찾을 사람이 개인 목소리가 아니다.
- 찾을 사람과 가족 역할: Zeroth-Korean 테스트 세트(CC BY 4.0)의 서로 다른 화자. 화자 번호가 있어 사람을 확실히 나눈다.
  찾을 사람과 목소리가 가장 닮은 화자를 일부러 '삼촌'으로 넣는다.
- 영어 화자: LibriSpeech 샘플, VOiCES 샘플(둘 다 CC BY 4.0). 연설 녹음(mlk.flac)은 넣지 않는다.
- 날짜 단서 세 가지(녹음 정보·파일 이름·파일 시각), 전화 음질, 찾을 사람이 없는 녹음을 그대로 넣는다.
결과: deploy/demo/library/ 녹음 5개, deploy/demo/reference.wav(참고 녹음 7초), ground_truth.json, SOURCES.txt
실행(맥, 추론만): .venv/bin/python tools/make_public_demo.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import make_demo_library as demo  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402

SR = audio.SR
OUT = ROOT / "deploy" / "demo"
LIB = OUT / "library"
ZEROTH = ROOT / "demo_sources" / "others" / "zeroth_ko_test_with_speakers.parquet"
OTHERS = ROOT / "demo_sources" / "others"


def zeroth_speakers() -> dict[str, list[np.ndarray]]:
    """화자 번호별 발화(3초 이상, 16kHz)."""
    rows = pq.read_table(ZEROTH, columns=["speaker_id", "audio"]).to_pylist()
    out: dict[str, list[np.ndarray]] = {}
    for r in rows:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            x = resample_poly(x, SR, sr).astype(np.float32)
        if len(x) >= 3 * SR:
            out.setdefault(str(r["speaker_id"]), []).append(x)
    return out


def take(utts: list[np.ndarray], sec: float) -> np.ndarray:
    parts, total = [], 0
    for x in utts:
        parts.append(x)
        total += len(x)
        if total >= sec * SR:
            break
    return np.concatenate(parts)[: int(sec * SR)]


def main() -> None:
    LIB.mkdir(parents=True, exist_ok=True)
    for f in LIB.iterdir():
        f.unlink()
    demo.LIB = LIB  # 저장 도우미가 이 폴더에 쓰게 한다

    spk = zeroth_speakers()
    # 찾을 사람: 발화가 가장 많은 화자(참고 7초 + 녹음 속 약 27초가 필요하다)
    target_id = max(spk, key=lambda k: sum(len(x) for x in spk[k]))
    me = take(spk[target_id], 34)
    audio.write_wav(OUT / "reference.wav", demo.seconds(me, 0, 7))

    engine = VoiceEngine(ROOT / "models")
    def emb(x: np.ndarray) -> np.ndarray:
        return engine.embed(x, np.array([[0, len(x) / SR]], np.float32))[0]
    me_emb = emb(me)
    others = sorted((k for k in spk if k != target_id and len(spk[k]) >= 3),
                    key=lambda k: -float(np.mean([emb(x) @ me_emb for x in spk[k][:4]])))
    uncle_id, cousin_id, mom_id, friend_id = others[0], others[1], others[2], others[3]
    uncle, cousin, mom, friend = (spk[k] for k in (uncle_id, cousin_id, mom_id, friend_id))
    sims = {name: float(np.mean([emb(x) @ me_emb for x in spk[k][:4]]))
            for name, k in (("삼촌", uncle_id), ("사촌", cousin_id), ("엄마", mom_id), ("친구", friend_id))}
    print("찾을 사람 Zeroth 화자", target_id, "· 닮은 정도", {k: round(v, 2) for k, v in sims.items()})

    libri = {n: audio.decode(OTHERS / n) for n in ("1.flac", "2.flac", "3.flac", "4.flac")}
    voices = audio.decode(OTHERS / "VOiCES_sp0307.wav")
    made, truth = [], {}

    def add(path: Path, spans: list[dict]) -> None:
        made.append(path)
        truth[path.name] = spans

    x, spans = demo.compose(("삼촌", uncle[0]), ("찾을사람", demo.seconds(me, 7, 14)), ("엄마", mom[0]),
                            ("영어화자1", demo.seconds(libri["1.flac"], 0, 6)),
                            ("찾을사람", demo.seconds(me, 14, 19)), ("삼촌", uncle[1]))
    add(demo.save_m4a("2019 추석 가족모임.m4a", demo.room_noise(x, 0.012), "2019-09-13T15:20:00Z"), spans)

    x, spans = demo.compose(("사촌", cousin[0]), ("찾을사람", demo.seconds(me, 19, 26)),
                            ("영어화자2", demo.seconds(libri["2.flac"], 0, 5)), ("사촌", cousin[1]))
    add(demo.save_video("KakaoTalk_Video_2021-05-08-19-22-10.mp4", demo.room_noise(x, 0.006)), spans)

    x, spans = demo.compose(("엄마", mom[1]), ("찾을사람", demo.seconds(me, 26, 34)), ("엄마", mom[2]))
    add(demo.save_m4a("통화녹음 2023.03.02.m4a", demo.phone(x)), spans)

    x, spans = demo.compose(("친구", friend[0]), ("삼촌", uncle[2]), ("사촌", cousin[2]),
                            ("영어화자3", voices), ("친구", friend[1]))
    add(demo.save_m4a("2024 동창회.m4a", demo.room_noise(x, 0.01), "2024-02-10T19:00:00Z"), spans)

    x, spans = demo.compose(("영어화자4", demo.seconds(libri["4.flac"], 0, 7)),
                            ("영어화자5", demo.seconds(libri["3.flac"], 0, 6)))
    memo = demo.save_m4a("음성메모 001.m4a", demo.room_noise(x, 0.004))
    t = time.mktime(datetime(2022, 6, 1, 10, 0).timetuple())
    os.utime(memo, (t, t))
    add(memo, spans)

    (OUT / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    (OUT / "SOURCES.txt").write_text(
        "예시 녹음은 공개 음성만으로 만든 합성 녹음이다(실제 가족 녹음이 아니다).\n"
        f"- 한국어 화자: Zeroth-Korean 테스트 세트(OpenSLR 40), CC BY 4.0. 찾을 사람은 화자 {target_id}\n"
        "- 영어 화자: LibriSpeech 샘플(CC BY 4.0), VOiCES 샘플(CC BY 4.0)\n"
        "- 잡음·전화 음질은 tools/make_demo_library.py의 방법으로 입혔다\n")
    for p in made:
        info = audio.probe(p)
        d = audio.recorded_date(p, info)
        print(f"{p.name:42s} {audio.duration_of(info):5.1f}초  {d.value} ({d.label})  "
              f"{', '.join(sorted({s['speaker'] for s in truth[p.name]}))}")


if __name__ == "__main__":
    main()
