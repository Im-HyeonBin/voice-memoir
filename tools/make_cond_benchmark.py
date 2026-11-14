"""조건별 시험 세트(bench/cond)를 만든다: 시끄러운 곳·멀리서 울리는 녹음에서 앱 전체가 얼마나 찾는지.

Zeroth-Korean 테스트 화자 10명을 한 명씩 '찾을 사람'으로 두고, 사람마다 폴더 하나에 대화 녹음 7개를 만든다.
찾을 사람과 다른 두 사람이 번갈아 말하고(겹치지 않게), 파일마다 녹음 조건이 하나씩 다르다.
소음·울림은 학습에 쓰지 않은 것(DEMAND 실제 소음, 실측 방 울림, Pansori TEDx 말소리를 TV로)만 쓴다.
찾을 사람의 참고 녹음(7초 남짓, 폴더 밖 참고.wav)은 폴더 안 녹음과 다른 문장이다.
정답은 bench/cond_truth.json에 폴더별로 남긴다.

    python tools/make_cond_benchmark.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from korean_sv import load  # noqa: E402
from vm import audio  # noqa: E402

SR = audio.SR
OUT = ROOT / "bench" / "cond"
TARGET = "찾을사람"
FILES = [  # 파일 이름, 울림, 소음 환경, SNR, TV 말소리
    ("1 조용한 방.m4a", False, None, None, False),
    ("2 카페.m4a", False, "카페", 5, False),
    ("3 식당.m4a", False, "식당", 0, False),
    ("4 거실 TV.m4a", False, "거실", 10, True),
    ("5 멀리서.m4a", True, None, None, False),
    ("6 멀리서 광장.m4a", True, "광장", 5, False),
    ("7 함정 카페.m4a", False, "카페", 5, False),  # 찾을 사람 없음
]


def encode(wav: np.ndarray, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.wav"
        audio.write_wav(src, np.clip(wav, -0.99, 0.99))
        subprocess.run([audio.FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(src),
                        "-c:a", "aac", "-b:a", "96k", str(path)], check=True)


def conversation(turns: list[tuple[str, np.ndarray]], rng, far: bool) -> tuple[np.ndarray, list[dict]]:
    """번갈아 말하기. far면 사람마다 다른 실측 방 울림을 입힌다."""
    rirs = C.rir_paths("test")
    room = {who: C.load_rir(rirs[int(rng.integers(len(rirs)))]) for who in {w for w, _ in turns}}
    spans, t = [], 0.5
    out = [np.zeros(int(0.5 * SR), np.float32)]
    for who, x in turns:
        x = C.level(x, 0.05 * 10 ** (rng.uniform(-3, 3) / 20))
        if far:
            x = C.apply_rir(x, room[who])
        spans.append({"speaker": who, "start": round(t, 3), "end": round(t + len(x) / SR, 3)})
        gap = rng.uniform(0.3, 0.9)
        out += [x, np.zeros(int(gap * SR), np.float32)]
        t += len(x) / SR + gap
    return np.concatenate(out), spans


def main() -> None:
    from pansori import load as load_pansori
    utts, spk = load()
    tv_utts, _ = load_pansori()
    speakers = sorted(set(spk.tolist()))
    by = {s: [utts[i] for i in np.where(spk == s)[0]] for s in speakers}
    if OUT.exists():
        shutil.rmtree(OUT)
    truth = {}
    for k, s in enumerate(speakers):
        rng = np.random.default_rng(100 + k)
        a, b = speakers[(k + 1) % len(speakers)], speakers[(k + 2) % len(speakers)]
        mine = list(by[s])
        rng.shuffle(mine)
        others = {"A": list(by[a]), "B": list(by[b])}
        for v in others.values():
            rng.shuffle(v)
        folder = OUT / f"사람{k + 1:02d}"
        folder.mkdir(parents=True)
        ref, n = [], 0
        while sum(len(x) for x in ref) < 7 * SR:  # 참고 녹음: 폴더에 안 넣는 문장
            ref.append(mine.pop())
        audio.write_wav(OUT / f"사람{k + 1:02d}_참고.wav", C.level(np.concatenate(ref)))
        truth[folder.name] = {}
        for name, far, env, snr, tv in FILES:
            seq = ["A", "B", "A", "B"] if "함정" in name else [TARGET, "A", TARGET, "B", TARGET]
            turns = []
            for who in seq:
                pool = mine if who == TARGET else others[who]
                turns.append((who, pool.pop() if pool else by[s if who == TARGET else (a if who == "A" else b)][0]))
            x, spans = conversation(turns, rng, far)
            if env:
                chans = C.demand(env)
                x = C.add_noise(x, chans[int(rng.integers(len(chans)))], snr, rng)
                spans.append({"speaker": "소음", "start": 0.0, "end": round(len(x) / SR, 3)})
            if tv:
                v = np.concatenate([tv_utts[int(i)] for i in rng.choice(len(tv_utts), 8)])
                x = x + C.level(C.crop_noise(v, len(x), rng), 0.05 * 10 ** (-8 / 20))
                spans.append({"speaker": "TV", "start": 0.0, "end": round(len(x) / SR, 3)})
            encode(x, folder / name)
            truth[folder.name][name] = spans
            n += 1
        print(f"{folder.name}: 찾을 사람 {s}, 다른 사람 {a}·{b}, 파일 {n}개", flush=True)
    (ROOT / "bench" / "cond_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
