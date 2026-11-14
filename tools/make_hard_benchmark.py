"""어려운 시험 세트(bench/hard)를 만든다.

데모 폴더보다 실제 가족 녹음에 가까운 어려운 조건을 하나씩 넣는다.
TV 소리, 멀리서 울리는 소리, 겹쳐 말하기, 심한 압축, 시끄러운 식당, 목소리 닮은 사람과
빠르게 주고받기, 압축된 통화 녹음, 찾을 사람이 없는 함정 파일.
누가 몇 초에 말했는지 정답은 bench/hard_truth.json에 남긴다(구간이 겹칠 수 있다).
찾을 사람 목소리 첫 7초(참고 녹음)는 넣지 않는다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_library as demo  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402

SR = audio.SR
OUT = ROOT / "bench" / "hard"
rng = np.random.default_rng(11)


def level(x: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.08 * 10 ** (gain_db / 20)).astype(np.float32)


def scene(duration: float, events: list[tuple], beds: list[tuple] = ()) -> tuple[np.ndarray, list[dict]]:
    """events: (누구, 소리, 시작초, 크기dB). beds: (이름, 소리, 크기dB) 파일 전체에 깔리는 소리."""
    out = np.zeros(int(duration * SR), np.float32)
    spans = []
    for who, w, t, g in events:
        w = level(w, g)
        a = int(t * SR)
        b = min(len(out), a + len(w))
        out[a:b] += w[:b - a]
        spans.append({"speaker": who, "start": round(t, 3), "end": round(b / SR, 3)})
    for name, w, g in beds:
        w = np.resize(level(w, g), len(out))
        out += w
        spans.append({"speaker": name, "start": 0.0, "end": round(duration, 3)})
    return out, spans


def reverb(x: np.ndarray, rt60: float = 0.8, wet: float = 0.75) -> np.ndarray:
    n = int(rt60 * SR)
    h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / SR / rt60)
    h[0] = 4.0
    y = fftconvolve(x, h)[: len(x)]
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-9)
    return ((1 - wet) * x + wet * y).astype(np.float32)


def music(duration: float) -> np.ndarray:
    t = np.arange(int(duration * SR)) / SR
    chords = [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 246.9, 293.7]]
    out = np.zeros_like(t)
    for i, start in enumerate(np.arange(0, duration, 2.0)):
        seg = (t >= start) & (t < start + 2.0)
        tt = t[seg] - start
        for f in chords[i % 4]:
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
                out[seg] += amp * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt * 1.2)
    return out.astype(np.float32)


def noise(duration: float) -> np.ndarray:
    x = np.cumsum(rng.standard_normal(int(duration * SR)))
    x -= np.convolve(x, np.ones(200) / 200, mode="same")
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def encode(wav: np.ndarray, path: Path, args: list[str], created: str | None = None) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.wav"
        audio.write_wav(src, np.clip(wav, -0.99, 0.99))
        cmd = [audio.FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(src), *args]
        if created:
            cmd += ["-metadata", f"creation_time={created}"]
        subprocess.run(cmd + [str(path)], check=True)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    me, who = demo.target_voice()
    print(f"찾을 사람 목소리: {who}")
    cut = lambda a, b: me[int(a * SR):int(b * SR)]  # noqa: E731
    lib = {n: demo.fetch(n) for n in ["1.flac", "2.flac", "3.flac", "4.flac", "mlk.flac"]}
    engine = VoiceEngine(ROOT / "models", model="ecapa")  # 정답 세트를 만들 때 쓴 모델 그대로
    me_emb = engine.embed(me, np.array([[0, len(me) / SR]], np.float32))[0]
    (_, uncle), (_, cousin), (_, mom), (_, friend) = demo.korean_speakers(engine, me_emb)
    T = "찾을사람"
    truth = {}

    # 1) TV가 켜진 거실: 다른 사람 말소리와 음악이 파일 내내 깔린다
    x, sp = scene(34, [(T, cut(7, 14), 1.0, 0), ("사촌", cousin[0], 8.6, 0),
                       (T, cut(19, 26), 17.5, 0), ("엄마", mom[0], 25.2, 0)],
                  beds=[("TV", np.concatenate([lib["mlk.flac"], lib["2.flac"]]), -9), ("TV음악", music(34), -14)])
    truth[encode(x, OUT / "2016 설날 거실.m4a", ["-c:a", "aac", "-b:a", "96k"], "2016-02-08T12:00:00Z").name] = sp

    # 2) 멀리서 찍은 영상: 찾을 사람은 멀리서 울리고 작게, 닮은 삼촌은 카메라 가까이
    far = lambda w: reverb(w, 0.9, 0.8)  # noqa: E731
    x, sp = scene(30, [("삼촌", uncle[0], 0.5, 0), (T, far(cut(14, 19)), 7.0, -6),
                       ("삼촌", uncle[1], 12.6, 0), (T, far(cut(26, 33.8)), 20.0, -6)])
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.wav"
        audio.write_wav(a, x)
        path = OUT / "VID_20170815_143000.mp4"
        subprocess.run([audio.FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=0x303030:s=320x240:r=10", "-i", str(a), "-shortest", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(path)], check=True)
    truth[path.name] = sp

    # 3) 저녁식사: 말이 겹치고, 한 번은 아예 동시에 말한다
    x, sp = scene(30, [(T, cut(7, 14), 0.5, 0), ("엄마", mom[1], 6.0, -2),
                       (T, cut(14, 19), 15.0, 0), ("사촌", cousin[1], 15.0, -1),
                       ("삼촌", uncle[2], 23.0, 0)])
    truth[encode(x, OUT / "2018 저녁식사.m4a", ["-c:a", "aac", "-b:a", "96k"], "2018-09-24T19:00:00Z").name] = sp

    # 4) 카카오톡 음성메시지: 혼자 말하지만 8kbps로 심하게 압축
    x, sp = scene(17, [(T, cut(19, 26), 0.4, 0), (T, cut(26, 33.8), 8.2, 0)])
    truth[encode(x, OUT / "카톡 음성메시지 2020-11-03.opus",
                 ["-c:a", "libopus", "-b:a", "8k", "-ar", "16000"]).name] = sp

    # 5) 시끄러운 식당: 웅성거림과 큰 잡음
    babble = np.concatenate([lib["3.flac"], lib["4.flac"], lib["1.flac"]])
    x, sp = scene(22, [(T, cut(7, 14), 1.0, 0), ("친구", friend[0], 9.0, 0)],
                  beds=[("웅성거림", babble, -8), ("잡음", noise(22), -6)])
    truth[encode(x, OUT / "2021 식당 모임.m4a", ["-c:a", "aac", "-b:a", "96k"], "2021-12-18T20:00:00Z").name] = sp

    # 6) 목소리 닮은 삼촌과 2초씩 빠르게 주고받기
    ev, t = [], 0.4
    for k in range(4):
        ev.append((T, cut(14 + 2 * k, 16 + 2 * k), t, 0))
        t += 2.0 + 0.15
        u = uncle[k % len(uncle)]
        ev.append(("삼촌", u[int(1.0 * SR):int(3.0 * SR)], t, 0))
        t += 2.0 + 0.15
    x, sp = scene(t + 0.5, ev)
    truth[encode(x, OUT / "2022 삼촌과 대화.m4a", ["-c:a", "aac", "-b:a", "96k"], "2022-05-05T15:00:00Z").name] = sp

    # 7) 압축된 통화 녹음: 전화 음질 + MP3 16kbps
    x, sp = scene(22, [(T, cut(26, 33.8), 0.4, 0), ("엄마", mom[2], 8.6, 0), (T, cut(7, 11), 17.0, 0)])
    truth[encode(demo.phone(x), OUT / "통화녹음 2024-07-21.mp3",
                 ["-c:a", "libmp3lame", "-b:a", "16k", "-ar", "16000"]).name] = sp

    # 8) 함정: 찾을 사람 없이 닮은 삼촌과 친구, TV 음악
    x, sp = scene(26, [("삼촌", uncle[3], 0.5, 0), ("친구", friend[2], 8.5, 0), ("영어화자", lib["4.flac"], 16.0, 0)],
                  beds=[("TV음악", music(26), -16)])
    truth[encode(x, OUT / "2023 뉴스 녹화.m4a", ["-c:a", "aac", "-b:a", "96k"], "2023-01-10T21:00:00Z").name] = sp

    # 9) 함정: 통화 음질로 닮은 삼촌만
    x, sp = scene(18, [("삼촌", uncle[1], 0.4, 0), ("친구", friend[3], 9.0, 0)])
    truth[encode(demo.phone(x), OUT / "통화녹음 2025-03-02.mp3",
                 ["-c:a", "libmp3lame", "-b:a", "16k", "-ar", "16000"]).name] = sp

    (ROOT / "bench" / "hard_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    for name, spans in truth.items():
        p = OUT / name
        info = audio.probe(p)
        d = audio.recorded_date(p, info)
        who = sorted({s["speaker"] for s in spans})
        print(f"{name:34s} {audio.duration_of(info):5.1f}초  {d.value} ({d.label})  {', '.join(who)}")


if __name__ == "__main__":
    main()
