"""데모용 '가족 녹음 폴더'를 만든다.

찾을 사람(임현빈 본인 목소리 33.8초)의 조각을 여러 해의 녹음에 다른 사람들과 섞어 넣는다.
- 다른 사람: Zeroth-Korean(CC BY 4.0) 실제 한국인 화자 4명, LibriSpeech·VOiCES 영어 화자.
  Zeroth에는 화자 정보가 없어서 이 프로젝트의 화자 지문으로 발화를 사람별로 묶어 고른다.
  찾을 사람과 목소리가 닮은 사람(삼촌 역할)을 일부러 넣고, 같은 사람이 여러 해에
  반복해 나오게 해 자동 발견 기능도 함께 시험한다
- 누가 몇 초에 말했는지 정답을 demo_sources/ground_truth.json에 남겨 정확도를 잰다
- 날짜: 녹음 정보(메타데이터) / 파일 이름 / 파일 시각 세 가지 경우를 모두 넣는다
- 녹음 환경: 실내 잡음, 전화 음질(300~3400Hz, 8kHz)을 일부러 입힌다
- 찾을 사람이 아예 없는 해도 넣어서 엉뚱한 구간을 잡지 않는지 본다

참고 음성 모드 시험용으로 본인 목소리 첫 7초는 라이브러리에 넣지 않고 따로 둔다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import io
import json
import shutil

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402

SR = audio.SR
SRC = ROOT / "demo_sources"
LIB = ROOT / "demo_library"
TARGET_SRC = SRC / "찾을사람_전체_33초.wav"  # 임현빈 본인 목소리 33.8초 (7월 실험 녹음)
VOICES_SRC = SRC / "others" / "VOiCES_sp0307.wav"  # VOiCES 영어 화자 (CC BY 4.0)
LIBRI = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{}"
rng = np.random.default_rng(7)


def target_voice() -> tuple[np.ndarray, str]:
    """찾을 사람 목소리 34초 남짓.

    본인 녹음(demo_sources/찾을사람_전체_33초.wav)이 있으면 그것을 쓴다. 개인 목소리라 저장소에는 넣지 않으므로,
    없으면 Zeroth-Korean 테스트 화자 한 명(공개, CC BY 4.0)의 문장을 이어 붙여 대신한다.
    README의 시험 결과는 본인 녹음으로 만든 세트 기준이다.
    """
    if TARGET_SRC.exists():
        return audio.decode(TARGET_SRC), "본인 녹음"
    rows = pq.read_table(SRC / "others" / "zeroth_ko_test_with_speakers.parquet",
                         columns=["speaker_id", "audio"]).to_pylist()
    spk = sorted({r["speaker_id"] for r in rows})[0]
    parts, total = [], 0
    for r in rows:
        if r["speaker_id"] != spk:
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        x = resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
        parts.append(x)
        total += len(x)
        if total >= 34 * SR:
            break
    return np.concatenate(parts)[: int(34 * SR)], f"Zeroth 테스트 화자 {spk}"


def fetch(name: str) -> np.ndarray:
    dst = SRC / "others" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LIBRI.format(name), dst)
    return audio.decode(dst)


def korean_speakers(engine: VoiceEngine, target: np.ndarray, n: int = 4,
                    per: int = 4) -> list[tuple[float, list[np.ndarray]]]:
    """Zeroth-Korean 발화를 화자 지문으로 묶어 서로 다른 사람 n명을 고른다.

    찾을 사람과 목소리가 비슷한 사람도 일부러 거르지 않는다(실제 가족은 목소리가 닮는다).
    돌려주는 값: [(찾을 사람과의 닮은 정도, 발화들)], 닮은 순서.
    """
    out_dir = SRC / "others" / "ko_speakers"
    meta = out_dir / "speakers.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        return [(g["similarity"], [audio.decode(out_dir / f) for f in g["files"]]) for g in info]

    rows = pq.read_table(SRC / "others" / "zeroth_ko_test.parquet", columns=["audio"]).to_pylist()
    utts = []
    for r in rows:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if sr != SR:
            x = resample_poly(x, SR, sr).astype(np.float32)
        if len(x) >= 3 * SR:
            utts.append(x)
    emb = np.concatenate([engine.embed(x, np.array([[0, len(x) / SR]], np.float32)) for x in utts])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=0.45).fit_predict(emb)
    chosen: list[tuple[float, list[np.ndarray]]] = []
    centroids: list[np.ndarray] = []
    for k in np.argsort(-np.bincount(labels)):
        idx = np.where(labels == k)[0]
        if len(idx) < per:
            break
        c = emb[idx].mean(0)
        c /= np.linalg.norm(c)
        if c @ target > 0.6 or any(c @ o > 0.5 for o in centroids):
            continue  # 사실상 같은 사람으로 보이는 경우만 뺀다
        best = idx[np.argsort(-(emb[idx] @ c))][:per]
        chosen.append((float(c @ target), [utts[i] for i in best]))
        centroids.append(c)
        if len(chosen) == n:
            break
    chosen.sort(key=lambda g: -g[0])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    info = []
    for g, (sim, xs) in enumerate(chosen):
        files = []
        for i, x in enumerate(xs):
            f = f"spk{g}_{i}.wav"
            audio.write_wav(out_dir / f, x)
            files.append(f)
        info.append({"similarity": round(sim, 3), "files": files})
    meta.write_text(json.dumps(info, indent=2))
    return chosen


def seconds(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return x[int(a * SR):int(b * SR)]


def compose(*parts: tuple[str, np.ndarray], gap: float = 0.35) -> tuple[np.ndarray, list[dict]]:
    """화자 조각을 쉼을 두고 이어 붙이고, 누가 몇 초에 말했는지(정답)를 함께 돌려준다."""
    pause = np.zeros(int(gap * SR), np.float32)
    out, spans, t = [pause], [], gap
    for who, p in parts:
        p = (p / (np.sqrt(np.mean(p ** 2)) + 1e-9) * 0.08).astype(np.float32)  # 크기를 비슷하게
        spans.append({"speaker": who, "start": round(t, 3), "end": round(t + len(p) / SR, 3)})
        out += [p, pause]
        t += (len(p) + len(pause)) / SR
    return np.concatenate(out), spans


def room_noise(x: np.ndarray, level: float) -> np.ndarray:
    pink = np.cumsum(rng.standard_normal(len(x))).astype(np.float32)
    pink -= np.convolve(pink, np.ones(400) / 400, mode="same")  # 낮은 웅웅거림만 남긴다
    pink /= np.abs(pink).max() + 1e-9
    return x + level * pink + 0.002 * rng.standard_normal(len(x)).astype(np.float32)


def phone(x: np.ndarray) -> np.ndarray:
    sos = butter(6, [300, 3400], btype="band", fs=SR, output="sos")
    y = sosfiltfilt(sos, x)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1)  # 8kHz로 떨어뜨렸다가 되돌린다
    return np.clip(y * 1.6, -0.6, 0.6).astype(np.float32)


def save_m4a(name: str, wav: np.ndarray, created: str | None = None) -> Path:
    path = LIB / name
    meta = {"creation_time": created} if created else None
    audio.encode_m4a(wav, path, metadata=meta)
    return path


def save_video(name: str, wav: np.ndarray) -> Path:
    path = LIB / name
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "a.wav"
        audio.write_wav(src, wav)
        subprocess.run([audio.FFMPEG, "-nostdin", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "color=c=0x202020:s=320x240:r=10",
                        "-i", str(src), "-shortest", "-c:v", "libx264", "-tune", "stillimage",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(path)],
                       check=True)
    return path


def main() -> None:
    SRC.mkdir(exist_ok=True)
    LIB.mkdir(exist_ok=True)
    for f in LIB.iterdir():
        f.unlink()

    me, who = target_voice()
    print(f"찾을 사람 목소리: {who}")
    audio.write_wav(SRC / "참고음성_본인_7초.wav", seconds(me, 0, 7))  # 라이브러리엔 안 넣음

    libri = {n: fetch(n) for n in ["1.flac", "2.flac", "3.flac", "4.flac", "mlk.flac"]}
    voices = audio.decode(VOICES_SRC)

    engine = VoiceEngine(ROOT / "models", model="ecapa")  # 정답 세트를 만들 때 쓴 모델 그대로
    me_emb = engine.embed(me, np.array([[0, len(me) / SR]], np.float32))[0]
    (s_uncle, uncle), (s_cousin, cousin), (s_mom, mom), (s_friend, friend) = korean_speakers(engine, me_emb)
    print(f"찾을 사람과의 닮은 정도(문장 단위): 삼촌 {s_uncle:.2f} · 사촌 {s_cousin:.2f} · "
          f"엄마 {s_mom:.2f} · 친구 {s_friend:.2f}")

    made, truth = [], {}

    def add(path: Path, spans: list[dict]) -> None:
        made.append(path)
        truth[path.name] = spans

    # 2019: 녹음 정보에 날짜가 있는 추석 영상, 실내 잡음. 목소리가 닮은 삼촌이 바로 옆에서 말한다
    x, spans = compose(("삼촌", uncle[0]), ("찾을사람", seconds(me, 7, 14)), ("엄마", mom[0]),
                       ("영어화자1", seconds(libri["1.flac"], 0, 6)),
                       ("찾을사람", seconds(me, 14, 19)), ("삼촌", uncle[1]))
    add(save_m4a("2019 추석 가족모임.m4a", room_noise(x, 0.012), "2019-09-13T15:20:00Z"), spans)

    # 2021: 카카오톡 영상. 날짜는 파일 이름에만 있다
    x, spans = compose(("사촌", cousin[0]), ("찾을사람", seconds(me, 19, 26)),
                       ("영어화자2", seconds(libri["2.flac"], 0, 5)), ("사촌", cousin[1]))
    add(save_video("KakaoTalk_Video_2021-05-08-19-22-10.mp4", room_noise(x, 0.006)), spans)

    # 2023: 엄마와의 통화 녹음. 전화 음질, 날짜는 파일 이름
    x, spans = compose(("엄마", mom[1]), ("찾을사람", seconds(me, 26, 33.8)), ("엄마", mom[2]))
    add(save_m4a("통화녹음 2023.03.02.m4a", phone(x)), spans)

    # 2024: 찾을 사람이 없는 동창회 녹음. 삼촌·친구·사촌·영어 화자만 나온다
    x, spans = compose(("친구", friend[0]), ("삼촌", uncle[2]), ("사촌", cousin[2]),
                       ("영어화자3", voices), ("친구", friend[1]))
    add(save_m4a("2024 동창회.m4a", room_noise(x, 0.01), "2024-02-10T19:00:00Z"), spans)

    # 날짜 단서가 파일 시각뿐인 음성메모, 찾을 사람 없음
    x, spans = compose(("영어화자4", seconds(libri["4.flac"], 0, 7)),
                       ("영어화자5", seconds(libri["mlk.flac"], 0, 8)))
    memo = save_m4a("음성메모 001.m4a", room_noise(x, 0.004))
    t = time.mktime(datetime(2022, 6, 1, 10, 0).timetuple())
    os.utime(memo, (t, t))
    add(memo, spans)

    (SRC / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    for p in made:
        info = audio.probe(p)
        d = audio.recorded_date(p, info)
        who = sorted({sp["speaker"] for sp in truth[p.name]})
        print(f"{p.name:42s} {audio.duration_of(info):5.1f}초  {d.value} ({d.label})  {', '.join(who)}")


if __name__ == "__main__":
    main()
