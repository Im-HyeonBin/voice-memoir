"""겹쳐 말하는 구간 시험: 대상 화자 추출(TSE)이 찾을 사람 목소리를 얼마나 살리고, 얼마나 잘못 만들어 내는지.

Zeroth-Korean 테스트 화자 10명(화자 지문 학습·TSE 학습 모두에 안 씀)으로 4초 섞인 소리를 만든다.
- 겹침: 찾을 사람 문장 + 다른 사람 문장을 동시에. 찾을 사람 세기는 다른 사람보다 +5 / 0 / −5dB
- 함정: 찾을 사람 없이 다른 두 사람만 동시에
각 섞인 소리의 가운데 2초(앱이 판단하는 길이)를 원본 그대로, 또는 TSE로 찾을 사람 목소리만 뽑은 뒤 앱과 같은
기준으로 판정한다: 찾을 사람과 닮은 정도 ≥ 기준, 그리고 함께 말한 사람보다 여유 이상 더 닮음.
- 살린 비율: 겹침에서 통과한 비율(높을수록 좋음)
- 잘못 만든 비율: 함정에서 통과한 비율(낮을수록 좋음)
--env room이면 섞인 소리 속 사람마다 실측 방 울림을 입히고 DEMAND 카페 소음(10dB)을 깐다(학습에 안 쓴 소리).
참고 녹음(7초)은 찾을 사람의 다른 문장이다. 함께 말한 사람의 지문은 그 사람의 다른 문장에서 뽑는다(앱에서 경쟁자로 잡힌 상황).

TSE 시험은 ESPnet이 깔린 환경에서 돌린다(서버 ~/tse_env).
    python tools/overlap_bench.py --tse ~/tse_models/td_speakerbeam/exp/enh_train_raw
    python tools/overlap_bench.py --tse <폴더> --ckpt <추가 학습한 가중치.pth>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from korean_sv import load  # noqa: E402
from vm.audio import SR  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402
from vm.speakers import _sim, target_from_reference  # noqa: E402

SEG = 4 * SR
SIRS = (5, 0, -5)
PER = 12  # 찾을 사람마다 조건별 섞인 소리 수


def norm(x: np.ndarray, rms: float = 0.05) -> np.ndarray:
    return (x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def crop(x: np.ndarray, n: int, rng) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    a = int(rng.integers(0, len(x) - n + 1))
    return x[a:a + n]


def build(seed: int = 0, env: str = "clean"):
    """찾을 사람별 참고 녹음과 겹침·함정 섞인 소리."""
    utts, spk = load()
    rng = np.random.default_rng(seed)
    rirs = C.rir_paths("test") if env == "room" else ()
    room = (lambda x: C.apply_rir(x, C.load_rir(rirs[int(rng.integers(len(rirs)))]))) if env == "room" else (lambda x: x)  # noqa: E731
    cafe = C.demand("카페") if env == "room" else ()
    bed = (lambda x: C.add_noise(x, cafe[int(rng.integers(len(cafe)))], 10, rng)) if env == "room" else (lambda x: x)  # noqa: E731
    speakers = sorted(set(spk.tolist()))
    by = {s: [utts[i] for i in np.where(spk == s)[0]] for s in speakers}
    cases = []
    for s in speakers:
        mine = list(by[s])
        rng.shuffle(mine)
        ref = []
        while sum(len(x) for x in ref) < 7 * SR:
            ref.append(mine.pop())
        enroll = norm(np.concatenate(ref))
        others = [o for o in speakers if o != s]
        for sir in SIRS:
            for _ in range(PER):
                o = others[int(rng.integers(len(others)))]
                t = room(norm(crop(mine[int(rng.integers(len(mine)))], SEG, rng)))
                i = room(norm(crop(by[o][int(rng.integers(len(by[o])))], SEG, rng), 0.05 * 10 ** (-sir / 20)))
                rival = norm(by[o][int(rng.integers(len(by[o])))])
                cases.append(("겹침", sir, s, enroll, bed(t + i), t, rival))
        for _ in range(PER):
            a, b = rng.choice(others, 2, replace=False)
            x = bed(room(norm(crop(by[a][int(rng.integers(len(by[a])))], SEG, rng))) +
                    room(norm(crop(by[b][int(rng.integers(len(by[b])))], SEG, rng))))
            rival = norm(by[a][int(rng.integers(len(by[a])))])
            cases.append(("함정", None, s, enroll, x, None, rival))
    return cases


def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    proj = (est @ ref) / (ref @ ref + 1e-9) * ref
    return float(10 * np.log10((proj @ proj) / ((est - proj) @ (est - proj)) + 1e-9))


class Extractor:
    """ESPnet TD-SpeakerBeam. ckpt를 주면 추가 학습한 가중치로 바꿔 끼운다."""

    def __init__(self, folder: str, ckpt: str | None, device: str):
        import torch
        from espnet2.bin.enh_tse_inference import SeparateSpeech
        folder = Path(folder).expanduser()
        self.sep = SeparateSpeech(train_config=str(folder / "config.yaml"), model_file=str(folder / "99epoch.pth"),
                                  normalize_output_wav=True, device=device)
        if ckpt:
            self.sep.enh_model.load_state_dict(torch.load(ckpt, map_location=device))

    def __call__(self, mix: np.ndarray, enroll: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self.sep(mix[None, :], fs=SR, enroll_ref1=enroll[None, :])
        return np.asarray(out[0]).reshape(-1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="voxblink2_samresnet34_ko")
    ap.add_argument("--tse", help="ESPnet TD-SpeakerBeam 폴더(config.yaml, 99epoch.pth)")
    ap.add_argument("--ckpt", help="추가 학습한 TSE 가중치")
    ap.add_argument("--device", default="cpu", help="화자 지문 모델 장치")
    ap.add_argument("--env", choices=("clean", "room"), default="clean")
    args = ap.parse_args()
    eng = VoiceEngine(ROOT / "models", args.device, args.model)
    thr, margin = eng.cal.threshold, eng.cal.rival_margin
    ext = Extractor(args.tse, args.ckpt, "cuda") if args.tse else None
    mid = np.array([[1.0, 3.0]], np.float32)  # 4초 중 가운데 2초
    protos = {}
    rows = {}
    for kind, sir, s, enroll, mix, clean, rival in build(env=args.env):
        if s not in protos:
            protos[s] = target_from_reference(eng, enroll).wide
        r = eng.embed(rival, np.array([[0, len(rival) / SR]], np.float32))
        variants = {"원본": mix}
        if ext:
            variants["추출"] = ext(norm(mix), enroll)
        for name, x in variants.items():
            e = eng.embed(norm(x), mid)
            sim = float(_sim(e, protos[s])[0])
            ok = sim >= thr and sim - float((e @ r.T).max()) >= margin
            key = (name, kind, sir)
            rows.setdefault(key, []).append(ok)
            if name == "추출" and clean is not None:
                rows.setdefault(("SI-SNR 향상", kind, sir), []).append(si_snr(x, clean) - si_snr(mix, clean))
    print(f"조건 {args.env}, 모델 {args.model}, 기준 {thr}, 여유 {margin}" + (f", TSE {Path(args.ckpt).name if args.ckpt else '공개 가중치'}" if ext else ""))
    for name in ("원본", "추출"):
        if not any(k[0] == name for k in rows):
            continue
        cells = [f"겹침 {sir:+d}dB 살림 {np.mean(rows[(name, '겹침', sir)]):5.1%}" for sir in SIRS]
        cells.append(f"함정 잘못 만듦 {np.mean(rows[(name, '함정', None)]):5.1%}")
        print(f"{name}: " + " | ".join(cells))
    if ext:
        print("추출 음질(SI-SNR 향상, dB): " + " | ".join(f"{sir:+d}dB {np.mean(rows[('SI-SNR 향상', '겹침', sir)]):5.1f}" for sir in SIRS))


if __name__ == "__main__":
    main()
