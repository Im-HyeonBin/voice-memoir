"""대상 화자 추출(TD-SpeakerBeam, ESPnet)을 한국어 대화 섞기로 추가 학습한다.

영어 낭독(LibriMix)으로 학습된 공개 가중치에서 시작해, 한국인 화자 95명(Zeroth 학습 세트, 화자 지문 학습과 같은
분할)의 목소리를 섞어 찾을 사람 목소리만 뽑아내도록 가르친다.
- 찾을 사람 4초 + 다른 사람 1~2명(세기 −6~+6dB, 절반은 일부 구간만 겹침)
- 30%는 방 울림(시뮬레이션), 30%는 소음(freesound, 5~20dB)을 입힌다(학습용 소리만, tools/conditions.py)
- 참고 녹음: 찾을 사람의 다른 문장 6초
- 손실: 뽑아낸 소리와 섞기 전 찾을 사람 소리 사이의 SI-SNR(높을수록 좋음)을 키운다
검증: 학습에서 뺀 화자 10명으로 만든 섞인 소리 160개의 SI-SNR 향상(dB).

ESPnet이 깔린 환경에서 돌린다(서버 ~/tse_env). 서버가 오래 쉬지 않고 돌면 멈춘 적이 있어 한 회차씩 끊어 돌린다.
    python tools/finetune_tse.py --tse ~/tse_models/td_speakerbeam/exp/enh_train_raw --epochs 6 --resume --stop-after 1
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from finetune import Data  # noqa: E402
from vm.audio import SR  # noqa: E402

SEG, ENR = 4 * SR, 6 * SR
OUT = ROOT / "models" / "tse_ko"


class Mixer:
    def __init__(self, data: Data):
        self.d = data

    def clip(self, u: int, n: int, rng) -> np.ndarray:
        L = int(self.d.len[u])
        a = int(rng.integers(0, max(1, L - n + 1)))
        x = self.d.audio[self.d.off[u] + a:self.d.off[u] + a + min(n, L)].astype(np.float32) / 32768
        return np.pad(x, (0, n - len(x))) if len(x) < n else x

    def sample(self, pool: np.ndarray, rng):
        d = self.d
        u = int(rng.choice(pool))
        s = d.spk[u]
        same = pool[(d.spk[pool] == s) & (pool != u)]
        e = int(rng.choice(same)) if len(same) else u
        reverb = rng.random() < 0.3
        room = lambda x: C.apply_rir(x, C.load_rir(d.rirs[int(rng.integers(len(d.rirs)))])) if reverb else x  # noqa: E731
        t = room(C.level(self.clip(u, SEG, rng)))
        mix = t.copy()
        for _ in range(1 if rng.random() < 0.75 else 2):
            v = int(rng.choice(pool[d.spk[pool] != s]))
            i = room(C.level(self.clip(v, SEG, rng), 0.05 * 10 ** (-rng.uniform(-6, 6) / 20)))
            if rng.random() < 0.5:  # 일부 구간만 겹침
                a = int(rng.integers(0, SEG - SR))
                b = int(rng.integers(a + SR, SEG + 1))
                i[:a] = 0
                i[b:] = 0
            mix += i
        if rng.random() < 0.3:
            mix = C.add_noise(mix, d.noise(int(rng.integers(len(d.noises)))), rng.uniform(5, 20), rng)
        k = 0.9 / (np.abs(mix).max() + 1e-9)
        return (mix * k).astype(np.float32), (t * k).astype(np.float32), C.level(self.clip(e, ENR, rng))


def batches(mixer: Mixer, pool, batch: int, steps: int, seed: int):
    q: queue.Queue = queue.Queue(maxsize=8)

    def work():
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            items = [mixer.sample(pool, rng) for _ in range(batch)]
            q.put(tuple(torch.from_numpy(np.stack(z)) for z in zip(*items)))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + 1e-8) * ref
    noise = est - proj
    return 10 * torch.log10((proj * proj).sum(-1) / ((noise * noise).sum(-1) + 1e-8) + 1e-8)


def extract(model, mix: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
    b = mix.shape[0]
    lens = torch.full((b,), mix.shape[1], dtype=torch.long, device=mix.device)
    elens = torch.full((b,), enroll.shape[1], dtype=torch.long, device=mix.device)
    out = model.forward_enhance(mix, lens, [enroll], [elens])[0][0]
    return out[:, :mix.shape[1]]


def validate(model, mixer: Mixer, pool, device) -> float:
    rng = np.random.default_rng(123)
    gains = []
    model.eval()
    with torch.no_grad():
        for _ in range(40):
            mix, t, e = (torch.from_numpy(np.stack(z)).to(device) for z in zip(*[mixer.sample(pool, rng) for _ in range(4)]))
            gains += (si_snr(extract(model, mix, e), t) - si_snr(mix, t)).cpu().tolist()
    model.train()
    return float(np.mean(gains))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tse", required=True, help="공개 TD-SpeakerBeam 폴더(config.yaml, 99epoch.pth)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1000, help="회차당 걸음 수")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--seg", type=float, default=4.0, help="학습 조각 길이(초). 짧을수록 GPU 메모리·계산이 준다")
    args = ap.parse_args()
    global SEG
    SEG = int(args.seg * SR)
    from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(args.tse).expanduser()
    model, _ = TargetSpeakerExtractionTask.build_model_from_file(folder / "config.yaml", folder / "99epoch.pth", device)
    model.train()
    data = Data()
    mixer = Mixer(data)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total = args.steps * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    OUT.mkdir(parents=True, exist_ok=True)
    state_file, start = OUT / "state.pt", 1
    if args.resume and state_file.exists():
        st = torch.load(state_file, map_location=device, weights_only=False)  # 이 스크립트가 만든 파일
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"회차 {st['epoch']}까지 끝난 상태에서 이어서 학습", flush=True)
    else:
        best = validate(model, mixer, data.val_idx, device)
        print(f"학습 문장 {len(data.train_idx)}개, 검증 화자 10명. 검증(학습 전): SI-SNR 향상 {best:.2f}dB", flush=True)
    last = min(args.stop_after or args.epochs, args.epochs)
    for ep in range(start, last + 1):
        t0, run, k = time.time(), 0.0, 0
        for mix, t, e in batches(mixer, data.train_idx, args.batch, args.steps, seed=1000 + ep):
            mix, t, e = mix.to(device), t.to(device), e.to(device)
            loss = -si_snr(extract(model, mix, e), t).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += loss.item()
            k += 1
            if k % 100 == 0:
                print(f"  회차 {ep} {k}/{args.steps}  SI-SNR {-run / 100:.2f}dB  {k * args.batch / (time.time() - t0):.1f}개/초", flush=True)
                run = 0.0
        gain = validate(model, mixer, data.val_idx, device)
        torch.save(model.state_dict(), OUT / f"epoch{ep}.pth")
        mark = ""
        if gain > best:
            best = gain
            torch.save(model.state_dict(), OUT / "best.pth")
            mark = "  ← 지금까지 가장 좋음, 저장"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": float(best)}, state_file)
        print(f"검증(회차 {ep}, {(time.time() - t0) / 60:.0f}분): SI-SNR 향상 {gain:.2f}dB{mark}", flush=True)


if __name__ == "__main__":
    main()
