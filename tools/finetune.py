"""한국어 음성으로 화자 지문 모델(SimAM-ResNet34, VoxBlink2 미세조정판)을 추가 학습한다.

가르치는 것
1. 소음·울림에 흔들리지 않기: 원래 모델(선생)이 깨끗한 2초 조각에서 뽑은 지문을, 같은 조각에
   소음·웅성거림·방 울림·전화 음질을 입혔을 때도 학생 모델이 그대로 뽑아내게 한다.
   선생 지문을 목표로 삼기 때문에 지문 공간이 원래 모델과 크게 달라지지 않는다.
2. 한국인 목소리 구분: Zeroth-Korean 학습 화자 95명을 서로 구분하는 학습(AAM-softmax)을 조금 더한다.

--train-from으로 학습할 층을 고른다. 기본(layer3)은 앞쪽 층(conv1~layer2)을 고정하고 layer3부터
학습해서 메모리 8GB 맥(M1)에서도 돈다. GPU 메모리가 넉넉하면(GTX 1070 8GB 등) stem으로 전체를 학습한다.
배치 정규화 통계는 고정한다. 학습 화자 중 10명은 빼 두었다가 매 회차 끝에 검증한다.
학습에 쓰는 소음·울림은 tools/conditions.py의 학습용(시뮬레이션 울림, freesound 소음)뿐이다.

    python tools/prep_zeroth.py      # 한 번
    python tools/finetune.py --epochs 3
    python tools/finetune.py --epochs 3 --train-from stem --batch 32   # GPU 서버에서 전체 학습
    python tools/finetune.py --base ecapa --out ecapa_ko --epochs 4 --batch 32   # 상업적으로 쓸 수 있는 ECAPA 바탕

--base ecapa는 SpeechBrain ECAPA-TDNN(Apache-2.0, VoxCeleb 학습)을 같은 방식으로 학습한다. 비상업 조건이 붙은
VoxBlink2 바탕 모델 대신 상업 서비스에 쓸 수 있는 한국어 모델을 만들려는 것이다. ECAPA는 작아서 전체를 학습한다.
"""
from __future__ import annotations

import argparse
import copy
import queue
import shutil
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import conditions as C  # noqa: E402
from korean_sv import eer  # noqa: E402
from vm.audio import SR, telephone_band  # noqa: E402
from vm.engine import _fbank  # noqa: E402
from vm.samresnet import SimAM_ResNet34_ASP  # noqa: E402

SRC = ROOT / "models" / "alt" / "voxblink2_samresnet34_ft"
OUT = ROOT / "models" / "alt" / "voxblink2_samresnet34_ko"
ZEROTH = ROOT / "data" / "zeroth"
CROP = int(2.0 * SR)
VAL_SPEAKERS = 10
STAGES = ["stem", "layer1", "layer2", "layer3", "layer4"]
KINDS = ["깨끗함", "소음", "웅성거림", "울림", "울림+소음", "울림+웅성거림", "전화"]
KIND_P = [0.15, 0.20, 0.15, 0.20, 0.15, 0.10, 0.05]


class Data:
    snr_noise = (0.0, 15.0)   # 소음 세기: 말소리보다 이만큼(dB) 작게
    snr_babble = (0.0, 10.0)  # 웅성거림 세기

    def __init__(self):
        idx = np.load(ZEROTH / "train_index.npz")
        self.audio = np.memmap(ZEROTH / "train_audio.int16", np.int16, "r")
        self.off, self.len, self.spk = idx["offsets"], idx["lengths"], idx["speakers"]
        speakers = np.array(sorted(set(self.spk.tolist())))
        val = set(np.random.default_rng(0).choice(speakers, VAL_SPEAKERS, replace=False).tolist())
        self.val_idx = np.where(np.isin(self.spk, list(val)))[0]
        self.train_idx = np.where(~np.isin(self.spk, list(val)))[0]
        self.labels = {s: i for i, s in enumerate(sorted(set(speakers.tolist()) - val))}
        self.rirs = C.rir_paths("train")
        self.noises = C.train_noise_paths()

    def crop(self, u: int, rng: np.random.Generator, center: bool = False) -> np.ndarray:
        a = (self.len[u] - CROP) // 2 if center else int(rng.integers(0, self.len[u] - CROP + 1))
        s = self.off[u] + a
        return self.audio[s:s + CROP].astype(np.float32) / 32768

    @lru_cache(maxsize=1024)
    def noise(self, i: int) -> np.ndarray:
        return C._mono(self.noises[i])

    def babble(self, rng: np.random.Generator, exclude: int) -> np.ndarray:
        out = np.zeros(CROP, np.float32)
        for u in rng.choice(self.train_idx, int(rng.integers(3, 7))):
            if self.spk[u] != exclude:
                out += C.level(self.crop(int(u), rng))
        return out

    def augment(self, x: np.ndarray, spk: int, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
        kind = kind or str(rng.choice(KINDS, p=KIND_P))
        y = x
        if "울림" in kind:
            y = C.apply_rir(y, C.load_rir(self.rirs[int(rng.integers(len(self.rirs)))]))
        if "소음" in kind:
            y = C.add_noise(y, self.noise(int(rng.integers(len(self.noises)))), rng.uniform(*self.snr_noise), rng)
        if "웅성거림" in kind:
            y = C.add_noise(y, self.babble(rng, spk), rng.uniform(*self.snr_babble), rng)
        if kind == "전화":
            y = telephone_band(y)
            if rng.random() < 0.5:
                y = C.add_noise(y, self.noise(int(rng.integers(len(self.noises)))), rng.uniform(5, 20), rng)
        return y.astype(np.float32)


def batches(data: Data, batch: int, steps: int, seed: int):
    """다른 스레드에서 배치를 미리 만들어 둔다(GPU가 쉬지 않게)."""
    q: queue.Queue = queue.Queue(maxsize=6)

    def work():
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            us = rng.choice(data.train_idx, batch)
            clean, inp, lab = [], [], []
            for u in us:
                x = data.crop(int(u), rng)
                clean.append(x)
                inp.append(data.augment(x, int(data.spk[u]), rng))
                lab.append(data.labels[int(data.spk[u])])
            q.put((torch.from_numpy(np.stack(clean)), torch.from_numpy(np.stack(inp)), torch.tensor(lab)))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


class Model:
    """고정한 앞쪽 층(trunk)은 선생·학생이 같이 쓰고, 그 뒤(head)는 학생만 따로 학습한다."""

    def __init__(self, device: str, train_from: str = "layer3"):
        cfg = yaml.safe_load((SRC / "config.yaml").read_text())
        self.teacher = SimAM_ResNet34_ASP(**cfg["model_args"])
        self.teacher.load_state_dict(torch.load(SRC / "avg_model.pt", map_location="cpu"))
        self.teacher.eval().to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = copy.deepcopy(self.teacher).eval()  # eval: 배치 정규화 통계 고정
        self.k = STAGES.index(train_from)
        f = self.student.front
        mods = {"stem": [f.conv1, f.bn1], "layer1": [f.layer1], "layer2": [f.layer2],
                "layer3": [f.layer3], "layer4": [f.layer4]}
        self.head_params = []
        for name in STAGES[self.k:]:
            for m in mods[name]:
                self.head_params += list(m.parameters())
        self.head_params += list(self.student.pooling.parameters()) + list(self.student.bottleneck.parameters())
        for p in self.head_params:
            p.requires_grad_(True)
        self.device = device

    @staticmethod
    def stage(net, name: str, x: torch.Tensor) -> torch.Tensor:
        f = net.front
        return f.relu(f.bn1(f.conv1(x))) if name == "stem" else getattr(f, name)(x)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device).permute(0, 2, 1).unsqueeze(1)
            for name in STAGES[:self.k]:
                x = self.stage(self.teacher, name, x)
            return x

    def head(self, net, h: torch.Tensor) -> torch.Tensor:
        for name in STAGES[self.k:]:
            h = self.stage(net, name, h)
        return net.bottleneck(net.pooling(h))

    def embed(self, net, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.head(net, self.trunk(x)), dim=1)

    @staticmethod
    def feats(wav: torch.Tensor) -> torch.Tensor:
        """소리(배치, 샘플) → 80차원 fbank(배치, 프레임, 80)."""
        return torch.stack([_fbank(x) for x in wav.numpy()])

    @property
    def dim(self) -> int:
        return self.teacher.bottleneck.out_features


class EcapaModel:
    """SpeechBrain ECAPA-TDNN을 선생·학생으로 쓴다. 특징 추출과 정규화는 학습하지 않는다."""

    def __init__(self, device: str):
        from speechbrain.inference.speaker import EncoderClassifier
        enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir=str(ROOT / "models" / "ecapa"), run_opts={"device": "cpu"})
        self.compute = enc.mods.compute_features.to(device)
        self.norm = enc.mods.mean_var_norm.to(device)
        self.teacher = enc.mods.embedding_model.eval().to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = copy.deepcopy(self.teacher).eval()  # eval: 배치 정규화 통계 고정
        self.head_params = list(self.student.parameters())
        for p in self.head_params:
            p.requires_grad_(True)
        self.device = device

    def feats(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wav = wav.to(self.device)
            f = self.compute(wav)
            return self.norm(f, torch.ones(wav.shape[0], device=self.device))

    @staticmethod
    def trunk(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def head(net, x: torch.Tensor) -> torch.Tensor:
        return net(x).squeeze(1)

    def embed(self, net, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.head(net, x), dim=1)

    @property
    def dim(self) -> int:
        return self.teacher.fc.conv.out_channels


def aam_logits(e: torch.Tensor, w: torch.Tensor, y: torch.Tensor, m: float = 0.2, s: float = 30.0) -> torch.Tensor:
    cos = e @ F.normalize(w, dim=1).T
    sin = torch.sqrt((1 - cos ** 2).clamp(min=1e-7))
    phi = cos * np.cos(m) - sin * np.sin(m)
    phi = torch.where(cos > np.cos(np.pi - m), phi, cos - np.sin(np.pi - m) * m)
    onehot = F.one_hot(y, cos.shape[1]).to(cos.dtype)
    return s * (onehot * phi + (1 - onehot) * cos)


def validate(model: Model, net, data: Data) -> dict:
    """빼 둔 화자 10명: 깨끗한 조각끼리, 깨끗한 조각 ↔ 학습용 조건을 입힌 조각."""
    rng = np.random.default_rng(7)
    pools = [data.val_idx[data.spk[data.val_idx] == s] for s in sorted(set(data.spk[data.val_idx].tolist()))]
    us = np.concatenate([rng.choice(p, min(20, len(p)), replace=False) for p in pools])
    spk = data.spk[us]
    clean_x = [data.crop(int(u), rng, center=True) for u in us]
    feats = lambda xs: model.feats(torch.from_numpy(np.stack(xs)))  # noqa: E731
    emb = lambda xs: torch.cat([model.embed(net, feats(xs[i:i + 32])) for i in range(0, len(xs), 32)]).cpu().numpy()  # noqa: E731
    clean = emb(clean_x)
    same, triu, off = spk[:, None] == spk[None, :], np.triu(np.ones((len(us),) * 2, bool), 1), ~np.eye(len(us), dtype=bool)
    out = {"깨끗함": eer((clean @ clean.T)[same & triu], (clean @ clean.T)[~same & triu])}
    saved = Data.snr_noise, Data.snr_babble
    Data.snr_noise, Data.snr_babble = (0.0, 15.0), (0.0, 10.0)  # 학습 설정이 달라도 같은 조건으로 잰다
    for kind in ("소음", "웅성거림", "울림", "울림+소음"):
        r = np.random.default_rng(11)
        deg = emb([data.augment(x, int(s), r, kind) for x, s in zip(clean_x, spk)])
        out[kind] = eer((clean @ deg.T)[same & off], (clean @ deg.T)[~same & off])
    Data.snr_noise, Data.snr_babble = saved
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--aam", type=float, default=0.2, help="화자 구분 학습의 비중")
    ap.add_argument("--steps", type=int, default=0, help="회차당 걸음 수(0이면 학습 문장 수 / 배치)")
    ap.add_argument("--train-from", choices=STAGES, default="layer3", help="이 층부터 끝까지 학습")
    ap.add_argument("--out", default=OUT.name, help="결과를 둘 models/alt/ 아래 폴더 이름")
    ap.add_argument("--resume", action="store_true", help="중간에 끊겼으면 마지막으로 끝난 회차 다음부터 잇는다")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="이 회차까지만 하고 멈춘다(서버를 쉬어 가며 돌릴 때 --resume과 함께). 학습률 일정은 --epochs 기준")
    ap.add_argument("--snr-noise", default="0,15", help="소음 세기 범위(dB, 낮을수록 시끄러움)")
    ap.add_argument("--snr-babble", default="0,10", help="웅성거림 세기 범위(dB)")
    ap.add_argument("--threads", type=int, default=2, help="CPU 스레드 수(서버 부담 줄이기)")
    ap.add_argument("--base", choices=("samresnet", "ecapa"), default="samresnet",
                    help="바탕 모델. ecapa는 상업적으로 쓸 수 있는 라이선스")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    out = OUT.parent / args.out
    Data.snr_noise = tuple(float(v) for v in args.snr_noise.split(","))
    Data.snr_babble = tuple(float(v) for v in args.snr_babble.split(","))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = Data()
    model = EcapaModel(device) if args.base == "ecapa" else Model(device, args.train_from)
    steps = args.steps or len(data.train_idx) // args.batch
    print(f"학습 문장 {len(data.train_idx)}개(화자 {len(data.labels)}명), 검증 화자 {VAL_SPEAKERS}명, "
          f"회차당 {steps}걸음 × {args.epochs}회차, 배치 {args.batch}, "
          f"{'ECAPA 전체' if args.base == 'ecapa' else args.train_from + '부터'} 학습, 장치 {device}, "
          f"소음 {Data.snr_noise}dB 웅성거림 {Data.snr_babble}dB", flush=True)

    # 화자 구분 층은 선생 지문의 화자별 평균에서 시작한다
    rng = np.random.default_rng(1)
    w = torch.zeros(len(data.labels), model.dim)
    for s, i in data.labels.items():
        pool = data.train_idx[data.spk[data.train_idx] == s]
        us = rng.choice(pool, min(8, len(pool)), replace=False)
        crops = torch.from_numpy(np.stack([data.crop(int(u), rng) for u in us]))
        w[i] = model.embed(model.teacher, model.feats(crops)).mean(0).cpu()
    w = torch.nn.Parameter(w.to(device))

    opt = torch.optim.AdamW([{"params": model.head_params, "lr": args.lr}, {"params": [w], "lr": 1e-3}],
                            weight_decay=1e-4)
    total = steps * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))

    out.mkdir(parents=True, exist_ok=True)
    if args.base == "samresnet":
        shutil.copy(SRC / "config.yaml", out / "config.yaml")
    weights = "embedding_model.ckpt" if args.base == "ecapa" else "avg_model.pt"  # 앱(vm/engine.py)이 읽는 이름
    state_file, start = out / "state.pt", 1
    if args.resume and state_file.exists():  # 서버가 멈춰도 끝난 회차까지는 살린다
        st = torch.load(state_file, map_location=device, weights_only=False)  # 이 스크립트가 만든 파일
        model.student.load_state_dict(st["student"])
        w.data.copy_(st["w"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"회차 {st['epoch']}까지 끝난 상태에서 이어서 학습", flush=True)
    else:
        base = validate(model, model.teacher, data)
        print("검증(학습 전): " + ", ".join(f"{k} {v:.2%}" for k, v in base.items()), flush=True)
        best = sum(base.values())
    last = args.stop_after or args.epochs
    for ep in range(start, min(last, args.epochs) + 1):
        t0, k, run = time.time(), 0, np.zeros(3)
        for clean, inp, lab in batches(data, args.batch, steps, seed=ep):
            clean, inp = model.feats(clean), model.feats(inp)
            hc = model.trunk(clean)
            with torch.no_grad():
                target = F.normalize(model.head(model.teacher, hc), dim=1)
            e = F.normalize(model.head(model.student, model.trunk(inp)), dim=1)
            l_dist = (1 - (e * target).sum(1)).mean()
            l_aam = F.cross_entropy(aam_logits(e, w, lab.to(device)), lab.to(device))
            loss = l_dist + args.aam * l_aam
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += [loss.item(), l_dist.item(), l_aam.item()]
            k += 1
            if k % 100 == 0:
                print(f"  회차 {ep} {k}/{steps}  손실 {run[0] / 100:.3f} (지문 {run[1] / 100:.3f}, 구분 {run[2] / 100:.2f})"
                      f"  {k * args.batch / (time.time() - t0):.1f}개/초", flush=True)
                run[:] = 0
        res = validate(model, model.student, data)
        score = sum(res.values())
        torch.save(model.student.state_dict(), out / f"epoch{ep}.pt")
        mark = ""
        if score < best:
            best = score
            torch.save(model.student.state_dict(), out / weights)
            mark = "  ← 지금까지 가장 좋음, 저장"
        torch.save({"student": model.student.state_dict(), "w": w.detach(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": float(best)}, state_file)
        print(f"검증(회차 {ep}, {(time.time() - t0) / 60:.0f}분): " + ", ".join(f"{k} {v:.2%}" for k, v in res.items()) + mark,
              flush=True)


if __name__ == "__main__":
    main()
