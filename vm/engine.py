"""말소리 구간 검출(Silero VAD)과 화자 지문 추출.

긴 말소리 구간 하나에 여러 사람이 섞일 수 있으므로, 구간 전체가 아니라
1.5초 창을 0.75초씩 밀면서 창마다 지문을 뽑는다. 화자가 바뀌는 지점을
0.75초 단위로 따라갈 수 있다.

화자 지문 모델은 바꿔 끼울 수 있다. 기본은 VoxBlink2로 학습한 SimAM-ResNet34
(미세조정판)로, 한국어 문장 시험과 어려운 세트에서 ECAPA보다 나았다
(bench/korean_sv_round1.txt, bench/compare_models_round1.txt). 이 가중치는
비상업 라이선스 데이터(VoxBlink2, CC BY-NC-SA 4.0)로 학습되어, 상업적으로 쓰려면
VM_MODEL=ecapa(Apache-2.0)로 바꾼다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .audio import SR


@dataclass(frozen=True)
class Calibration:
    """모델마다 다른 '닮은 정도' 숫자 크기에 맞춘 기준값.

    ECAPA 값은 데모·어려운 세트 평가로 정했다(bench/tune_round1.txt). 다른 모델 값은
    tools/calibrate.py가 한국어 2초 조각에서 '다른 사람끼리 이 점수 이상이 나오는 비율'이
    ECAPA 값과 같아지는 점수로 환산한 것이다.
    """
    levels: tuple[float, ...]  # 담는 기준: 넉넉하게 … 엄격하게. 가운데가 기본
    cluster_distance: float    # 조각 지문 군집화 기준(코사인 거리)
    link: float                # 통화 녹음 군집을 일반 녹음 군집과 같은 사람으로 이을 기준
    same_person: float         # 참고 녹음과 이만큼 닮지 않은 군집은 다른 사람(경쟁자)으로 본다
    candidate_margin: float    # 후보는 기준보다 이만큼 낮춰 넉넉히 잡는다
    candidate_floor: float     # 후보 기준의 하한
    maybe_band: float          # 기준보다 이만큼 아래까지는 사람이 확인해 볼 구간으로 보여 준다
    rival_margin: float        # 경쟁자보다 이만큼은 더 닮아야 담는다
    same_person_sure: float | None = None  # 이만큼 닮으면 확실히 같은 사람. same_person과 사이는 사람에게 묻는다
    # 아래 둘은 실제 회의 녹음(AMI)에서 한 사람이 수십 묶음으로 쪼개지는 문제 때문에 넣었다. 실제 녹음과 합성 세트에서
    # 모두 확인한 기본 모델에만 켠다(ECAPA 한국어 모델은 켜면 다른 사람까지 이어 붙었다, README '실제 녹음 검증').
    merge: bool = False  # 쪼개진 같은 사람 묶음을 same_person_sure로 합친다(vm/speakers.py _merge_fragments)
    chain: bool = False  # 참고 녹음에서 '확실히 같은 목소리'와 확실히 같은 목소리도 같은 사람으로 따라간다

    def __post_init__(self):
        if self.same_person_sure is None:
            object.__setattr__(self, "same_person_sure", self.same_person)

    @property
    def threshold(self) -> float:
        return self.levels[len(self.levels) // 2]


# same_person_sure: 시험 세트(조건별 10명·데모·어려운 세트)에서 다른 사람 군집이 참고 녹음과 닮은 최댓값보다
# 조금 위로 정했다(bench/server/same_person_ko.txt). 그 아래는 사람에게 같은 사람인지 묻는다.
ECAPA_CAL = Calibration(levels=(0.35, 0.40, 0.45, 0.50, 0.55), cluster_distance=0.55, link=0.60,
                        same_person=0.50, candidate_margin=0.15, candidate_floor=0.20,
                        maybe_band=0.10, rival_margin=0.05, same_person_sure=0.56)

KO_MODEL = "voxblink2_samresnet34_ko"

# 앱에서 쓸 수 있는 모델: 이름 → (형식, 가중치 파일, 내려받을 곳, 기준값)
# 형식이 speechbrain이면 models/ecapa/, 나머지는 models/alt/<이름>/ 에 둔다.
MODELS = {
    "voxblink2_samresnet34_ft": ("samresnet", "avg_model.pt",
                                 ("gaunernst/wespeaker-voxblink2-samresnet34-ft",
                                  "0bfae5ffe2cc3e29379bb0f07511bac878b8c25b"),
                                 Calibration(levels=(0.476, 0.528, 0.578, 0.628, 0.685), cluster_distance=0.422,
                                             link=0.712, same_person=0.628, candidate_margin=0.154,
                                             candidate_floor=0.307, maybe_band=0.102, rival_margin=0.05,
                                             same_person_sure=0.74)),
    "ecapa": ("speechbrain", None, ("speechbrain/spkrec-ecapa-voxceleb", None), ECAPA_CAL),
    # ECAPA를 같은 방식으로 한국어 학습한 것(tools/finetune.py --base ecapa). 바탕(Apache-2.0)·학습 데이터
    # (Zeroth CC BY 4.0, RIRS_NOISES Apache-2.0)가 모두 상업적으로 쓸 수 있어 서비스용으로 쓴다.
    # 기준값은 models/alt/ecapa_ko/calibration.json(tools/calibrate.py, tools/same_person_stats.py).
    "ecapa_ko": ("speechbrain", "embedding_model.ckpt", ("speechbrain/spkrec-ecapa-voxceleb", None), None),
    # 위 모델을 한국어 음성으로 추가 학습한 것(tools/finetune.py). 내려받을 곳이 없어 models/alt/ 아래에 직접 둔다.
    # 기준값은 tools/calibrate.py 환산값에서 다섯 단계를 한 칸씩 엄격한 쪽으로 옮겼다(ECAPA 0.40~0.60에 해당).
    # 조건별 시험 세트에서 찾는 양은 거의 같고 잘못 담는 시간이 1/3~1/5로 줄었다
    # (bench/server/eval_cond_ko_l3_final.txt → eval_cond_ko_l3_level3.txt).
    KO_MODEL: ("samresnet", "avg_model.pt", (None, None),
               Calibration(levels=(0.575, 0.623, 0.674, 0.738, 0.784), cluster_distance=0.377, link=0.784,
                           same_person=0.674, candidate_margin=0.146, candidate_floor=0.345, maybe_band=0.096,
                           rival_margin=0.048, same_person_sure=0.76, merge=True, chain=True)),
}


def _default_model() -> str:
    """VM_MODEL을 따르고, 없으면 한국어 학습 모델, 그것도 없으면 VoxBlink2 미세조정판."""
    if os.environ.get("VM_MODEL"):
        return os.environ["VM_MODEL"]
    here = Path(__file__).resolve().parents[1] / "models" / "alt" / KO_MODEL / "avg_model.pt"
    return KO_MODEL if here.exists() else "voxblink2_samresnet34_ft"


DEFAULT_MODEL = _default_model()

# 비교 시험에만 쓰는 모델(tools/korean_sv.py). 가중치는 직접 받아 models/alt/<이름>/ 에 둔다.
BENCH_MODELS = {
    "wespeaker_vox_resnet34": ("onnx", "voxceleb_resnet34_LM.onnx"),
    "wespeaker_vox_resnet293": ("onnx", "voxceleb_resnet293_LM.onnx"),
    "wespeaker_vox_campplus": ("onnx", "voxceleb_CAM++.onnx"),
    "wespeaker_cnceleb_resnet34": ("onnx", "cnceleb_resnet34_LM.onnx"),
    "campplus_zh200k": ("onnx", "campplus_zh_cn_common_200k.onnx"),
    "campplus_cnen200k": ("onnx", "campplus_cn_en_common_200k.onnx"),
    "voxblink2_samresnet34": ("samresnet", "avg_model.pt"),
    # tools/finetune.py로 한국어 추가 학습한 것(뒤쪽 층만 / 전체)
    "voxblink2_samresnet34_ko_l3": ("samresnet", "avg_model.pt"),
    "voxblink2_samresnet34_ko_all": ("samresnet", "avg_model.pt"),
    "voxblink2_samresnet34_ko_all_hard": ("samresnet", "avg_model.pt"),  # 전체 층 + 더 시끄러운 조건
    "voxblink2_samresnet34_ko_l3_hard": ("samresnet", "avg_model.pt"),   # 뒤쪽 층 + 더 시끄러운 조건
}

WIN = 1.5      # 지문을 뽑는 창 길이(초)
HOP = 0.75     # 창 이동 간격(초)
MIN_WIN = 0.8  # 이보다 짧은 말소리 조각은 지문이 불안정해 버린다
EMB_DIM = 192  # ECAPA 지문 길이


def _fbank(x: np.ndarray) -> torch.Tensor:
    """WeSpeaker·3D-Speaker 학습 때와 같은 80차원 Kaldi fbank + 평균 빼기."""
    import torchaudio.compliance.kaldi as kaldi
    f = kaldi.fbank(torch.from_numpy(x).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25,
                    frame_shift=10, dither=0.0, sample_frequency=SR, window_type="hamming")
    return f - f.mean(0, keepdim=True)


class _FbankModel:
    """fbank를 입력으로 받는 화자 지문 모델(ONNX 또는 SimAM-ResNet)."""

    def __init__(self, kind: str, path: Path, device: str):
        self.batch = 8 if device == "cpu" else 64  # CPU는 배치를 키워도 빨라지지 않았다(bench/speed_round2.txt)
        if kind == "onnx":
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            self._run = lambda feats: sess.run(None, {name: feats.numpy()})[0]
            return
        import yaml

        from .samresnet import SimAM_ResNet34_ASP
        cfg = yaml.safe_load((path.parent / "config.yaml").read_text())
        net = SimAM_ResNet34_ASP(**cfg["model_args"])
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval().to(device)

        def run(feats: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                return net(feats.to(device)).cpu().numpy()
        self._run = run

    def embed(self, segs: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(segs), 0), np.float32)
        feats = [_fbank(x) for x in segs]
        by_len: dict[int, list[int]] = {}
        for i, f in enumerate(feats):
            by_len.setdefault(f.shape[0], []).append(i)  # 길이가 같은 것끼리 묶어 덧채움 없이 계산
        for same in by_len.values():
            for k in range(0, len(same), self.batch):  # 긴 녹음은 창이 수천 개라 나눠서 계산
                idx = same[k:k + self.batch]
                e = self._run(torch.stack([feats[i] for i in idx]))
                if out.shape[1] == 0:
                    out = np.zeros((len(segs), e.shape[1]), np.float32)
                out[idx] = e
        return out


def _load_cal(path: Path) -> Calibration:
    """tools/calibrate.py와 tools/same_person_stats.py가 남긴 기준값."""
    import json
    if not path.exists():
        raise FileNotFoundError(f"기준값 파일이 없어요: {path}. tools/calibrate.py로 만들어 주세요.")
    d = json.loads(path.read_text())
    return Calibration(**{**d, "levels": tuple(d["levels"])})


def _fetch(repo: str, revision: str | None, dest: Path, fname: str) -> None:
    """가중치가 없으면 처음 한 번 내려받는다."""
    if (dest / fname).exists():
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=dest)


class VoiceEngine:
    """models_dir 아래에 모델별 가중치를 둔다(없으면 처음 쓸 때 내려받음)."""

    def __init__(self, models_dir: str | Path, device: str = "cpu", model: str = DEFAULT_MODEL,
                 require_cal: bool = True):
        """require_cal=False는 기준값을 만드는 도구(tools/calibrate.py)용이다. 기준값 파일이 없어도 불러오고 cal은 None."""
        self.models_dir = Path(models_dir)
        self.model = model
        self.alt = None
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        self.vad = load_silero_vad()
        if model in MODELS:
            kind, fname, (repo, revision), self.cal = MODELS[model]
            cal_file = self.models_dir / "alt" / model / "calibration.json"
            if self.cal is None and (require_cal or cal_file.exists()):  # 학습한 모델: tools/calibrate.py가 남긴 파일
                self.cal = _load_cal(cal_file)
        else:
            kind, fname = BENCH_MODELS[model]
            repo = revision = self.cal = None  # 기준값이 없으면 비교 시험에만 쓴다
            cal_file = self.models_dir / "alt" / model / "calibration.json"
            if cal_file.exists():
                self.cal = _load_cal(cal_file)
        if kind == "speechbrain":
            from speechbrain.inference.speaker import EncoderClassifier
            self.model_dir = self.models_dir / "ecapa"
            self.dim = EMB_DIM
            # SpeechBrain 1.1은 처음부터 mps로 불러오면 오류가 나서, CPU로 불러온 뒤 옮긴다.
            self.encoder = EncoderClassifier.from_hparams(
                source=repo, savedir=str(self.model_dir), run_opts={"device": "cpu"})
            if fname:  # 추가 학습한 지문 부분만 바꿔 끼운다
                tuned = self.models_dir / "alt" / model / fname
                if not tuned.exists():
                    raise FileNotFoundError(f"{tuned}가 없어요. tools/finetune.py --base ecapa로 학습하세요.")
                self.encoder.mods.embedding_model.load_state_dict(torch.load(tuned, map_location="cpu"))
            if device != "cpu":
                self.encoder.mods.to(device)
                self.encoder.device = device
            self.device = device
            return
        self.model_dir = self.models_dir / "alt" / model
        if repo:
            _fetch(repo, revision, self.model_dir, fname)
        elif not (self.model_dir / fname).exists():
            raise FileNotFoundError(f"{self.model_dir / fname}가 없어요. tools/finetune.py로 학습하거나 "
                                    f"VM_MODEL=voxblink2_samresnet34_ft로 기존 모델을 쓰세요.")
        self.device = device if kind == "samresnet" else "cpu"
        self.alt = _FbankModel(kind, self.model_dir / fname, self.device)
        self.dim = int(self.alt.embed([np.zeros(SR, np.float32)]).shape[1])

    def speech_regions(self, wav: np.ndarray) -> list[tuple[float, float]]:
        if len(wav) < SR * MIN_WIN:
            return []
        stamps = get_speech_timestamps(
            torch.from_numpy(wav), self.vad, sampling_rate=SR,
            min_speech_duration_ms=300, min_silence_duration_ms=200,
            speech_pad_ms=60, return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in stamps]

    @staticmethod
    def windows(regions: list[tuple[float, float]]) -> np.ndarray:
        out: list[tuple[float, float]] = []
        for s, e in regions:
            length = e - s
            if length < MIN_WIN:
                continue
            if length <= WIN:
                out.append((s, e))
                continue
            t = s
            while t + WIN <= e + 1e-6:
                out.append((t, t + WIN))
                t += HOP
            if out[-1][1] < e - 0.25:  # 꼬리 부분도 덮는다
                out.append((e - WIN, e))
        return np.asarray(out, dtype=np.float32).reshape(-1, 2)

    def embed(self, wav: np.ndarray, windows: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """창마다 화자 지문을 뽑아 길이 1로 정규화해 돌려준다.

        배치는 8로 둔다. torch 2.12 CPU에서 배치를 16 이상으로 키우면 같은 양이
        100배 넘게 느려지는 현상을 실측했다(12까지는 정상).
        """
        if len(windows) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.alt is not None:
            emb = self.alt.embed([wav[int(s * SR):int(e * SR)] for s, e in windows]).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            return emb
        chunks = []
        for i in range(0, len(windows), batch_size):
            segs = [wav[int(s * SR):int(e * SR)] for s, e in windows[i:i + batch_size]]
            longest = max(len(x) for x in segs)
            batch = np.zeros((len(segs), longest), dtype=np.float32)
            rel = np.zeros(len(segs), dtype=np.float32)
            for j, x in enumerate(segs):
                batch[j, :len(x)] = x
                rel[j] = len(x) / longest
            with torch.no_grad():
                e = self.encoder.encode_batch(
                    torch.from_numpy(batch).to(self.device),
                    wav_lens=torch.from_numpy(rel).to(self.device),
                )
            chunks.append(e.squeeze(1).cpu().numpy())
        emb = np.concatenate(chunks).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb

    def analyze_wave(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = self.windows(self.speech_regions(wav))
        return win, self.embed(wav, win)
