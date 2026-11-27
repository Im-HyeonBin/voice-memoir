# 대상 화자 추출(TD-SpeakerBeam): 섞인 소리와 찾을 사람의 참고 녹음을 받아 그 사람 목소리만 뽑는다.
# ESPnet(https://github.com/espnet/espnet, Apache-2.0)의 espnet2/enh/encoder/conv_encoder.py,
# decoder/conv_decoder.py, extractor/td_speakerbeam_extractor.py, layers/tcn.py, layers/adapt_layers.py에서
# 이 앱이 쓰는 설정(비인과, gLN, 곱셈 적응층)만 옮겼다. 학습된 가중치를 그대로 읽게 층 이름은 원본과 같다.
# TCN 부분은 원래 kaituoxu/Conv-TasNet(MIT)에 바탕을 둔다.
#
# Copyright 2023 Wangyou Zhang (ESPnet), Licensed under the Apache License, Version 2.0
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EPS = torch.finfo(torch.float32).eps


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=(1, 2), keepdim=True)
        var = ((y - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=1, keepdim=True)
        var = y.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, hidden: int, bottleneck: int, skip: int, kernel: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel, padding=(kernel - 1) * dilation // 2, dilation=dilation,
                      groups=hidden, bias=False),
            nn.PReLU(), GlobalLayerNorm(hidden), nn.Conv1d(hidden, bottleneck, 1, bias=False))
        self.skip_conv = nn.Conv1d(hidden, skip, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.net[:-1](x)
        return self.net[-1](shared), self.skip_conv(shared)


class TemporalBlock(nn.Module):
    def __init__(self, bottleneck: int, hidden: int, skip: int, kernel: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(bottleneck, hidden, 1, bias=False), nn.PReLU(), GlobalLayerNorm(hidden),
                                 DepthwiseSeparableConv(hidden, bottleneck, skip, kernel, dilation))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)


class TemporalConvNet(nn.Module):
    """Conv-TasNet의 TCN. enroll을 주면 i_adapt번째 블록 출력에 화자 정보를 곱한다."""

    def __init__(self, N: int, B: int, H: int, P: int, X: int, R: int, out: int, mask: str):
        super().__init__()
        self.mask = mask
        repeats = [nn.Sequential(*[TemporalBlock(B, H, B, P, 2 ** x) for x in range(X)]) for _ in range(R)]
        self.network = nn.Sequential(ChannelwiseLayerNorm(N), nn.Conv1d(N, B, 1, bias=False),
                                     nn.Sequential(*repeats), nn.PReLU(), nn.Conv1d(B, out, 1, bias=False))

    def forward(self, x: torch.Tensor, enroll: torch.Tensor | None = None, i_adapt: int = -1) -> torch.Tensor:
        out = self.network[:2](x)
        skip_sum = 0.0
        idx = 0
        for block in self.network[2]:
            for layer in block:
                residual, skip = layer(out)
                if enroll is not None and idx == i_adapt:
                    e_res, e_skip = torch.chunk(enroll, 2, dim=1)
                    residual, skip = e_res[:, :, None] * residual, e_skip[:, :, None] * skip
                skip_sum = skip_sum + skip
                out = out + residual
                idx += 1
        score = self.network[3:](skip_sum)
        return torch.relu(score) if self.mask == "relu" else score


class _Encoder(nn.Module):
    def __init__(self, channel: int, kernel: int, stride: int):
        super().__init__()
        self.conv1d = nn.Conv1d(1, channel, kernel, stride=stride, bias=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv1d(wav.unsqueeze(1)))  # (배치, 채널, 프레임)


class _Decoder(nn.Module):
    def __init__(self, channel: int, kernel: int, stride: int):
        super().__init__()
        self.convtrans1d = nn.ConvTranspose1d(channel, 1, kernel, stride=stride, bias=False)

    def forward(self, feat: torch.Tensor, length: int) -> torch.Tensor:
        return self.convtrans1d(feat, output_size=(feat.shape[0], 1, length)).squeeze(1)


class _Extractor(nn.Module):
    def __init__(self, N: int, B: int, H: int, P: int, X: int, R: int, i_adapt: int):
        super().__init__()
        self.i_adapt = i_adapt
        self.tcn = TemporalConvNet(N, B, H, P, X, R, out=N, mask="relu")
        self.auxiliary_net = TemporalConvNet(N, B, H, P, X, 1, out=2 * B, mask="linear")

    def forward(self, feat: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        enroll = self.auxiliary_net(aux).mean(dim=-1)  # 참고 녹음 전체를 평균한 화자 정보
        return feat * self.tcn(feat, enroll, self.i_adapt)


class TDSpeakerBeam(nn.Module):
    """ESPnet 'Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam' 설정(인코더 256, 8층×4묶음)."""

    def __init__(self, N: int = 256, kernel: int = 32, stride: int = 16, B: int = 256, H: int = 512,
                 P: int = 3, X: int = 8, R: int = 4, i_adapt: int = 7):
        super().__init__()
        self.encoder = _Encoder(N, kernel, stride)
        self.extractor = _Extractor(N, B, H, P, X, R, i_adapt)
        self.decoder = _Decoder(N, kernel, stride)

    def forward(self, mix: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.extractor(self.encoder(mix), self.encoder(enroll)), mix.shape[-1])


class Extractor:
    """앱에서 쓰는 꼴: 섞인 소리(넘파이)와 참고 녹음으로 찾을 사람 목소리를 뽑는다.

    key(어느 녹음의 어느 구간인지)를 주면 같은 구간을 다시 뽑지 않는다. 슬라이더를 옮기거나 확인 버튼을 눌러
    다시 찾을 때마다 전체를 새로 뽑으면 너무 느리기 때문이다. 결과는 절반 정밀도로 최근 것 CACHE개만 둔다.
    """
    CACHE = 256

    def __init__(self, weights: str | Path, device: str = "cpu"):
        self.device = device
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.net = TDSpeakerBeam()
        state = torch.load(weights, map_location="cpu")
        self.net.load_state_dict({k: v for k, v in state.items() if k.split(".")[0] in ("encoder", "extractor", "decoder")})
        self.net.eval().to(device)

    @staticmethod
    def _fingerprint(enroll: np.ndarray) -> tuple:
        return (len(enroll), round(float(enroll[::97].sum()), 4))

    def forget(self, prefix: str) -> int:
        """prefix로 시작하는 녹음에서 뽑은 결과를 지운다(온라인 서비스에서 세션을 지울 때). 지운 개수를 돌려준다."""
        gone = [k for k in self._cache if isinstance(k[0], tuple) and str(k[0][0]).startswith(prefix)]
        for k in gone:
            del self._cache[k]
        return len(gone)

    def __call__(self, mix: np.ndarray, enroll: np.ndarray, key: tuple | None = None) -> np.ndarray:
        if key is not None:
            key = (key, len(mix), self._fingerprint(enroll))
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key].astype(np.float32)
        out = self._run(mix, enroll)
        if key is not None:
            self._cache[key] = out.astype(np.float16)
            while len(self._cache) > self.CACHE:
                self._cache.popitem(last=False)
        return out

    def _run(self, mix: np.ndarray, enroll: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            m = torch.from_numpy(mix.astype(np.float32))[None].to(self.device)
            e = torch.from_numpy(enroll.astype(np.float32))[None].to(self.device)
            out = self.net(m, e)[0].cpu().numpy()
        peak = np.abs(out).max()
        return (out / peak * 0.9).astype(np.float32) if peak > 0 else out
