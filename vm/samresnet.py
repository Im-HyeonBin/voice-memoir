# SimAM-ResNet34 + ASP 화자 지문 모델 정의.
# WeSpeaker(https://github.com/wenet-e2e/wespeaker, commit 9fecd6c)의
# wespeaker/models/samresnet.py와 pooling_layers.py(ASP)에서 이 앱이 쓰는 부분만 옮겼다.
# 학습된 가중치를 그대로 읽을 수 있게 층 이름과 구조는 원본과 같다.
#
# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2024 XiaoyiQin, Yuke Lin (linyuke0609@gmail.com)
#               2024 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import torch
import torch.nn as nn


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.simam(out)
        out += self.downsample(x)
        return self.relu(out)

    def simam(self, x: torch.Tensor, lambda_p: float = 1e-4) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        return x * self.sigmoid(d / (4 * (v + lambda_p)) + 0.5)


class ResNet(nn.Module):
    def __init__(self, in_planes: int, num_blocks: list[int]):
        super().__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(in_planes, num_blocks[0], 1)
        self.layer2 = self._make_layer(in_planes * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(in_planes * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(in_planes * 8, num_blocks[3], 2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(SimAMBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * SimAMBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ASP(nn.Module):
    """Attentive Statistics Pooling."""

    def __init__(self, in_planes: int, acoustic_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = in_planes * 8 * int(acoustic_dim / 8)
        self.out_dim = self.feature_dim * 2
        self.attention = nn.Sequential(
            nn.Conv1d(self.feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, self.feature_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1, x.size(-1))  # [B, C, F, T] → [B, C·F, T]
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sg], dim=1)


class SimAM_ResNet34_ASP(nn.Module):
    def __init__(self, in_planes: int = 64, embed_dim: int = 256, acoustic_dim: int = 80):
        super().__init__()
        self.front = ResNet(in_planes, [3, 4, 6, 3])
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (배치, 프레임, 80) fbank → (배치, embed_dim)."""
        x = self.front(x.permute(0, 2, 1).unsqueeze(1))
        return self.bottleneck(self.pooling(x))
