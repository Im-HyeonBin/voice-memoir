# 가져다 쓴 코드·모델·데이터

저장소에는 코드만 들어 있다. 모델 가중치와 데이터는 넣지 않았고, 처음 실행할 때 원래 출처에서 받거나(`tools/fetch_data.py`) 직접 학습해 만든다.

## 코드

| 파일 | 원본 | 라이선스 | 바꾼 점 |
|---|---|---|---|
| `vm/tse.py` | ESPnet `espnet2/enh/` TD-SpeakerBeam 관련 파일 (Copyright 2023 Wangyou Zhang) | Apache-2.0 | 이 앱이 쓰는 설정(비인과, gLN, 곱셈 적응층)만 남김. 층 이름은 원본과 같아 원본 가중치를 그대로 읽음 |
| `vm/tse.py`의 TCN 부분 | kaituoxu/Conv-TasNet | MIT | ESPnet을 거쳐 옮김 |
| `vm/samresnet.py` | WeSpeaker `wespeaker/models/` SimAM-ResNet | Apache-2.0 | 모델 정의만 옮김(패키지 전체를 불러오면 지금 torchaudio와 맞지 않는 모듈이 따라옴) |

## 실행 중에 쓰는 라이브러리와 모델

| 이름 | 쓰는 곳 | 라이선스 |
|---|---|---|
| Silero VAD | 말소리 구간 찾기 | MIT |
| SpeechBrain, `speechbrain/spkrec-ecapa-voxceleb` | ECAPA 화자 지문(상업용 선택지의 바탕) | Apache-2.0 (학습 데이터 VoxCeleb 조건은 따로 확인 필요) |
| `gaunernst/wespeaker-voxblink2-samresnet34-ft` | 기본 화자 지문의 바탕 | 표기 없음, 학습 데이터 VoxBlink2가 CC BY-NC-SA 4.0 → 비상업으로 취급 |
| ESPnet `Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw` | 대상 화자 추출의 출발점 | CC BY 4.0 |
| faster-whisper, Whisper | 받아쓰기(선택) | MIT |
| Streamlit | 화면 | Apache-2.0 |
| 고운바탕, Pretendard | 화면 글꼴(CDN) | SIL OFL 1.1 |

## 직접 학습한 가중치(저장소에 없음)

| 가중치 | 바탕 | 학습 데이터 | 쓸 수 있는 범위 |
|---|---|---|---|
| `models/alt/voxblink2_samresnet34_ko` | VoxBlink2 SimAM-ResNet34 | Zeroth-Korean, RIRS_NOISES(시뮬레이션 울림, 점 소음 843개) | 바탕 모델 조건을 따라 비상업 |
| `models/alt/ecapa_ko` | SpeechBrain ECAPA | Zeroth-Korean, RIRS_NOISES(시뮬레이션 울림, 점 소음 843개) | 바탕·데이터가 상업 이용 가능(출처 표기) |
| `models/tse/td_speakerbeam_ko.pth` | ESPnet TD-SpeakerBeam | Zeroth-Korean, RIRS_NOISES(시뮬레이션 울림, 점 소음 843개) | CC BY 4.0 파생, 출처 표기 |

## 데이터

| 데이터 | 쓴 곳 | 라이선스 |
|---|---|---|
| Zeroth-Korean (OpenSLR 40) | 학습 세트는 추가 학습, 테스트 세트는 시험 세트·시험 | CC BY 4.0 |
| OpenSLR 28 RIRS_NOISES | 시뮬레이션 울림·점 소음은 학습, 실측 울림은 시험 | Apache-2.0 |
| DEMAND (Zenodo 1227121) | 시험에만 | CC BY 4.0 |
| Pansori-TEDxKR (OpenSLR 58) | 시험에만, 바꾸거나 다시 배포하지 않음 | CC BY-NC-ND 4.0 |
| AMI Meeting Corpus 시험 분할 세 모임(ES2004, IS1009, TS3003), 정답은 pyannote AMI-diarization-setup의 RTTM | 실제 녹음 검증에만 | CC BY 4.0 (RTTM은 MIT) |
| LibriSpeech 샘플, VOiCES 샘플 | 데모·어려운 세트의 다른 사람 목소리 | CC BY 4.0 |
| 찾을 사람 목소리 | 데모·어려운 세트 | 만든 사람 본인 녹음, 저장소에 넣지 않음 |
