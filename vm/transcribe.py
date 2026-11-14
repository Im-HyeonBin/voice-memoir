"""하이라이트 구간 받아쓰기(선택). 처음 한 번 Whisper 모델을 내려받는다."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, size: str = "small"):
        self.model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(self, wav: np.ndarray) -> str:
        segments, _ = self.model.transcribe(wav, language="ko", beam_size=1,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
