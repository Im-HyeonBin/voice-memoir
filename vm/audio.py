"""오디오 입출력과 녹음 날짜 추정.

어떤 포맷이든(영상 포함) 16kHz 모노 float32로 읽고, 메타데이터·파일명·파일 시각
순서로 녹음 날짜를 추정한다.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from collections import OrderedDict

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

SR = 16000
MEDIA_EXTS = {
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".amr", ".3gp",
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
}
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
# 남이 올린 파일을 여는 온라인 서비스에서만 채운다(web/service.py). 비워 두면 예전과 똑같이 동작한다.
# 입력 형식과 프로토콜을 녹음 형식과 로컬 파일로만 묶어, 재생목록 같은 파일로 다른 파일을 읽게 하는 일을 막는다.
INPUT_OPTS: list[str] = []
SAFE_INPUT_OPTS = ["-protocol_whitelist", "file",
                   "-format_whitelist", "mov,mp3,wav,ogg,flac,matroska,aac,amr,amrnb,amrwb,avi"]
TIMEOUT: float | None = None  # ffmpeg·ffprobe 한 번에 줄 최대 시간(초)

DATE_SOURCE_LABEL = {
    "metadata": "녹음 정보",
    "filename": "파일 이름",
    "file": "파일 시각(부정확할 수 있음)",
    "unknown": "알 수 없음",
}


def decode(path: str | Path, sr: int = SR) -> np.ndarray:
    """파일을 디코딩해 float32 모노 배열로 돌려준다. 영상이면 음성 트랙만 읽는다."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", *INPUT_OPTS, "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=TIMEOUT).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def probe(path: str | Path) -> dict:
    cmd = [FFPROBE, "-v", "error", *INPUT_OPTS, "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {}
    if res.returncode != 0:
        return {}
    return json.loads(res.stdout or "{}")


def duration_of(info: dict) -> float:
    try:
        return float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class RecordedDate:
    value: date | None
    source: str  # metadata | filename | file | unknown

    @property
    def label(self) -> str:
        return DATE_SOURCE_LABEL[self.source]


_META_KEYS = ("com.apple.quicktime.creationdate", "creation_time", "date")
_FILENAME_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_. ]?(0[1-9]|1[0-2])[-_. ]?(0[1-9]|[12]\d|3[01])(?!\d)"
)


def _plausible(d: date) -> bool:
    return 1990 <= d.year <= datetime.now().year + 1


def _parse_meta_date(text: str) -> date | None:
    text = text.strip()
    for candidate in (text.replace("Z", "+00:00"), text[:19], text[:10]):
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            continue
    return None


def recorded_date(path: str | Path, info: dict) -> RecordedDate:
    """녹음 날짜 추정: 메타데이터 > 파일 이름 > 파일 시각."""
    tag_sets = [info.get("format", {}).get("tags", {})]
    tag_sets += [s.get("tags", {}) for s in info.get("streams", [])]
    for tags in tag_sets:
        lowered = {k.lower(): v for k, v in tags.items()}
        for key in _META_KEYS:
            if key in lowered:
                d = _parse_meta_date(str(lowered[key]))
                if d and _plausible(d):
                    return RecordedDate(d, "metadata")

    m = _FILENAME_DATE.search(Path(path).stem)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if _plausible(d):
                return RecordedDate(d, "filename")
        except ValueError:
            pass

    # 복사·전송을 거치면 시각이 바뀌므로 알려진 가장 이른 시각을 쓴다.
    st = os.stat(path)
    stamps = [st.st_mtime] + ([st.st_birthtime] if hasattr(st, "st_birthtime") else [])
    d = datetime.fromtimestamp(min(stamps)).date()
    if _plausible(d):
        return RecordedDate(d, "file")
    return RecordedDate(None, "unknown")


def telephone_band(wav: np.ndarray) -> np.ndarray:
    """전화 음질(300~3400Hz, 8kHz)로 바꾼다. 통화 녹음과 비교할 기준 지문을 만들 때 쓴다."""
    sos = butter(6, [300, 3400], btype="band", fs=SR, output="sos")
    y = sosfiltfilt(sos, wav)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(wav)].astype(np.float32)


def high_band_ratio(wav: np.ndarray) -> float:
    """4kHz 위 에너지 / 말소리 대역 에너지. 전화 음질이면 거의 0이다."""
    if len(wav) < SR:
        return 1.0
    f, p = welch(wav, fs=SR, nperseg=1024)
    band = p[(f >= 300) & (f <= 3400)].sum()
    high = p[(f >= 4000) & (f <= 7500)].sum()
    return float(high / band) if band > 0 else 1.0


NARROWBAND_RATIO = 0.002  # 실측으로 정함


def is_narrowband(wav: np.ndarray) -> bool:
    return high_band_ratio(wav) < NARROWBAND_RATIO


class AudioCache:
    """같은 파일을 여러 번 디코딩하지 않도록 최근 파일 몇 개를 메모리에 둔다."""

    def __init__(self, max_files: int = 8):
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_files = max_files

    def get(self, path: str) -> np.ndarray:
        if path in self._items:
            self._items.move_to_end(path)
            return self._items[path]
        wav = decode(path)
        self._items[path] = wav
        if len(self._items) > self.max_files:
            self._items.popitem(last=False)
        return wav

    def slice(self, path: str, start: float, end: float) -> np.ndarray:
        return self.get(path)[int(start * SR):int(end * SR)]


def wav_bytes(wav: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_wav(path: str | Path, wav: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), wav, sr, subtype="PCM_16")


def encode_m4a(wav: np.ndarray, out_path: str | Path, sr: int = SR,
               metadata: dict[str, str] | None = None) -> None:
    """AAC(m4a)로 저장한다. 공유용으로 WAV보다 훨씬 작다."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        write_wav(src, wav, sr)
        cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-c:a", "aac", "-b:a", "96k"]
        for k, v in (metadata or {}).items():
            cmd += ["-metadata", f"{k}={v}"]
        cmd.append(str(out_path))
        subprocess.run(cmd, check=True)
