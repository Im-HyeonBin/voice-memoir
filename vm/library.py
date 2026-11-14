"""녹음 라이브러리 스캔.

폴더 안 파일마다 날짜·말소리 창·화자 지문을 계산하고 디스크에 캐시한다.
두 번째 스캔부터는 바뀐 파일만 다시 계산한다. 캐시는 이 컴퓨터에만 남는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio
from .engine import HOP, WIN, VoiceEngine

CACHE_VERSION = f"v3-win{WIN}-hop{HOP}"


@dataclass
class FileRecord:
    path: str
    duration: float
    date: date | None
    date_source: str
    windows: np.ndarray      # (N, 2) 초 단위 [시작, 끝]
    embeddings: np.ndarray   # (N, 지문 길이)
    narrowband: bool = False  # 전화 음질 녹음 여부
    error: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def date_label(self) -> str:
        return audio.DATE_SOURCE_LABEL[self.date_source]

    @property
    def speech_seconds(self) -> float:
        if len(self.windows) == 0:
            return 0.0
        return float(len(self.windows) * HOP + (WIN - HOP))


_WORKER: dict = {}


def _init_worker(models_dir: str, cache_dir: str, devices: list[str], counter, model: str) -> None:
    import torch
    torch.set_num_threads(1)
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    _WORKER["lib"] = Library(VoiceEngine(models_dir, devices[i % len(devices)], model), cache_dir)


def _analyze_in_worker(path: str) -> str:
    _WORKER["lib"].analyze(path)
    return path


@contextmanager
def _workers_skip_main():
    """작업자 프로세스를 띄우는 동안 __main__을 빈 모듈로 바꿔 둔다.

    spawn 방식 작업자는 시작할 때 __main__ 파일을 다시 실행한다. Streamlit은 화면 파일(app.py)을 __main__으로
    돌리기 때문에, 그대로 두면 작업자마다 화면 코드가 처음부터 돌다 죽고 스캔이 실패한다. 작업자가 쓰는 함수는
    이 모듈에 있어서 __main__이 없어도 된다.
    """
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if main is not None:
            sys.modules["__main__"] = main


def default_devices(model: str) -> list[str]:
    """작업자마다 쓸 장치.

    83분 분량 실측(실시간 대비)
    - ECAPA(bench/speed_round1.txt): CPU 1개 25~37배, CPU 작업자 4개 69배, 맥 GPU 작업자 2개 79~82배.
      GPU와 CPU를 섞거나 작업자를 더 늘리면 오히려 느려졌다.
    - SimAM-ResNet34(bench/speed_round2.txt): 맥 GPU 작업자 1개 33배, 2개는 서로 부딪혀 13배.
    """
    import torch
    if torch.cuda.is_available():
        return ["cuda"]
    if torch.backends.mps.is_available():
        return ["mps", "mps"] if model.startswith("ecapa") else ["mps"]
    return ["cpu"] * max(1, min(4, (os.cpu_count() or 2) // 2))


def find_media(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in audio.MEDIA_EXTS
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]
    return sorted(files)


class Library:
    def __init__(self, engine: VoiceEngine, cache_dir: str | Path):
        self.engine = engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_cached(self, path: Path) -> bool:
        key = self._key(path)
        meta_f, arr_f = self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.npz"
        try:  # 쓰다가 끊겨 빈 파일이 남았으면 캐시가 없는 것으로 본다
            return meta_f.stat().st_size > 0 and arr_f.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _read_cache(path: Path, meta_f: Path, arr_f: Path) -> FileRecord | None:
        """캐시를 읽는다. 읽다가 오류가 나면(쓰다 끊긴 파일 등) None을 돌려줘 다시 계산하게 한다."""
        try:
            meta = json.loads(meta_f.read_text())
            with np.load(arr_f) as arrs:
                windows, emb = arrs["windows"], arrs["embeddings"]
            return FileRecord(
                path=str(path), duration=meta["duration"],
                date=date.fromisoformat(meta["date"]) if meta["date"] else None,
                date_source=meta["date_source"],
                windows=windows, embeddings=emb,
                narrowband=meta.get("narrowband", False),
            )
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def _key(self, path: Path) -> str:
        st = path.stat()
        model = "" if self.engine.model == "ecapa" else f"|{self.engine.model}"  # 기존 캐시는 그대로 쓴다
        raw = f"{CACHE_VERSION}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}{model}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def analyze(self, path: str | Path) -> FileRecord:
        path = Path(path)
        key = self._key(path)
        meta_f = self.cache_dir / f"{key}.json"
        arr_f = self.cache_dir / f"{key}.npz"
        if meta_f.exists() and arr_f.exists():
            rec = self._read_cache(path, meta_f, arr_f)
            if rec is not None:
                return rec
            meta_f.unlink(missing_ok=True)  # 깨진 캐시는 지우고 다시 계산한다
            arr_f.unlink(missing_ok=True)

        info = audio.probe(path)
        rec_date = audio.recorded_date(path, info)
        try:
            wav = audio.decode(path)
        except Exception as exc:  # 손상·미지원 파일은 건너뛰고 표시만 한다
            return FileRecord(str(path), 0.0, rec_date.value, rec_date.source,
                              np.zeros((0, 2), np.float32), np.zeros((0, self.engine.dim), np.float32),
                              error=str(exc)[:200])
        windows, emb = self.engine.analyze_wave(wav)
        rec = FileRecord(
            path=str(path), duration=len(wav) / audio.SR,
            date=rec_date.value, date_source=rec_date.source,
            windows=windows, embeddings=emb,
            narrowband=audio.is_narrowband(wav),
        )
        # 임시 파일에 쓴 뒤 바꿔 넣는다. 중간에 끊겨도 반쯤 쓴 파일이 캐시로 남지 않는다.
        # json을 마지막에 바꿔 넣어, json이 있으면 배열도 온전히 있는 것으로 본다.
        tmp_arr, tmp_meta = arr_f.with_name(arr_f.name + ".tmp"), meta_f.with_name(meta_f.name + ".tmp")
        with open(tmp_arr, "wb") as f:
            np.savez(f, windows=windows, embeddings=emb)
        tmp_meta.write_text(json.dumps({
            "duration": rec.duration,
            "date": rec.date.isoformat() if rec.date else None,
            "date_source": rec.date_source,
            "narrowband": rec.narrowband,
        }))
        os.replace(tmp_arr, arr_f)
        os.replace(tmp_meta, meta_f)
        return rec

    def scan(self, folder: str | Path, progress: Callable[[int, int, str], None] | None = None,
             devices: list[str] | None = None) -> list[FileRecord]:
        """폴더를 살펴본다. 처음 보는 파일이 여럿이면 여러 프로세스에 나눠 맡긴다."""
        files = find_media(folder)
        todo = [f for f in files if not self.is_cached(f)]
        devices = default_devices(self.engine.model) if devices is None else devices
        done = len(files) - len(todo)
        if progress:
            progress(done, len(files), "")
        if len(devices) > 1 and len(todo) >= 3:
            # 작업자마다 수학 라이브러리 스레드를 하나로 묶어 서로 코어를 뺏지 않게 한다.
            for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[var] = "1"
            ctx = get_context("spawn")
            try:
                with ProcessPoolExecutor(min(len(devices), len(todo)), mp_context=ctx,
                                         initializer=_init_worker,
                                         initargs=(str(self.engine.models_dir), str(self.cache_dir), devices,
                                                   ctx.Value("i", 0), self.engine.model)) as ex:
                    with _workers_skip_main():  # 작업자는 submit할 때 뜬다
                        futures = [ex.submit(_analyze_in_worker, str(f)) for f in todo]
                    for fut in as_completed(futures):
                        name = Path(fut.result()).name
                        done += 1
                        if progress:
                            progress(done, len(files), name)
            except BrokenProcessPool:
                pass  # 작업자가 죽으면 남은 파일은 아래에서 이 프로세스로 이어서 계산한다
            todo = [f for f in todo if not self.is_cached(f)]
        for f in todo:
            self.analyze(f)
            done += 1
            if progress:
                progress(done, len(files), f.name)
        return [self.analyze(f) for f in files]  # 모두 캐시에 있으므로 바로 읽힌다
