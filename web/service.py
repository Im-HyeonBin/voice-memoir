"""온라인 서비스의 계산 부분(웹 프레임워크와 무관하게 시험할 수 있게 따로 둔다).

- 세션 하나 = 한 사람이 올린 녹음과 그 결과. 세션마다 전용 폴더를 두고, 다 쓰면 폴더째 지운다.
- 무거운 계산(녹음 살펴보기, 찾기)은 작업자 스레드 하나가 줄을 세워 하나씩 처리한다.
  서버 CPU를 여러 사람이 나눠 쓰지 않아 한 사람 작업이 빨리 끝나고, 서버가 감당할 양도 예측할 수 있다.
- 마지막으로 쓴 뒤 정해진 시간이 지나면 청소 스레드가 세션을 지운다. 서버를 다시 켜면 남은 세션 폴더도 모두 지운다.
- 올린 파일은 형식·크기·길이를 확인하고, ffmpeg는 녹음 형식과 로컬 파일만 열게 묶는다(vm/audio.py SAFE_INPUT_OPTS).
"""
from __future__ import annotations

import io
import os
import re
import secrets
import shutil
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from visual import envelope
from vm import audio
from vm.library import Library
from vm.speakers import (Target, apply_review, discover_speakers, search, target_from_clusters,
                         target_from_reference, with_review)
from vm.timeline import (all_selections, build_timeline, compile_highlight, select_highlights, source_rows,
                         sources_csv)

LEVELS = ("넉넉하게", "조금 넉넉하게", "보통", "조금 엄격하게", "엄격하게")  # engine.cal.levels와 짝
DATE_BASIS = {
    "metadata": "녹음 정보 기준",
    "filename": "파일 이름 기준",
    "file": "파일 시각 기준이라 정확하지 않을 수 있어요",
    "unknown": "날짜를 알 수 없어요",
}
BUSY_STATES = {"queued", "scanning", "searching"}


class UserError(Exception):
    """화면에 그대로 보여 줄 수 있는 오류(한국어 문장)."""


class Busy(UserError):
    """서버가 지금 더 받을 수 없을 때."""


class Gone(UserError):
    """세션이 없거나 이미 지워졌을 때."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 30
    max_bytes: int = 1024 ** 3                # 한 세션에 올리는 파일 크기 합(영상이 크다)
    max_file_bytes: int = 700 * 1024 ** 2
    max_audio_sec: float = 30 * 60            # 한 세션 녹음 길이 합. 서버 CPU로 15분 안쪽에 끝나게 정했다
    max_ref_bytes: int = 50 * 1024 ** 2
    max_ref_sec: float = 120.0
    chunk_bytes: int = 16 * 1024 ** 2         # 한 번에 받는 조각 크기 상한(Cloudflare 요청 한도 100MB보다 작게)
    ttl_sec: float = 60 * 60                  # 마지막으로 쓴 뒤 이만큼 지나면 지운다
    max_age_sec: float = 3 * 60 * 60          # 쓰고 있어도 이만큼 지나면 지운다
    max_sessions: int = 20
    max_queue: int = 6
    max_clip_sec: float = 120.0


@dataclass
class Upload:
    id: str
    name: str
    path: Path
    size: int
    received: int = 0
    done: bool = False
    seconds: float = 0.0
    modified: float | None = None  # 브라우저가 알려 준 파일 시각(밀리초)


@dataclass
class Session:
    id: str
    dir: Path
    media_dir: Path
    cache_dir: Path
    demo: bool = False
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    state: str = "empty"   # empty → scanning → ready → searching → done (기다리면 queued, 실패하면 error)
    job: str = ""
    message: str = ""
    progress: dict = field(default_factory=lambda: {"done": 0, "total": 0, "names": []})
    uploads: dict = field(default_factory=dict)
    records: list | None = None
    clusters: list | None = None
    target: Target | None = None
    result: object | None = None
    person: str = "찾는 사람"
    level: int = 2
    accepted: list = field(default_factory=list)   # 사람이 '맞아요'로 확인한 구간 (파일, 시작, 끝)
    rejected: list = field(default_factory=list)
    overlap_regions: set = field(default_factory=set)  # 위 둘 중 다른 목소리와 겹친 구간
    ref_wav: np.ndarray | None = None
    same: set = field(default_factory=set)
    not_same: set = field(default_factory=set)
    notice: str = ""
    started: float = 0.0    # 지금 작업을 시작한 시각
    estimate: float = 0.0   # 지금 작업에 걸릴 것으로 보는 시간(초)
    hint: str = ""          # 줄에 선 찾기가 참고 녹음으로 찾는 것인지(어림에 쓴다)
    deleted: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    cache: audio.AudioCache = field(default_factory=lambda: audio.AudioCache(max_files=6))
    views: dict = field(default_factory=dict)  # 결과 화면용 소리·파형(결과가 바뀌면 비운다)

    @property
    def busy(self) -> bool:
        return self.state in BUSY_STATES

    def reset_search(self) -> None:
        self.target, self.result, self.views = None, None, {}
        self.accepted, self.rejected, self.overlap_regions = [], [], set()
        self.ref_wav, self.same, self.not_same, self.notice = None, set(), set(), ""


def mmss(sec: float) -> str:
    return f"{int(sec // 60)}분 {int(sec % 60)}초" if sec >= 60 else f"{sec:.1f}초"


def day_text(d: date | None) -> str:
    return f"{d.year}년 {d.month}월 {d.day}일" if d else "날짜 모름"


def clean_name(name: str, limit: int = 120) -> str:
    """올린 파일 이름에서 경로와 제어 문자를 빼고 길이를 줄인다(날짜를 읽으려고 이름은 최대한 살린다)."""
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().lstrip(".")
    stem, ext = os.path.splitext(name)
    return (stem[: limit - len(ext)] + ext) if stem else ""


def clean_person(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f<>]", "", text or "").strip()
    return text[:20] or "찾는 사람"


def file_stem(text: str) -> str:
    """내려받는 파일 이름에 쓸 수 있는 글자만 남긴다."""
    return re.sub(r"[^\w가-힣]+", "_", text).strip("_")[:30] or "목소리"


class Service:
    def __init__(self, data_dir: str | Path, engine_factory: Callable[[], object], *,
                 limits: Limits = Limits(), demo_dir: str | Path | None = None,
                 demo_ref: str | Path | None = None, extractor_factory: Callable[[], object] | None = None,
                 devices: list[str] | None = None, start: bool = True):
        self.limits = limits
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        shutil.rmtree(self.sessions_dir, ignore_errors=True)  # 지난번에 남은 세션은 하나도 남기지 않는다
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.demo_dir = Path(demo_dir) if demo_dir and Path(demo_dir).is_dir() else None
        self.demo_ref = Path(demo_ref) if demo_ref and Path(demo_ref).exists() else None
        self.demo_cache = self.data_dir / "demo_cache"
        self.engine_factory, self.extractor_factory = engine_factory, extractor_factory
        self.devices = devices
        # 걸리는 시간 어림. 화면에 '남은 시간'으로 보여 준다. 서버(vm2 CPU 3개)에서 녹음 28분·말소리 18분으로
        # 잰 값(tools/measure_web.py): 살펴보기 200초, 고른 목소리로 찾기 144초, 참고 녹음으로 찾기 316초
        self.rates = {"scan": float(os.environ.get("VM_SCAN_RATE", 8.5)),          # 녹음 초 / 걸린 초
                      "pick": float(os.environ.get("VM_SEARCH_RATE", 7.5)),        # 말소리 초 / 걸린 초
                      "reference": float(os.environ.get("VM_REF_RATE", 3.4)),
                      "overhead": 10.0}
        self._demo_seconds: float | None = None
        self._engine = self._extractor = None
        self.ready = threading.Event()
        self.sessions: dict[str, Session] = {}
        self.queue: deque = deque()
        self.running: str | None = None
        self.cv = threading.Condition()
        self._stop = False
        if start:
            threading.Thread(target=self._work, name="vm-worker", daemon=True).start()
            threading.Thread(target=self._janitor, name="vm-janitor", daemon=True).start()

    # ── 모델 ───────────────────────────────────────────
    @property
    def engine(self):
        if self._engine is None:
            self._engine = self.engine_factory()
            self._extractor = self.extractor_factory() if self.extractor_factory else None
            self.ready.set()
        return self._engine

    @property
    def extractor(self):
        self.engine  # noqa: B018 - 모델을 함께 불러온다
        return self._extractor

    # ── 세션 ───────────────────────────────────────────
    def create(self) -> Session:
        with self.cv:
            alive = [s for s in self.sessions.values() if not s.deleted]
            if len(alive) >= self.limits.max_sessions:
                raise Busy("지금 쓰는 사람이 많아요. 잠시 뒤에 다시 와 주세요.")
            sid = secrets.token_urlsafe(24)
            root = self.sessions_dir / sid
            (root / "uploads").mkdir(parents=True)
            s = Session(sid, root, root / "uploads", root / "cache")
            self.sessions[sid] = s
            return s

    def get(self, sid: str) -> Session:
        s = self.sessions.get(sid)
        if s is None or s.deleted:
            raise Gone("녹음과 결과가 지워져 처음 화면으로 돌아왔어요. 오래 쓰지 않으면 자동으로 지워요.")
        s.touched = time.time()
        return s

    def delete(self, sid: str) -> None:
        with self.cv:
            s = self.sessions.get(sid)
            if s is None:
                return
            s.deleted = True
            self.queue = deque(j for j in self.queue if j[0] != sid)
            if self.running != sid:
                self._purge(s)

    def _purge(self, s: Session) -> None:
        with s.lock:
            s.records = s.clusters = s.target = s.result = s.ref_wav = None
            s.views, s.cache = {}, audio.AudioCache(max_files=1)
        shutil.rmtree(s.dir, ignore_errors=True)
        self.sessions.pop(s.id, None)

    def _janitor(self) -> None:
        while not self._stop:
            time.sleep(30)
            self.sweep()

    def sweep(self, now: float | None = None) -> int:
        """오래 쓰지 않은 세션을 지운다. 지운 개수를 돌려준다."""
        now = time.time() if now is None else now
        old = [s for s in list(self.sessions.values())
               if not s.deleted and not s.busy and
               (now - s.touched > self.limits.ttl_sec or now - s.created > self.limits.max_age_sec)]
        for s in old:
            self.delete(s.id)
        return len(old)

    # ── 올리기 ─────────────────────────────────────────
    def start_upload(self, s: Session, name: str, size: int, modified_ms: float | None = None) -> Upload:
        lim = self.limits
        if s.busy:
            raise UserError("녹음을 살펴보는 중에는 파일을 더 올릴 수 없어요.")
        if s.demo:
            raise UserError("예시 녹음을 보는 중이에요. 처음 화면에서 내 녹음을 올려 주세요.")
        name = clean_name(name)
        if not name or Path(name).suffix.lower() not in audio.MEDIA_EXTS:
            raise UserError(f"{name or '이 파일'}은 올릴 수 없는 형식이에요. 영상이나 녹음 파일을 올려 주세요.")
        if size <= 0 or size > lim.max_file_bytes:
            raise UserError(f"{name}은 너무 커요. 파일 하나는 {lim.max_file_bytes // 1024 ** 2}MB까지 올릴 수 있어요.")
        live = [u for u in s.uploads.values()]
        if len(live) >= lim.max_files:
            raise UserError(f"한 번에 파일 {lim.max_files}개까지 올릴 수 있어요.")
        if sum(u.size for u in live) + size > lim.max_bytes:
            raise UserError(f"파일 크기를 모두 합쳐 {lim.max_bytes // 1024 ** 2}MB까지 올릴 수 있어요.")
        uid = secrets.token_hex(6)
        folder = s.media_dir / uid  # 파일마다 폴더를 따로 둬 이름이 같아도 겹치지 않고, 원래 이름을 지킨다
        folder.mkdir(parents=True)
        u = Upload(uid, name, folder / name, size, modified=modified_ms)
        u.path.touch()
        self._keep_time(u)
        s.uploads[uid] = u
        return u

    def write_chunk(self, s: Session, uid: str, offset: int, data: bytes) -> int:
        u = s.uploads.get(uid)
        if u is None or u.done:
            raise UserError("올리던 파일을 찾지 못했어요. 다시 올려 주세요.")
        if offset < u.received and offset + len(data) <= u.received:
            return u.received  # 응답을 못 받아 같은 조각을 다시 보낸 경우(이미 받았다)
        if offset != u.received:
            raise UserError("파일 조각 순서가 맞지 않아요. 다시 올려 주세요.")
        if u.received + len(data) > u.size or len(data) > self.limits.chunk_bytes:
            raise UserError("파일이 처음에 알려 준 크기보다 커요.")
        with open(u.path, "ab") as f:
            f.write(data)
        u.received += len(data)
        self._keep_time(u)
        return u.received

    @staticmethod
    def _keep_time(u: Upload) -> None:
        """파일 시각을 원래 파일 시각으로 돌려 둔다(녹음 날짜를 다른 데서 못 찾을 때 쓴다)."""
        if u.modified and 0 < u.modified < 4102444800000:
            t = float(u.modified) / 1000
            os.utime(u.path, (t, t))

    def finish_upload(self, s: Session, uid: str) -> Upload:
        u = s.uploads.get(uid)
        if u is None:
            raise UserError("올리던 파일을 찾지 못했어요. 다시 올려 주세요.")
        if u.received != u.size:
            raise UserError(f"{u.name}을 끝까지 받지 못했어요. 다시 올려 주세요.")
        sec = audio.duration_of(audio.probe(u.path))
        if sec <= 0:
            self.remove_upload(s, uid)
            raise UserError(f"{u.name}에서 소리를 읽지 못했어요. 영상이나 녹음 파일이 맞는지 확인해 주세요.")
        used = sum(x.seconds for x in s.uploads.values() if x.done)
        if used + sec > self.limits.max_audio_sec:
            self.remove_upload(s, uid)
            raise UserError(f"녹음 길이를 모두 합쳐 {int(self.limits.max_audio_sec // 60)}분까지 살펴볼 수 있어요. "
                            f"{u.name}은 빼고 진행해 주세요.")
        u.seconds, u.done = sec, True
        return u

    def remove_upload(self, s: Session, uid: str) -> None:
        if s.busy:
            raise UserError("녹음을 살펴보는 중에는 파일을 뺄 수 없어요.")
        u = s.uploads.pop(uid, None)
        if u is not None:
            shutil.rmtree(u.path.parent, ignore_errors=True)

    def use_demo(self, s: Session) -> None:
        if self.demo_dir is None:
            raise UserError("예시 녹음이 준비되어 있지 않아요.")
        if s.busy:
            raise UserError("하던 일이 끝난 뒤에 다시 눌러 주세요.")
        for uid in list(s.uploads):
            self.remove_upload(s, uid)
        s.demo, s.media_dir, s.cache_dir = True, self.demo_dir, self.demo_cache
        s.records = s.clusters = None
        s.reset_search()

    # ── 작업 줄 ────────────────────────────────────────
    def submit(self, s: Session, kind: str, fn: Callable[[Session], None]) -> None:
        """작업을 줄에 세운다. 같은 세션의 같은 종류 작업이 아직 기다리고 있으면 새 것으로 바꾼다."""
        with self.cv:
            for i, (sid, k, _) in enumerate(self.queue):
                if sid == s.id and k == kind:
                    self.queue[i] = (s.id, kind, fn)
                    return
            if self.running == s.id and kind == "scan":
                raise UserError("지금 녹음을 살펴보고 있어요.")
            if len(self.queue) >= self.limits.max_queue:
                raise Busy("지금 기다리는 사람이 많아요. 잠시 뒤에 다시 눌러 주세요.")
            self.queue.append((s.id, kind, fn))
            s.state, s.job, s.message = "queued", kind, ""
            self.cv.notify()

    def position(self, sid: str) -> int | None:
        """앞에서 기다리는 작업 수(지금 처리 중인 것 포함). 줄에 없으면 None."""
        with self.cv:
            for i, (q, _, _) in enumerate(self.queue):
                if q == sid:
                    return i + (1 if self.running else 0)
        return None

    def _work(self) -> None:
        try:
            self.engine  # noqa: B018 - 첫 손님이 기다리지 않게 미리 불러 둔다
        except Exception:  # 모델을 못 불러와도 줄은 돌려, 각 작업이 오류 문구를 남기게 한다
            pass
        while not self._stop:
            with self.cv:
                while not self.queue and not self._stop:
                    self.cv.wait()
                if self._stop:
                    return
                sid, kind, fn = self.queue.popleft()
                s = self.sessions.get(sid)
                if s is None or s.deleted:
                    continue
                self.running = sid
            self.run_job(s, kind, fn)

    def demo_seconds(self) -> float:
        if self._demo_seconds is None:
            self._demo_seconds = sum(audio.duration_of(audio.probe(p)) for p in sorted(self.demo_dir.iterdir())
                                     if p.suffix.lower() in audio.MEDIA_EXTS) if self.demo_dir else 0.0
        return self._demo_seconds

    def estimate(self, s: Session, kind: str) -> float:
        """작업에 걸릴 시간을 녹음 길이로 어림한다(초)."""
        r = self.rates
        if kind == "scan":
            sec = self.demo_seconds() if s.demo else sum(u.seconds for u in s.uploads.values() if u.done)
            return r["overhead"] + sec / r["scan"]
        speech = sum(rec.speech_seconds for rec in s.records or [])
        ref = s.hint == "reference" or (not s.hint and s.target is not None and s.target.source == "참고 녹음")
        return r["overhead"] + speech / r["reference" if ref else "pick"]

    def run_job(self, s: Session, kind: str, fn: Callable[[Session], None]) -> None:
        s.state = "scanning" if kind == "scan" else "searching"
        s.started = time.time()
        try:
            s.estimate = self.estimate(s, kind)
        except Exception:  # 어림을 못 해도 작업은 한다
            s.estimate = 0.0
        s.hint = ""
        try:
            fn(s)
            s.state = "ready" if kind == "scan" else "done"
        except UserError as exc:
            s.state, s.message = ("ready" if s.records is not None else "error"), str(exc)
        except Exception:  # 녹음 내용·경로가 남지 않게 자세한 내용은 기록하지 않는다
            s.state = "error" if kind == "scan" else ("ready" if s.records is not None else "error")
            s.message = "처리하다가 문제가 생겼어요. 다시 시도해 주세요."
        finally:
            with self.cv:
                self.running = None
                if s.deleted:
                    self._purge(s)
            s.touched = time.time()

    # ── 계산 ───────────────────────────────────────────
    def scan(self, s: Session) -> None:
        if not s.demo:
            if any(not u.done for u in s.uploads.values()):
                raise UserError("아직 다 올라가지 않은 파일이 있어요.")
            if not s.uploads:
                raise UserError("먼저 녹음 파일을 올려 주세요.")

        def job(s: Session) -> None:
            engine = self.engine
            lib = Library(engine, s.cache_dir)
            with s.lock:
                s.records = s.clusters = None
                s.reset_search()
            s.progress = {"done": 0, "total": 0, "names": []}

            def tick(i: int, n: int, name: str) -> None:
                names = s.progress["names"] + ([name] if name else [])
                s.progress = {"done": i, "total": n, "names": names}

            records = lib.scan(s.media_dir, progress=tick, devices=self.devices)
            cache = audio.AudioCache(max_files=6)
            clusters = discover_speakers(records, engine, cache)
            with s.lock:
                s.records, s.clusters, s.cache = records, clusters, cache
                s.reset_search()

        self.submit(s, "scan", job)

    def _need_voices(self, s: Session) -> None:
        if s.records is None or s.clusters is None:
            raise UserError("먼저 녹음을 살펴봐야 해요.")

    def pick(self, s: Session, indices: list[int], follow: bool | None = None) -> None:
        """고른 목소리로 찾는다. follow=None이면 모델 기준값대로 같은 사람 목소리를 따라가고, False면 고른 것만 쓴다."""
        self._need_voices(s)
        by_index = {c.index: c for c in s.clusters}
        picked = [by_index[i] for i in dict.fromkeys(indices) if i in by_index]
        if not picked:
            raise UserError("찾을 목소리를 골라 주세요.")

        def job(s: Session) -> None:
            t = target_from_clusters(picked, s.clusters, self.engine, s.cache, follow=follow)
            with s.lock:
                s.reset_search()
                s.target = t
            self._search(s)

        s.hint = "pick"
        self.submit(s, "search", job)

    def reference(self, s: Session, path: Path) -> None:
        self._need_voices(s)
        info = audio.probe(path)
        sec = audio.duration_of(info)
        if sec <= 0:
            raise UserError("참고 녹음에서 소리를 읽지 못했어요. 녹음 파일이 맞는지 확인해 주세요.")
        if sec > self.limits.max_ref_sec:
            raise UserError(f"참고 녹음은 {int(self.limits.max_ref_sec)}초까지 써요. 짧게 잘라서 올려 주세요.")

        def job(s: Session) -> None:
            try:
                wav = audio.decode(path)
                t = target_from_reference(self.engine, wav, s.clusters)
            except ValueError as exc:  # 말소리가 너무 짧을 때 등
                raise UserError(str(exc)) from exc
            with s.lock:
                s.reset_search()
                s.target, s.ref_wav = t, wav
                if t.info.get("mixed"):
                    s.notice = "참고 녹음에 다른 목소리도 섞여 있어서, 가장 많이 말한 사람을 기준으로 삼았어요."
            self._search(s)

        s.hint = "reference"
        self.submit(s, "search", job)

    def demo_reference(self, s: Session) -> None:
        if self.demo_ref is None or not s.demo:
            raise UserError("예시 참고 녹음은 예시 녹음에서만 쓸 수 있어요.")
        self.reference(s, self.demo_ref)

    def answer_same(self, s: Session, index: int, yes: bool) -> None:
        """참고 녹음과 닮았지만 확실하지 않은 목소리를 사람이 정한다. 같은 사람이면 기준을 다시 만들어 찾는다."""
        if s.target is None or s.ref_wav is None:
            raise UserError("참고 녹음으로 찾은 뒤에 쓸 수 있어요.")
        with s.lock:
            (s.same if yes else s.not_same).add(int(index))
        if not yes:
            return

        def job(s: Session) -> None:
            t = target_from_reference(self.engine, s.ref_wav, s.clusters, same=s.same)
            with s.lock:
                s.target, s.result, s.views = t, None, {}
                s.accepted, s.rejected, s.overlap_regions = [], [], set()
            self._search(s)

        self.submit(s, "search", job)

    def review(self, s: Session, file: int, start: float, end: float, answer: str, overlap: bool = False) -> None:
        """'맞아요'·'아니에요'·되돌리기. 판단을 기준에 더해 다시 찾는다."""
        if s.target is None or s.records is None:
            raise UserError("먼저 목소리를 찾아 주세요.")
        if not 0 <= file < len(s.records):
            raise UserError("구간을 찾지 못했어요.")
        region = (s.records[file].path, round(float(start), 2), round(float(end), 2))
        with s.lock:
            if answer == "undo":
                for lst in (s.accepted, s.rejected):
                    if region in lst:
                        lst.remove(region)
                s.overlap_regions.discard(region)
            elif answer in ("yes", "no"):
                if overlap:
                    s.overlap_regions.add(region)
                keep, drop = (s.accepted, s.rejected) if answer == "yes" else (s.rejected, s.accepted)
                if region in drop:
                    drop.remove(region)
                if region not in keep:
                    keep.append(region)
            else:
                raise UserError("알 수 없는 답이에요.")
        self.submit(s, "search", self._search)

    def settings(self, s: Session, person: str | None = None, level: int | None = None) -> None:
        if person is not None:
            s.person = clean_person(person)
        if level is not None:
            if not 0 <= int(level) < len(LEVELS):
                raise UserError("알 수 없는 단계예요.")
            if int(level) != s.level:
                s.level = int(level)
                if s.target is not None:
                    self.submit(s, "search", self._search)

    def _search(self, s: Session) -> None:
        engine = self.engine
        threshold = engine.cal.levels[s.level]
        acc = [r for r in s.accepted if r not in s.overlap_regions]
        rej = [r for r in s.rejected if r not in s.overlap_regions]
        target = with_review(s.target, engine, s.cache, s.records, acc, rej) if acc or rej else s.target
        found = search(s.records, target, engine, s.cache, threshold, self.extractor)
        found = apply_review(found, s.records, list(s.accepted), list(s.rejected))
        with s.lock:
            s.result, s.views = found, {}

    # ── 화면에 줄 내용 ─────────────────────────────────
    def status(self, s: Session) -> dict:
        recs = s.records or []
        out = {
            "state": s.state, "job": s.job, "message": s.message, "notice": s.notice,
            "queue": self.position(s.id) if s.state == "queued" else None,
            "progress": s.progress, "demo": s.demo, "person": s.person, "level": s.level, "levels": LEVELS,
            "has_demo": self.demo_dir is not None, "has_demo_ref": self.demo_ref is not None and s.demo,
            "voices_ready": s.clusters is not None, "has_target": s.target is not None,
            "has_result": s.result is not None,
            "uploads": [{"id": u.id, "name": u.name, "size": u.size, "received": u.received, "done": u.done,
                         "seconds": round(u.seconds, 1)} for u in s.uploads.values()],
            "expires_in": max(0, int(self.limits.ttl_sec - (time.time() - s.touched))),
            "elapsed": round(time.time() - s.started, 1) if s.state in ("scanning", "searching") else 0,
            "estimate": round(s.estimate, 1),
        }
        if recs:
            out["files"] = [{"id": i, "name": Path(r.path).name, "date": day_text(r.date) if r.date else None,
                             "basis": DATE_BASIS[r.date_source], "seconds": round(r.duration, 1),
                             "speech": round(r.speech_seconds, 1), "phone": bool(r.narrowband),
                             "error": bool(r.error)} for i, r in enumerate(recs)]
            out["summary"] = {"files": len(recs), "seconds": round(sum(r.duration for r in recs), 1),
                              "speech": round(sum(r.speech_seconds for r in recs), 1),
                              "phone": sum(bool(r.narrowband) for r in recs),
                              "errors": sum(bool(r.error) for r in recs)}
        return out

    def _sample(self, s: Session, c) -> np.ndarray:
        key = ("sample", c.index)
        if key not in s.views:
            p, a, b = c.sample
            s.views[key] = s.cache.slice(p, a, b)
        return s.views[key]

    def _voice(self, s: Session, c, bars: int = 60) -> dict:
        wav = self._sample(s, c)
        return {"index": c.index, "files": len(c.files), "seconds": round(c.seconds, 1),
                "phone": bool(c.phone_linked), "env": np.round(envelope(wav, bars), 2).tolist()}

    def voices(self, s: Session) -> dict:
        self._need_voices(s)
        with s.lock:
            picked = s.target.info.get("clusters", []) if s.target is not None and s.target.source == "자동 발견" else []
            return {"voices": [self._voice(s, c) for c in s.clusters], "picked": list(picked)}

    def voice_audio(self, s: Session, index: int) -> bytes:
        self._need_voices(s)
        with s.lock:
            c = next((c for c in s.clusters if c.index == index), None)
            if c is None:
                raise UserError("목소리를 찾지 못했어요.")
            return audio.wav_bytes(self._sample(s, c))

    def _groups(self, s: Session, view: str) -> list[tuple[str, str, list, bool]]:
        clips = s.result.clips
        if view == "date":
            by_date: dict = {}
            for c in clips:
                by_date.setdefault(c.date, []).append(c)
            return [(d.isoformat() if d else "unknown", day_text(d), by_date[d], False)
                    for d in sorted(by_date, key=lambda d: (d is None, d or date.min))]
        return [(str(y.year) if y.year else "unknown", str(y.year) if y.year else "날짜 모름", y.clips, True)
                for y in build_timeline(clips)]

    def _group_audio(self, s: Session, view: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        cache_key = ("group", view, key)
        if cache_key not in s.views:
            group = next((g for g in self._groups(s, view) if g[0] == key), None)
            if group is None:
                raise UserError("이 묶음을 찾지 못했어요.")
            _, _, gclips, highlight = group
            full = compile_highlight(all_selections(gclips), s.cache)
            main = compile_highlight(select_highlights(gclips), s.cache) if highlight else full
            s.views[cache_key] = (main, full)
        return s.views[cache_key]

    def _file_env(self, s: Session, path: str) -> list[float]:
        key = ("env", path)
        if key not in s.views:
            s.views[key] = np.round(envelope(s.cache.get(path), 120), 2).tolist()
        return s.views[key]

    def result(self, s: Session, view: str = "year") -> dict:
        if s.result is None:
            raise UserError("아직 찾은 결과가 없어요.")
        view = "date" if view == "date" else "year"
        with s.lock:
            recs = s.records
            fid = {r.path: i for i, r in enumerate(recs)}
            res = s.result
            groups = []
            for key, title, gclips, highlight in self._groups(s, view):
                main, full = self._group_audio(s, view, key)
                files = []
                for path in sorted({c.path for c in gclips}, key=lambda p: fid[p]):
                    r = recs[fid[path]]
                    files.append({
                        "id": fid[path], "name": Path(path).name,
                        "date": f"{day_text(r.date)} · {DATE_BASIS[r.date_source]}",
                        "phone": bool(r.narrowband), "env": self._file_env(s, path),
                        "spans": [[round(c.start / r.duration, 4), round(c.end / r.duration, 4)]
                                  for c in gclips if c.path == path],
                    })
                groups.append({
                    "key": key, "title": title, "highlight": highlight,
                    "seconds": round(sum(c.duration for c in gclips), 1), "files": files,
                    "main": {"seconds": round(len(main) / audio.SR, 1), "env": np.round(envelope(main, 90), 2).tolist()},
                    "full_seconds": round(len(full) / audio.SR, 1),
                    "clips": [{"file": fid[c.path], "name": Path(c.path).name, "start": round(c.start, 2),
                               "end": round(c.end, 2), "confirmed": c.confirmed}
                              for c in sorted(gclips, key=lambda c: (fid[c.path], c.start))],
                })
            maybe = []
            for m in res.maybe[:6]:
                maybe.append({"file": fid[m.path], "name": Path(m.path).name, "date": day_text(m.date),
                              "start": round(m.start, 2), "end": round(m.end, 2), "overlap": m.overlap})
            t = s.target
            by_index = {c.index: c for c in s.clusters}
            unsure = []
            if t is not None and t.source == "참고 녹음":
                unsure = [self._voice(s, by_index[i], 40) for i in t.info.get("unsure", [])
                          if i in by_index and i not in s.same | s.not_same][:4]
            followed = t.info.get("followed", []) if t is not None and t.source == "자동 발견" else []
            reviewed = [{"file": fid[p], "name": Path(p).name, "start": a, "end": b, "answer": ans}
                        for ans, lst in (("yes", s.accepted), ("no", s.rejected)) for p, a, b in lst if p in fid]
            return {
                "view": view, "person": s.person, "level": s.level, "levels": LEVELS,
                "source": t.source if t is not None else "",
                "total": round(sum(c.duration for c in res.clips), 1),
                "years": len(build_timeline(res.clips)), "n_files": len({c.path for c in res.clips}),
                "groups": groups, "maybe": maybe, "maybe_more": max(0, len(res.maybe) - 6),
                "unsure": unsure, "followed": followed, "notice": s.notice, "reviewed": reviewed,
                "particle": "과" if _has_final(s.person) else "와",
            }

    def group_audio(self, s: Session, view: str, key: str, full: bool = False) -> bytes:
        if s.result is None:
            raise UserError("아직 찾은 결과가 없어요.")
        with s.lock:
            main, whole = self._group_audio(s, "date" if view == "date" else "year", key)
            return audio.wav_bytes(whole if full else main)

    def clip_audio(self, s: Session, file: int, start: float, end: float) -> bytes:
        if s.records is None or not 0 <= file < len(s.records):
            raise UserError("구간을 찾지 못했어요.")
        r = s.records[file]
        start, end = float(start), float(end)
        if not (0 <= start < end <= r.duration + 0.05) or end - start > self.limits.max_clip_sec:
            raise UserError("들을 수 없는 구간이에요.")
        with s.lock:
            return audio.wav_bytes(s.cache.slice(r.path, start, end))

    def zip_bytes(self, s: Session) -> tuple[str, bytes]:
        if s.result is None or not s.result.clips:
            raise UserError("받을 결과가 없어요.")
        person = file_stem(s.person)
        buf, rows = io.BytesIO(), []
        with s.lock, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for y in build_timeline(s.result.clips):
                label = y.year or "날짜모름"
                zf.writestr(f"{person}_{label}_올해의목소리.wav",
                            audio.wav_bytes(compile_highlight(select_highlights(y.clips), s.cache)))
                sels = all_selections(y.clips)
                year_rows = source_rows(y, sels, None)
                rows += year_rows
                for sel, row in zip(sels, year_rows):
                    name = f"{row['녹음 날짜'] or '날짜모름'}_{file_stem(Path(sel.clip.path).stem)}_{sel.start:.1f}-{sel.end:.1f}초.wav"
                    zf.writestr(f"구간별/{name}", audio.wav_bytes(s.cache.slice(sel.clip.path, sel.start, sel.end)))
            zf.writestr(f"{person}_구간목록.csv", sources_csv(rows))
        return f"{person}_목소리_모음.zip", buf.getvalue()

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()


def _has_final(word: str) -> bool:
    ch = word[-1] if word else ""
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0
