"""목소리 발굴 온라인 서비스.

실행(개발): .venv/bin/uvicorn web.server:app --port 8600
설정(환경 변수)
- VM_DATA: 세션 폴더와 예시 캐시를 둘 곳(기본 .webdata)
- VM_DEVICE: 모델을 돌릴 장치(cpu, cuda, mps, auto. 기본 auto)
- VM_SCAN_WORKERS: 녹음 살펴보기 작업자 수(기본: 장치에 맞춰 자동)
- VM_THREADS: torch가 쓸 CPU 스레드 수
- VM_OVERLAP: 1이면 겹쳐 말한 구간도 찾는다(기본 1, 가중치가 없으면 꺼짐)
- VM_DEMO_DIR, VM_DEMO_REF: 예시 녹음 폴더와 예시 참고 녹음
- VM_TRUST_PROXY: 1이면 Cloudflare가 붙여 주는 CF-Connecting-IP로 사람을 구분한다
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from vm import audio  # noqa: E402
from web.service import Busy, Gone, Service, UserError  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; "
       "media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def _engine_factory():
    from vm.engine import VoiceEngine
    threads = os.environ.get("VM_THREADS")
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    return VoiceEngine(ROOT / "models", device=os.environ.get("VM_DEVICE", "auto"))


def _extractor_factory():
    if os.environ.get("VM_OVERLAP", "1") != "1":
        return None
    weights = ROOT / "models" / "tse" / "td_speakerbeam_ko.pth"
    if not weights.exists():
        return None
    import torch
    from vm.tse import Extractor
    dev = os.environ.get("VM_DEVICE", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return Extractor(weights, dev)


def _devices() -> list[str] | None:
    n = os.environ.get("VM_SCAN_WORKERS")
    if not n:
        return None
    dev = os.environ.get("VM_DEVICE", "cpu")
    return [dev if dev != "auto" else "cpu"] * max(1, int(n))


def build_service(**kw) -> Service:
    audio.INPUT_OPTS = audio.SAFE_INPUT_OPTS
    audio.TIMEOUT = 300
    data = Path(os.environ.get("VM_DATA", ROOT / ".webdata"))
    # 예시 녹음은 공개 음성만으로 만든 것(tools/make_public_demo.py)을 쓴다. 개인 목소리가 든 demo_library는 쓰지 않는다.
    demo_dir = os.environ.get("VM_DEMO_DIR", str(ROOT / "deploy" / "demo" / "library"))
    demo_ref = os.environ.get("VM_DEMO_REF", str(ROOT / "deploy" / "demo" / "reference.wav"))
    opts = dict(demo_dir=demo_dir, demo_ref=demo_ref, extractor_factory=_extractor_factory, devices=_devices())
    opts.update(kw)
    return Service(data, opts.pop("engine_factory", _engine_factory), **opts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:
        app.state.service = build_service()
    yield
    app.state.service.stop()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.state.service = None
app.state.limiter = defaultdict(deque)  # 사람마다 최근 세션을 만든 시각


def svc(request: Request) -> Service:
    return request.app.state.service


@app.middleware("http")
async def headers(request: Request, call_next):
    response = await call_next(request)
    h = response.headers
    h["Content-Security-Policy"] = CSP
    h["X-Content-Type-Options"] = "nosniff"
    h["Referrer-Policy"] = "no-referrer"
    h["X-Frame-Options"] = "DENY"
    h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    h["Cross-Origin-Opener-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        h["Cache-Control"] = "no-store"
    elif "Cache-Control" not in h:
        h["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(UserError)
async def user_error(request: Request, exc: UserError):
    code = 404 if isinstance(exc, Gone) else 429 if isinstance(exc, Busy) else 400
    return JSONResponse({"error": str(exc)}, status_code=code)


def client_ip(request: Request) -> str:
    if os.environ.get("VM_TRUST_PROXY") == "1" and request.headers.get("cf-connecting-ip"):
        return request.headers["cf-connecting-ip"]
    return request.client.host if request.client else "?"


def wav(data: bytes) -> Response:
    return Response(data, media_type="audio/wav")


# ── 화면 ────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def limits_json(service: Service) -> dict:
    lim = service.limits
    return {"files": lim.max_files, "mb": lim.max_bytes // 1024 ** 2, "file_mb": lim.max_file_bytes // 1024 ** 2,
            "minutes": int(lim.max_audio_sec // 60), "chunk": lim.chunk_bytes, "ttl_minutes": int(lim.ttl_sec // 60),
            "exts": sorted(e.strip(".") for e in audio.MEDIA_EXTS)}


@app.get("/api/info")
def info(request: Request):
    s = svc(request)
    return {"limits": limits_json(s), "demo": s.demo_dir is not None}


@app.get("/api/health")
def health(request: Request):
    s = svc(request)
    return {"ready": s.ready.is_set(), "queue": len(s.queue)}


# ── 세션 ────────────────────────────────────────────────
@app.post("/api/sessions")
def create_session(request: Request):
    ip, now = client_ip(request), time.time()
    seen = request.app.state.limiter[ip]
    while seen and now - seen[0] > 3600:
        seen.popleft()
    if len(seen) >= 12:
        raise Busy("짧은 시간에 너무 많이 시작했어요. 잠시 뒤에 다시 와 주세요.")
    seen.append(now)
    s = svc(request).create()
    return {"id": s.id, "limits": limits_json(svc(request))}


@app.get("/api/sessions/{sid}")
def status(sid: str, request: Request):
    s = svc(request)
    return s.status(s.get(sid))


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str, request: Request):
    svc(request).delete(sid)
    return {"ok": True}


class UploadStart(BaseModel):
    name: str = Field(max_length=300)
    size: int
    modified: float | None = None


@app.post("/api/sessions/{sid}/uploads")
def start_upload(sid: str, body: UploadStart, request: Request):
    s = svc(request)
    u = s.start_upload(s.get(sid), body.name, body.size, body.modified)
    return {"id": u.id, "name": u.name}


async def read_body(request: Request, limit: int) -> bytes:
    """본문을 읽되 limit을 넘으면 거기서 멈춘다(큰 요청으로 메모리를 채우지 못하게)."""
    chunks, size = [], 0
    async for part in request.stream():
        size += len(part)
        if size > limit:
            raise UserError("한 번에 보낸 내용이 너무 커요.")
        chunks.append(part)
    return b"".join(chunks)


@app.put("/api/sessions/{sid}/uploads/{uid}")
async def upload_chunk(sid: str, uid: str, offset: int, request: Request):
    s = svc(request)
    sess = s.get(sid)
    data = await read_body(request, s.limits.chunk_bytes)
    return {"received": s.write_chunk(sess, uid, offset, data)}


@app.post("/api/sessions/{sid}/uploads/{uid}/done")
def finish_upload(sid: str, uid: str, request: Request):
    s = svc(request)
    u = s.finish_upload(s.get(sid), uid)
    return {"id": u.id, "seconds": round(u.seconds, 1)}


@app.delete("/api/sessions/{sid}/uploads/{uid}")
def remove_upload(sid: str, uid: str, request: Request):
    s = svc(request)
    s.remove_upload(s.get(sid), uid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/demo")
def use_demo(sid: str, request: Request):
    s = svc(request)
    sess = s.get(sid)
    s.use_demo(sess)
    s.scan(sess)
    return {"ok": True}


@app.post("/api/sessions/{sid}/scan")
def scan(sid: str, request: Request):
    s = svc(request)
    s.scan(s.get(sid))
    return {"ok": True}


# ── 목소리 고르기 ───────────────────────────────────────
@app.get("/api/sessions/{sid}/voices")
def voices(sid: str, request: Request):
    s = svc(request)
    return s.voices(s.get(sid))


@app.get("/api/sessions/{sid}/voices/{index}.wav")
def voice_audio(sid: str, index: int, request: Request):
    s = svc(request)
    return wav(s.voice_audio(s.get(sid), index))


class Pick(BaseModel):
    voices: list[int] = Field(max_length=500)
    follow: bool | None = None
    person: str | None = Field(default=None, max_length=60)


@app.post("/api/sessions/{sid}/pick")
def pick(sid: str, body: Pick, request: Request):
    s = svc(request)
    sess = s.get(sid)
    if body.person is not None:
        s.settings(sess, person=body.person)
    s.pick(sess, body.voices, body.follow)
    return {"ok": True}


@app.put("/api/sessions/{sid}/reference")
async def reference(sid: str, request: Request, name: str = "참고.m4a", person: str | None = None):
    s = svc(request)
    sess = s.get(sid)
    if Path(name).suffix.lower() not in audio.MEDIA_EXTS:
        raise UserError("참고 녹음은 영상이나 녹음 파일로 올려 주세요.")
    data = await read_body(request, s.limits.max_ref_bytes)
    ref_dir = sess.dir / "ref"
    ref_dir.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ref_dir, suffix=Path(name).suffix.lower())
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    if person is not None:
        s.settings(sess, person=person)
    s.reference(sess, Path(tmp))
    return {"ok": True}


@app.post("/api/sessions/{sid}/demo-reference")
def demo_reference(sid: str, request: Request, person: str | None = None):
    s = svc(request)
    sess = s.get(sid)
    if person is not None:
        s.settings(sess, person=person)
    s.demo_reference(sess)
    return {"ok": True}


class Same(BaseModel):
    index: int
    yes: bool


@app.post("/api/sessions/{sid}/same")
def same(sid: str, body: Same, request: Request):
    s = svc(request)
    s.answer_same(s.get(sid), body.index, body.yes)
    return {"ok": True}


# ── 결과 ────────────────────────────────────────────────
class Review(BaseModel):
    file: int
    start: float
    end: float
    answer: str = Field(pattern="^(yes|no|undo)$")
    overlap: bool = False


@app.post("/api/sessions/{sid}/review")
def review(sid: str, body: Review, request: Request):
    s = svc(request)
    s.review(s.get(sid), body.file, body.start, body.end, body.answer, body.overlap)
    return {"ok": True}


class Settings(BaseModel):
    person: str | None = Field(default=None, max_length=60)
    level: int | None = None


@app.post("/api/sessions/{sid}/settings")
def settings(sid: str, body: Settings, request: Request):
    s = svc(request)
    s.settings(s.get(sid), body.person, body.level)
    return {"ok": True}


@app.get("/api/sessions/{sid}/result")
def result(sid: str, request: Request, view: str = "year"):
    s = svc(request)
    return s.result(s.get(sid), view)


@app.get("/api/sessions/{sid}/groups/{view}/{key}.wav")
def group_audio(sid: str, view: str, key: str, request: Request, full: int = 0):
    s = svc(request)
    return wav(s.group_audio(s.get(sid), view, key, bool(full)))


@app.get("/api/sessions/{sid}/clip.wav")
def clip_audio(sid: str, file: int, start: float, end: float, request: Request):
    s = svc(request)
    return wav(s.clip_audio(s.get(sid), file, start, end))


@app.get("/api/sessions/{sid}/download.zip")
def download(sid: str, request: Request):
    s = svc(request)
    name, data = s.zip_bytes(s.get(sid))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=\"voices.zip\"; filename*=UTF-8''{quote(name)}"})
