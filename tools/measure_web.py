"""온라인 서비스에 녹음 폴더를 올려 단계별로 걸리는 시간을 잰다(표준 라이브러리만, 서버에서 바로 돌린다).

    python3 tools/measure_web.py http://127.0.0.1:8600 ~/loadtest [참고녹음.wav]

잰 값으로 web/service.py의 걸리는 시간 어림(VM_SCAN_RATE, VM_SEARCH_RATE)을 정한다.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".amr", ".3gp", ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def call(base: str, method: str, path: str, body=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"Content-Type": "application/octet-stream" if raw is not None else "application/json",
               "User-Agent": "voice-memoir-measure/1.0"}
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wait(base: str, sid: str) -> tuple[dict, float, float]:
    """작업이 끝날 때까지 기다린다. (상태, 걸린 시간, 화면이 처음 보여 준 어림)."""
    t0, first_estimate = time.time(), 0.0
    while True:
        st = call(base, "GET", f"/api/sessions/{sid}")
        if st.get("estimate") and not first_estimate:
            first_estimate = st["estimate"]
        if st["state"] not in ("queued", "scanning", "searching"):
            return st, time.time() - t0, first_estimate
        time.sleep(1)


def main() -> None:
    base, folder = sys.argv[1].rstrip("/"), Path(sys.argv[2]).expanduser()
    ref = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 else None
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    sid = call(base, "POST", "/api/sessions")["id"]
    t0, total = time.time(), 0.0
    for f in files:
        data = f.read_bytes()
        u = call(base, "POST", f"/api/sessions/{sid}/uploads",
                 {"name": f.name, "size": len(data), "modified": f.stat().st_mtime * 1000})
        for off in range(0, len(data), 8 * 1024 * 1024):
            call(base, "PUT", f"/api/sessions/{sid}/uploads/{u['id']}?offset={off}", raw=data[off:off + 8 * 1024 * 1024])
        total += call(base, "POST", f"/api/sessions/{sid}/uploads/{u['id']}/done")["seconds"]
    print(f"올리기: 파일 {len(files)}개, 녹음 {total / 60:.1f}분, {time.time() - t0:.1f}초", flush=True)

    call(base, "POST", f"/api/sessions/{sid}/scan")
    st, sec, est = wait(base, sid)
    speech = st["summary"]["speech"]
    voices = call(base, "GET", f"/api/sessions/{sid}/voices")["voices"]
    print(f"살펴보기: {sec:.1f}초 (어림 {est:.0f}초) · 녹음 {total / sec:.1f}배 빠르기 · 말소리 {speech / 60:.1f}분 · 목소리 {len(voices)}개",
          flush=True)

    call(base, "POST", f"/api/sessions/{sid}/pick", {"voices": [voices[0]["index"]], "person": "측정"})
    st, sec, est = wait(base, sid)
    res = call(base, "GET", f"/api/sessions/{sid}/result")
    print(f"고른 목소리로 찾기: {sec:.1f}초 (어림 {est:.0f}초) · 말소리 {speech / sec:.1f}배 빠르기 · 찾은 {res['total']:.0f}초", flush=True)

    if ref:
        data = ref.read_bytes()
        req = urllib.request.Request(f"{base}/api/sessions/{sid}/reference?name={ref.name}", data=data, method="PUT",
                                     headers={"Content-Type": "application/octet-stream", "User-Agent": "voice-memoir-measure/1.0"})
        urllib.request.urlopen(req, timeout=120).read()
        st, sec, est = wait(base, sid)
        res = call(base, "GET", f"/api/sessions/{sid}/result")
        print(f"참고 녹음으로 찾기: {sec:.1f}초 (어림 {est:.0f}초) · 찾은 {res['total']:.0f}초", flush=True)

    t = time.time()
    urllib.request.urlopen(urllib.request.Request(f"{base}/api/sessions/{sid}/download.zip",
                                                  headers={"User-Agent": "voice-memoir-measure/1.0"}), timeout=300).read()
    print(f"zip 만들기: {time.time() - t:.1f}초", flush=True)
    call(base, "DELETE", f"/api/sessions/{sid}")


if __name__ == "__main__":
    main()
