"""온라인 서비스(web/): 올리기 → 살펴보기 → 고르기 → 결과, 그리고 올린 녹음을 지키는 장치들.

가짜 엔진(conftest)으로 모델 없이 돈다. 작업자 스레드를 켜지 않고 drain()으로 줄을 직접 비워
결과가 매번 같게 한다.
"""
from __future__ import annotations

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from vm import audio
from web import server
from web.service import Busy, Gone, Limits, Service, UserError, clean_name

from conftest import FakeEngine, compose, needs_ffmpeg

pytestmark = needs_ffmpeg


def wav_bytes(*parts) -> bytes:
    return audio.wav_bytes(compose(*parts))


@pytest.fixture
def demo(tmp_path):
    d = tmp_path / "예시"
    d.mkdir()
    audio.write_wav(d / "2019-03-01 거실.wav", compose(("me", 5), ("other", 3)))
    audio.write_wav(d / "2021-07-07 식당.wav", compose(("other", 3), ("me", 4)))
    ref = tmp_path / "참고.wav"
    audio.write_wav(ref, compose(("me", 4)))
    return d, ref


def make_service(tmp_path, demo=None, **limits) -> Service:
    d, ref = demo if demo else (None, None)
    return Service(tmp_path / "data", FakeEngine, limits=Limits(**limits), demo_dir=d, demo_ref=ref,
                   devices=["cpu"], start=False)


def drain(svc: Service) -> None:
    """작업자 대신 줄에 선 작업을 차례로 처리한다."""
    while svc.queue:
        sid, kind, fn = svc.queue.popleft()
        s = svc.sessions.get(sid)
        if s is not None and not s.deleted:
            svc.running = sid
            svc.run_job(s, kind, fn)


@pytest.fixture
def client(tmp_path, demo):
    svc = make_service(tmp_path, demo)
    server.app.state.service = svc
    with TestClient(server.app) as c:
        c.svc = svc
        yield c
    server.app.state.service = None


def upload(c, sid: str, name: str, data: bytes, chunk: int = 50_000, modified: float | None = None) -> dict:
    r = c.post(f"/api/sessions/{sid}/uploads", json={"name": name, "size": len(data), "modified": modified})
    assert r.status_code == 200, r.text
    uid = r.json()["id"]
    for off in range(0, len(data), chunk):
        r = c.put(f"/api/sessions/{sid}/uploads/{uid}?offset={off}", content=data[off:off + chunk])
        assert r.status_code == 200, r.text
    r = c.post(f"/api/sessions/{sid}/uploads/{uid}/done")
    assert r.status_code == 200, r.text
    return r.json()


def new_session(c) -> str:
    r = c.post("/api/sessions")
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ── 흐름 ────────────────────────────────────────────────
def test_upload_scan_pick_and_result(client):
    c = client
    sid = new_session(c)
    upload(c, sid, "2018-02-03 설날.wav", wav_bytes(("me", 6), ("other", 3)))
    upload(c, sid, "통화녹음 2022-05-01.wav", wav_bytes(("other", 2), ("me", 5)))
    assert c.post(f"/api/sessions/{sid}/scan").json() == {"ok": True}
    assert c.get(f"/api/sessions/{sid}").json()["state"] == "queued"
    drain(c.svc)
    st = c.get(f"/api/sessions/{sid}").json()
    assert st["state"] == "ready" and st["summary"]["files"] == 2
    assert {f["date"] for f in st["files"]} == {"2018년 2월 3일", "2022년 5월 1일"}

    voices = c.get(f"/api/sessions/{sid}/voices").json()["voices"]
    assert len(voices) >= 2 and all(len(v["env"]) == 60 for v in voices)
    sample = c.get(f"/api/sessions/{sid}/voices/{voices[0]['index']}.wav")
    assert sample.headers["content-type"] == "audio/wav" and sample.content[:4] == b"RIFF"

    me = voices[0]["index"]  # 찾을 사람이 가장 길게 말한다
    assert c.post(f"/api/sessions/{sid}/pick", json={"voices": [me], "person": "할머니"}).status_code == 200
    drain(c.svc)
    res = c.get(f"/api/sessions/{sid}/result").json()
    assert res["person"] == "할머니" and res["particle"] == "와"
    assert [g["title"] for g in res["groups"]] == ["2018", "2022"]
    assert 9.0 <= res["total"] <= 12.0, res["total"]
    g = res["groups"][0]
    assert g["files"][0]["spans"] and len(g["main"]["env"]) == 90
    assert c.get(f"/api/sessions/{sid}/groups/year/2018.wav").content[:4] == b"RIFF"
    by_date = c.get(f"/api/sessions/{sid}/result?view=date").json()
    assert [g["title"] for g in by_date["groups"]] == ["2018년 2월 3일", "2022년 5월 1일"]


def test_demo_reference_review_level_and_zip(client):
    c = client
    sid = new_session(c)
    assert c.post(f"/api/sessions/{sid}/demo").status_code == 200
    drain(c.svc)
    assert c.post(f"/api/sessions/{sid}/demo-reference?person=아버지").status_code == 200
    drain(c.svc)
    res = c.get(f"/api/sessions/{sid}/result").json()
    assert res["source"] == "참고 녹음" and res["total"] > 0
    clip = res["groups"][0]["clips"][0]

    # '아니에요'로 고른 구간은 결과에서 빠지고, 되돌리면 다시 들어온다
    body = {"file": clip["file"], "start": clip["start"], "end": clip["end"], "answer": "no"}
    assert c.post(f"/api/sessions/{sid}/review", json=body).status_code == 200
    drain(c.svc)
    after = c.get(f"/api/sessions/{sid}/result").json()
    starts = [(x["file"], x["start"]) for g in after["groups"] for x in g["clips"]]
    assert (clip["file"], clip["start"]) not in starts
    assert after["reviewed"][0]["answer"] == "no"
    c.post(f"/api/sessions/{sid}/review", json={**body, "answer": "undo"})
    drain(c.svc)
    assert c.get(f"/api/sessions/{sid}/result").json()["reviewed"] == []

    assert c.post(f"/api/sessions/{sid}/settings", json={"level": 4}).status_code == 200
    drain(c.svc)
    assert c.get(f"/api/sessions/{sid}/result").json()["level"] == 4

    z = c.get(f"/api/sessions/{sid}/download.zip")
    assert z.status_code == 200 and "filename*=UTF-8''" in z.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "아버지_구간목록.csv" in names and any(n.startswith("구간별/") for n in names)


def test_demo_files_are_never_deleted(client, demo):
    c = client
    sid = new_session(c)
    c.post(f"/api/sessions/{sid}/demo")
    drain(c.svc)
    c.delete(f"/api/sessions/{sid}")
    assert len(list(demo[0].iterdir())) == 2  # 예시 녹음은 여러 사람이 함께 쓰므로 지우지 않는다
    assert c.get(f"/api/sessions/{sid}").status_code == 404


# ── 올린 녹음 지키기 ────────────────────────────────────
def test_wipe_removes_everything(client):
    c = client
    sid = new_session(c)
    upload(c, sid, "a.wav", wav_bytes(("me", 3)))
    folder = c.svc.sessions[sid].dir
    assert any(folder.rglob("a.wav"))
    assert c.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    assert not folder.exists()
    r = c.get(f"/api/sessions/{sid}")
    assert r.status_code == 404 and "지워져" in r.json()["error"]


def test_idle_sessions_are_swept(tmp_path):
    svc = make_service(tmp_path, ttl_sec=60, max_age_sec=600)
    old, fresh = svc.create(), svc.create()
    old.touched -= 120
    assert svc.sweep() == 1
    assert not old.dir.exists() and fresh.dir.exists()
    with pytest.raises(Gone):
        svc.get(old.id)
    fresh.created -= 601  # 계속 쓰고 있어도 최대 시간이 지나면 지운다
    assert svc.sweep() == 1


def test_busy_sessions_are_not_swept(tmp_path):
    svc = make_service(tmp_path, ttl_sec=60)
    s = svc.create()
    s.state, s.touched = "searching", time.time() - 999
    assert svc.sweep() == 0


def test_leftover_sessions_are_wiped_on_start(tmp_path):
    left = tmp_path / "data" / "sessions" / "지난번" / "uploads"
    left.mkdir(parents=True)
    (left / "가족.m4a").write_bytes(b"x")
    make_service(tmp_path)
    assert not (tmp_path / "data" / "sessions" / "지난번").exists()


@pytest.mark.parametrize("raw,clean", [
    ("../../etc/passwd.wav", "passwd.wav"),
    ("C:\\Users\\엄마\\통화녹음 2023.m4a", "통화녹음 2023.m4a"),
    (".숨김.wav", "숨김.wav"),
    ("가\x00족\n.mp4", "가족.mp4"),
])
def test_upload_names_cannot_escape(raw, clean):
    assert clean_name(raw) == clean


def test_upload_is_saved_inside_session_with_original_name(client):
    c = client
    sid = new_session(c)
    upload(c, sid, "../../밖으로 2020-01-01.wav", wav_bytes(("me", 2)), modified=1577836800000)
    s = c.svc.sessions[sid]
    (u,) = s.uploads.values()
    assert u.path.name == "밖으로 2020-01-01.wav" and s.media_dir in u.path.parents
    assert abs(u.path.stat().st_mtime - 1577836800) < 1  # 파일 시각도 원래대로


def test_bad_uploads_are_refused(client):
    c = client
    sid = new_session(c)
    for body in ({"name": "설치.exe", "size": 10}, {"name": "빈.wav", "size": 0},
                 {"name": "큰.mp4", "size": 10 ** 12}):
        r = c.post(f"/api/sessions/{sid}/uploads", json=body)
        assert r.status_code == 400 and r.json()["error"], body
    r = c.post(f"/api/sessions/{sid}/uploads", json={"name": "a.wav", "size": 10})
    uid = r.json()["id"]
    assert c.put(f"/api/sessions/{sid}/uploads/{uid}?offset=5", content=b"12345").status_code == 400  # 순서가 틀림
    assert c.put(f"/api/sessions/{sid}/uploads/{uid}?offset=0", content=b"x" * 11).status_code == 400  # 알린 크기보다 큼
    assert c.post(f"/api/sessions/{sid}/uploads/{uid}/done").status_code == 400  # 끝까지 안 받음


def test_unreadable_upload_is_removed(client):
    c = client
    sid = new_session(c)
    data = b"not audio at all" * 100
    r = c.post(f"/api/sessions/{sid}/uploads", json={"name": "가짜.m4a", "size": len(data)})
    uid = r.json()["id"]
    c.put(f"/api/sessions/{sid}/uploads/{uid}?offset=0", content=data)
    r = c.post(f"/api/sessions/{sid}/uploads/{uid}/done")
    assert r.status_code == 400 and "소리를 읽지 못했어요" in r.json()["error"]
    assert c.svc.sessions[sid].uploads == {}


def test_total_audio_length_is_limited(tmp_path):
    svc = make_service(tmp_path, max_audio_sec=5)
    s = svc.create()
    data = wav_bytes(("me", 4))
    u = svc.start_upload(s, "a.wav", len(data))
    svc.write_chunk(s, u.id, 0, data)
    svc.finish_upload(s, u.id)
    u2 = svc.start_upload(s, "b.wav", len(data))
    svc.write_chunk(s, u2.id, 0, data)
    with pytest.raises(UserError, match="5분|0분"):
        svc.finish_upload(s, u2.id)
    assert list(s.uploads) == [u.id] and not u2.path.exists()


def test_playlist_disguised_as_video_is_not_opened(tmp_path, monkeypatch):
    """재생목록을 영상인 척 올려 서버의 다른 파일을 읽게 하는 시도를 ffmpeg 단계에서 막는다."""
    monkeypatch.setattr(audio, "INPUT_OPTS", audio.SAFE_INPUT_OPTS)
    secret = tmp_path / "secret.wav"
    audio.write_wav(secret, compose(("me", 2)))
    evil = tmp_path / "가족.mp4"
    evil.write_text(f"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:2.0,\n{secret}\n#EXT-X-ENDLIST\n")
    assert audio.duration_of(audio.probe(evil)) == 0
    with pytest.raises(Exception):
        audio.decode(evil)
    assert audio.duration_of(audio.probe(secret)) > 1.5  # 정상 녹음은 그대로 열린다


def test_clip_requests_stay_inside_the_recording(client):
    c = client
    sid = new_session(c)
    upload(c, sid, "a.wav", wav_bytes(("me", 3)))
    c.post(f"/api/sessions/{sid}/scan")
    drain(c.svc)
    ok = c.get(f"/api/sessions/{sid}/clip.wav?file=0&start=0.5&end=2.5")
    assert ok.status_code == 200 and ok.content[:4] == b"RIFF"
    for q in ("file=1&start=0&end=1", "file=0&start=2&end=1", "file=0&start=0&end=99", "file=-1&start=0&end=1"):
        assert c.get(f"/api/sessions/{sid}/clip.wav?{q}").status_code == 400, q


def test_queue_is_bounded_and_coalesces(tmp_path):
    svc = make_service(tmp_path, max_queue=1)
    a, b = svc.create(), svc.create()
    svc.submit(a, "search", lambda s: None)
    svc.submit(a, "search", lambda s: None)  # 같은 세션의 같은 일은 새 것으로 바꾼다
    assert len(svc.queue) == 1
    with pytest.raises(Busy):
        svc.submit(b, "search", lambda s: None)
    assert svc.position(a.id) == 0


def test_too_many_sessions_are_refused(tmp_path):
    svc = make_service(tmp_path, max_sessions=2)
    svc.create(), svc.create()
    with pytest.raises(Busy):
        svc.create()


def test_session_creation_is_rate_limited(client):
    """한 접속 주소에서 동시에 3개, 한 시간에 12개까지만 만든다."""
    c = client
    made = [new_session(c) for _ in range(3)]
    r = c.post("/api/sessions")
    assert r.status_code == 429 and "이미 열어 둔 작업" in r.json()["error"]
    for _ in range(9):  # 다 쓰고 지우면 다시 만들 수 있지만, 한 시간에 12번까지
        c.delete(f"/api/sessions/{made.pop()}")
        made.append(new_session(c))
    c.delete(f"/api/sessions/{made.pop()}")
    r = c.post("/api/sessions")
    assert r.status_code == 429 and "너무 많이" in r.json()["error"]


def test_security_headers(client):
    c = client
    page = c.get("/")
    assert page.status_code == 200 and "목소리 발굴" in page.text
    h = page.headers
    assert "default-src 'self'" in h["content-security-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff" and h["referrer-policy"] == "no-referrer"
    assert c.get("/api/info").headers["cache-control"] == "no-store"
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_page_uses_no_outside_resources(client):
    """글꼴·스크립트를 모두 서버에서 내줘, 방문 기록이 다른 회사로 새지 않게 한다."""
    c = client
    for path in ("/", "/static/app.css", "/static/app.js"):
        text = c.get(path).text
        assert "googleapis" not in text and "jsdelivr" not in text and "cdn." not in text, path


# ── 다듬기: 다시 보내기, 남은 시간, 처리방침 ─────────────
def test_resent_chunk_is_accepted_once(client):
    """응답을 못 받아 같은 조각을 다시 보내도 두 번 쓰지 않는다."""
    c = client
    sid = new_session(c)
    data = wav_bytes(("me", 2))
    uid = c.post(f"/api/sessions/{sid}/uploads", json={"name": "a.wav", "size": len(data)}).json()["id"]
    half = len(data) // 2
    assert c.put(f"/api/sessions/{sid}/uploads/{uid}?offset=0", content=data[:half]).json()["received"] == half
    assert c.put(f"/api/sessions/{sid}/uploads/{uid}?offset=0", content=data[:half]).json()["received"] == half
    c.put(f"/api/sessions/{sid}/uploads/{uid}?offset={half}", content=data[half:])
    assert c.post(f"/api/sessions/{sid}/uploads/{uid}/done").status_code == 200
    (u,) = c.svc.sessions[sid].uploads.values()
    assert u.path.read_bytes() == data


def test_status_carries_time_estimate(client):
    c = client
    sid = new_session(c)
    upload(c, sid, "a.wav", wav_bytes(("me", 4), ("other", 4)))
    c.post(f"/api/sessions/{sid}/scan")
    sid_, kind, fn = c.svc.queue[0]
    s = c.svc.sessions[sid]
    seen = {}

    def spy(sess):
        seen.update(c.get(f"/api/sessions/{sid}").json())
        fn(sess)

    c.svc.queue.clear()
    c.svc.run_job(s, kind, spy)
    assert seen["state"] == "scanning" and seen["estimate"] > 0 and seen["elapsed"] >= 0
    assert c.get(f"/api/sessions/{sid}").json()["elapsed"] == 0  # 끝나면 0


def test_privacy_page_and_ip_forgetting(client):
    c = client
    page = c.get("/privacy")
    assert page.status_code == 200 and "개인정보 처리방침" in page.text and "1시간" in page.text
    assert "style=" not in page.text and "style=" not in c.get("/").text  # CSP가 막는 인라인 스타일을 쓰지 않는다
    svc = c.svc
    c.post("/api/sessions")
    (ip,) = svc.recent
    svc.recent[ip][-1] -= 3601  # 한 시간 전 기록이 되면
    svc.sweep()  # 청소 때 잊는다(새 세션을 만들지 않아도)
    assert ip not in svc.recent


# ── 보안 검토 뒤 고친 것 ────────────────────────────────
def test_review_rejects_out_of_range_regions(client):
    """'맞아요·아니에요'의 구간도 듣기 구간과 같은 범위만 받는다(긴 구간으로 작업자를 붙잡지 못하게)."""
    c = client
    sid = new_session(c)
    c.post(f"/api/sessions/{sid}/demo")
    drain(c.svc)
    c.post(f"/api/sessions/{sid}/demo-reference")
    drain(c.svc)
    for start, end in ((0, 1e9), (-1, 2), (3, 2), (0, 500)):
        body = {"file": 0, "start": start, "end": end, "answer": "no"}
        r = c.post(f"/api/sessions/{sid}/review", json=body)
        assert r.status_code == 400, (start, end, r.text)
    r = c.post(f"/api/sessions/{sid}/review", content='{"file":0,"start":0,"end":Infinity,"answer":"no"}',
               headers={"content-type": "application/json"})
    assert r.status_code in (400, 422), r.text
    assert not c.svc.queue
    assert c.post(f"/api/sessions/{sid}/review", json={"file": 0, "start": 1, "end": 2, "answer": "no"}).status_code == 200
    assert c.post(f"/api/sessions/{sid}/same", json={"index": 999, "yes": True}).status_code == 400


def test_empty_sessions_are_swept_sooner(tmp_path):
    svc = make_service(tmp_path, empty_ttl_sec=600)
    empty, used = svc.create(), svc.create()
    data = wav_bytes(("me", 1))
    u = svc.start_upload(used, "a.wav", len(data))
    svc.write_chunk(used, u.id, 0, data)
    empty.created -= 601
    used.created -= 601
    assert svc.sweep() == 1
    assert empty.id not in svc.sessions and used.id in svc.sessions


def test_failed_delete_is_retried(tmp_path, monkeypatch):
    import web.service as service_mod
    svc = make_service(tmp_path)
    s = svc.create()
    real = service_mod.shutil.rmtree
    monkeypatch.setattr(service_mod.shutil, "rmtree", lambda *a, **k: None)  # 지우기 실패
    svc.delete(s.id)
    assert s.dir.exists() and svc.leftovers == [s.dir] and svc.delete_failures == 1
    monkeypatch.setattr(service_mod.shutil, "rmtree", real)
    svc.sweep()
    assert not s.dir.exists() and svc.leftovers == []


def test_purge_forgets_extracted_audio(tmp_path):
    class Ex:
        def __init__(self):
            self.forgot = []

        def forget(self, prefix):
            self.forgot.append(prefix)

    svc = make_service(tmp_path)
    svc._engine, svc._extractor = FakeEngine(), Ex()
    s = svc.create()
    svc.delete(s.id)
    assert svc._extractor.forgot == [str(s.dir)]


def test_extractor_forget_drops_only_that_session():
    from collections import OrderedDict
    from vm.tse import Extractor
    ex = Extractor.__new__(Extractor)
    ex._cache = OrderedDict({(("/data/sessions/A/uploads/x/a.wav", 0, 4), 1, (1, 0.0)): 1,
                             (("/data/sessions/B/uploads/y/b.wav", 0, 4), 1, (1, 0.0)): 2})
    assert ex.forget("/data/sessions/A") == 1
    assert [k[0][0] for k in ex._cache] == ["/data/sessions/B/uploads/y/b.wav"]


def test_decoding_is_capped(tmp_path, monkeypatch):
    """길이 정보를 속인 파일이라도 정한 길이 이상은 풀지 않는다."""
    f = tmp_path / "long.wav"
    audio.write_wav(f, compose(("me", 6)))
    monkeypatch.setattr(audio, "MAX_SEC", 2)
    assert len(audio.decode(f)) <= 2 * audio.SR + 10


def test_zip_is_built_once_per_result(client):
    c = client
    sid = new_session(c)
    c.post(f"/api/sessions/{sid}/demo")
    drain(c.svc)
    c.post(f"/api/sessions/{sid}/demo-reference")
    drain(c.svc)
    s = c.svc.sessions[sid]
    first = c.svc.zip_bytes(s)
    assert c.svc.zip_bytes(s) is first  # 같은 결과면 다시 만들지 않는다
    c.post(f"/api/sessions/{sid}/settings", json={"level": 4})
    drain(c.svc)
    assert c.svc.zip_bytes(s) is not first  # 결과가 바뀌면 새로 만든다


def test_http_is_sent_to_https_with_hsts(client, monkeypatch):
    c = client
    monkeypatch.setenv("VM_TRUST_PROXY", "1")
    r = c.get("/privacy?x=1", headers={"cf-visitor": '{"scheme":"http"}'}, follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "https://testserver/privacy?x=1"
    r = c.get("/", headers={"cf-visitor": '{"scheme":"https"}'})
    assert r.status_code == 200 and r.headers["strict-transport-security"].startswith("max-age=")
    monkeypatch.delenv("VM_TRUST_PROXY")
    assert "strict-transport-security" not in c.get("/").headers  # 로컬 실행에서는 넘기지 않는다


def test_static_files_are_versioned(client):
    page = client.get("/").text
    assert "/static/app.js?v=" in page and "/static/app.css?v=" in page
    assert "/static/app.css?v=" in client.get("/privacy").text


def test_downloads_do_not_put_session_in_links(client):
    """세션 번호가 든 주소를 링크로 걸지 않아 다운로드 기록에 남지 않는다."""
    js = client.get("/static/app.js").text
    assert "download=\"" not in js and ".href = apiUrl" not in js and "createObjectURL" in js


def test_service_files_are_private(tmp_path, monkeypatch):
    import os
    from web import server as srv
    monkeypatch.setenv("VM_DATA", str(tmp_path / "data"))
    for name in ("INPUT_OPTS", "TIMEOUT", "MAX_SEC"):
        monkeypatch.setattr(audio, name, getattr(audio, name))
    before = os.umask(0o022)
    try:
        svc = srv.build_service(engine_factory=FakeEngine, start=False)
        s = svc.create()
        data = wav_bytes(("me", 1))
        u = svc.start_upload(s, "a.wav", len(data))
        assert (s.dir.stat().st_mode & 0o777) == 0o700
        assert (u.path.stat().st_mode & 0o777) == 0o600
        assert audio.MAX_SEC == svc.limits.max_audio_sec + 60
    finally:
        os.umask(before)
