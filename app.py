"""목소리 발굴 — 화면.

실행: .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import html
import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from visual import CSS, MOSS, OTHER, SAMPLE_IDLE, envelope, hero_strip, strip_svg
from vm import audio
from vm.engine import VoiceEngine
from vm.library import Library
from vm.text import josa
from vm.speakers import (apply_review, discover_speakers, search,
                         target_from_clusters, target_from_reference, with_review)
from vm.timeline import (all_selections, build_timeline, compile_highlight, select_highlights,
                         source_rows, sources_csv)

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_library"
DEMO_REF = ROOT / "demo_sources" / "참고음성_본인_7초.wav"
LEVELS = ("넉넉하게", "조금 넉넉하게", "보통", "조금 엄격하게", "엄격하게")  # engine.cal.levels와 짝
DATE_BASIS = {
    "metadata": "녹음 정보 기준",
    "filename": "파일 이름 기준",
    "file": "파일 시각 기준이라 정확하지 않을 수 있어요",
    "unknown": "날짜를 알 수 없어요",
}

st.set_page_config(page_title="목소리 발굴", page_icon="🎙️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="목소리 분석 모델을 불러오는 중이에요… (처음 한 번 약 100MB를 내려받아요)")
def get_engine() -> VoiceEngine:
    return VoiceEngine(ROOT / "models", device="auto")


@st.cache_resource(show_spinner=False)
def get_extractor():
    """다른 목소리와 겹친 구간용 대상 화자 추출. 가중치가 없으면 이 기능은 꺼 둔다."""
    weights = ROOT / "models" / "tse" / "td_speakerbeam_ko.pth"
    if not weights.exists():
        return None
    import torch
    from vm.tse import Extractor
    return Extractor(weights, "mps" if torch.backends.mps.is_available() else "cpu")


@st.cache_resource(show_spinner="받아쓰기 모델을 불러오는 중이에요… (처음 한 번 약 460MB를 내려받아요)")
def get_transcriber():
    from vm.transcribe import Transcriber
    return Transcriber()


def mmss(sec: float) -> str:
    return f"{int(sec // 60)}분 {int(sec % 60)}초" if sec >= 60 else f"{sec:.1f}초"


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def step(n: int, title: str, lede: str | None = None, state: str = "current") -> None:
    cls = {"done": "vm-step done", "current": "vm-step", "later": "vm-step later"}[state]
    html_block(f'<div class="{cls}"><span class="vm-step-n">{n}</span><h2>{html.escape(title)}</h2></div>'
               + (f'<p class="vm-lede">{lede}</p>' if lede else ""))


def date_text(d, source: str) -> str:
    when = f"{d.year}년 {d.month}월 {d.day}일에 녹음" if d else "녹음 날짜 모름"
    return f"{when} ({DATE_BASIS[source]})"


S = st.session_state
SHOWN = 9  # 처음에 펼쳐 보여 줄 목소리 카드 수
S.setdefault("records", None)
S.setdefault("clusters", None)
S.setdefault("target", None)
S.setdefault("results", {})
S.setdefault("cache", audio.AudioCache(max_files=12))
S.setdefault("accepted", [])  # 사람이 '맞아요'로 확인한 구간 (파일, 시작, 끝)
S.setdefault("rejected", [])  # 사람이 '아니에요'로 확인한 구간
S.setdefault("overlap_regions", set())  # 위 둘 중 다른 목소리와 겹친 구간(기준 지문·경쟁자에는 넣지 않는다)
S.setdefault("ref_wav", None)  # 참고 녹음(애매한 목소리를 확인하면 기준을 다시 만든다)
S.setdefault("same", set())    # 참고 녹음과 같은 사람이라고 확인한 목소리 번호
S.setdefault("not_same", set())
engine = get_engine()

# ── 설정 ─────────────────────────────────────────────────
with st.sidebar:
    person = st.text_input("찾는 사람 이름", value="아버지",
                           help="결과 파일 이름과 화면 제목에 들어가요.").strip() or "찾는 사람"
    level = st.select_slider("얼마나 확실할 때만 담을까요", options=LEVELS, value="보통")
    threshold = engine.cal.levels[LEVELS.index(level)]
    st.caption("높이면 다른 사람이 섞일 일이 줄고, 낮추면 더 많이 찾지만 목소리가 닮은 가족이 "
               "섞일 수 있어요. 결과를 들어 보고 맞게 조절해 주세요.")
    has_overlap = get_extractor() is not None
    use_overlap = has_overlap and st.checkbox(
        "겹쳐 말한 구간도 찾기", value=True,
        help="다른 사람과 동시에 말해 묻힌 목소리를 따로 뽑아 확인해 볼 구간으로 올려요. 처음 찾을 때 시간이 더 걸려요"
             "(말소리 1시간에 맥 GPU로 약 3분). 끄면 더 빨라요.")
    do_transcribe = st.checkbox("한 말을 글로 옮기기", value=False,
                                help="자동 받아쓰기라 틀린 글자가 있을 수 있어요. 처음 켤 때 받아쓰기 모델을 한 번 내려받아요(약 460MB).")
    st.divider()
    st.caption(":material/lock: 녹음과 분석 결과는 이 컴퓨터 밖으로 나가지 않아요.")

# ── 머리말 ───────────────────────────────────────────────
html_block(
    '<div class="vm-hero"><h1>목소리 발굴</h1>'
    '<p>가족 영상, 통화 녹음, 음성 메모에 흩어진 한 사람의 목소리를 찾아 해마다 모아 드려요. '
    '새로 녹음하지 않아도 돼요.</p>'
    f'<figure>{hero_strip()}<figcaption class="vm-legend">'
    f'<span><i style="background:{OTHER}"></i>다른 사람</span>'
    f'<span><i style="background:{MOSS}"></i>찾는 사람</span>'
    '</figcaption></figure></div>'
)

# ── 1 녹음 폴더 ──────────────────────────────────────────
step(1, "녹음 폴더 고르기", "영상과 통화 녹음이 섞여 있어도 괜찮아요. 하위 폴더까지 살펴봐요.",
     "done" if S.records is not None else "current")
with st.container(key="body-1"):
    c_in, c_btn = st.columns([5, 1.3], vertical_alignment="bottom")
    folder = c_in.text_input("녹음 폴더 위치", value=str(DEMO), label_visibility="collapsed")
    load = c_btn.button("불러오기", type="primary", width="stretch")

    if load:
        path = Path(folder).expanduser()
        if not path.is_dir():
            st.error(f"폴더를 찾지 못했어요. 위치를 다시 확인해 주세요: {path}")
        else:
            lib = Library(engine, ROOT / ".cache")
            bar = st.progress(0.0, text="녹음을 살펴보는 중이에요…")
            records = lib.scan(path, progress=lambda i, n, name: bar.progress(
                i / max(n, 1), text=f"녹음을 살펴보는 중이에요… {i}/{n} {name}"))
            bar.empty()
            with st.spinner("폴더에 나오는 목소리를 사람별로 나누는 중이에요…"):
                S.clusters = discover_speakers(records, engine, S.cache)
            S.records, S.target, S.results, S.root = records, None, {}, str(path)
            S.accepted, S.rejected, S.overlap_regions = [], [], set()
            S.ref_wav, S.same, S.not_same = None, set(), set()
            st.rerun()

    with st.expander("녹음을 어디서 가져오나요?"):
        st.markdown(
            "**아이폰 음성 메모**  \n"
            "아이폰 설정의 iCloud에서 ‘음성 메모’를 켜면 맥의 음성 메모 앱에 녹음이 나타나요. "
            "녹음을 Finder로 끌어다 놓으면 파일로 꺼내져요.\n\n"
            "**사진·영상**  \n"
            "맥 사진 앱에서 영상을 고른 뒤 파일 메뉴의 내보내기에서 ‘수정되지 않은 원본 내보내기’를 쓰면 "
            "찍은 날짜 정보가 그대로 남아요. 구글 포토를 쓴다면 Google 테이크아웃으로 원본을 한꺼번에 받을 수 있어요.\n\n"
            "**카카오톡**  \n"
            "PC 카카오톡에서 대화방 메뉴의 사진·동영상 모아보기로 여러 개를 골라 한 번에 저장할 수 있어요. "
            "기본 저장 위치는 ‘문서 > 카카오톡 받은 파일’이에요.\n\n"
            "**갤럭시 통화 녹음**  \n"
            "‘내 파일 > 내장 저장공간 > Recordings > Call’ 폴더에 있어요(One UI 4 이상). "
            "케이블이나 클라우드로 컴퓨터에 옮겨 주세요.\n\n"
            "**아이폰 통화 녹음**  \n"
            "iOS 18.1 이상의 기본 통화 녹음을 썼다면 메모 앱에 저장돼요. "
            "나라와 버전에 따라 쓸 수 없을 수도 있어요.\n\n"
            ":gray[녹음 날짜를 알아보기 쉽게 파일을 고치지 말고 원본 그대로 옮겨 주세요. "
            "메뉴 이름은 기기와 앱 버전에 따라 조금 다를 수 있어요.]")

    if S.records is None:
        html_block('<p class="vm-text">데모 폴더가 미리 채워져 있어요. <b>불러오기</b>를 누르면 가족 모임 영상, '
                   '카카오톡 영상, 통화 녹음 등 5개 파일로 바로 해 볼 수 있어요.</p>')
    else:
        records = S.records
        total_dur = sum(r.duration for r in records)
        speech = sum(r.speech_seconds for r in records)
        n_phone = sum(r.narrowband for r in records)
        n_err = sum(bool(r.error) for r in records)
        summary = (f"파일 {len(records)}개, 모두 {mmss(total_dur)} 분량을 살펴봤어요. "
                   f"그중 사람이 말하는 부분은 {mmss(speech)}예요.")
        if n_phone:
            summary += f" 통화 녹음으로 보이는 파일 {n_phone}개는 통화 음질에 맞춰 따로 비교해요."
        if n_err:
            summary += f" 읽지 못한 파일 {n_err}개는 아래 목록에서 확인할 수 있어요."
        html_block(f'<p class="vm-text">{summary}</p>')
        with st.expander("파일마다 녹음 날짜 확인하기"):
            st.dataframe(
                [{"파일": r.name, "녹음 날짜": str(r.date or "모름"), "날짜 근거": DATE_BASIS[r.date_source],
                  "길이": mmss(r.duration), "통화 녹음": "예" if r.narrowband else "",
                  "상태": f"읽지 못함: {r.error}" if r.error else "정상"} for r in records],
                hide_index=True, width="stretch")

if S.records is None:
    step(2, "찾는 사람 고르기", "폴더를 불러오면 이 자리에서 목소리들을 들어 보고 고를 수 있어요.", "later")
    step(3, "해마다 모은 목소리", "고른 사람의 목소리를 해마다, 날짜마다 모아 보여 드려요.", "later")
    step(4, "모아서 내려받기", "해마다 모은 음성 파일과, 어느 녹음의 몇 초에서 가져왔는지 적은 목록을 함께 받을 수 있어요.", "later")
    st.stop()
records = S.records

# ── 2 찾는 사람 ──────────────────────────────────────────
step(2, "찾는 사람 고르기",
     "목소리를 들어 보고 찾는 사람을 골라 주세요. 같은 사람으로 확실해 보이는 다른 목소리는 함께 찾아요.",
     "done" if S.target is not None else "current")
body2 = st.container(key="body-2")
tab_auto, tab_ref = body2.tabs(["이 폴더에 나온 목소리에서 고르기", "참고 녹음으로 지정하기"])

with tab_auto:
    clusters = S.clusters or []
    if not clusters:
        st.warning("말소리를 충분히 찾지 못했어요. 녹음이 너무 짧거나 조용하지 않은지 확인해 주세요.")
    else:
        chosen = {c.index for c in clusters if S.get(f"pick_{c.index}")}
        if chosen:
            html_block("<style>" + "".join(
                f".st-key-voice-{i} {{ border-color: var(--moss) !important; background: var(--moss-tint) !important; }}"
                f".st-key-voice-{i} .vm-chip.moss {{ background: var(--card); }}"
                for i in chosen) + "</style>")
        picked = []

        def voice_cards(group: list) -> None:
            for i, c in enumerate(group):
                if i % 3 == 0:
                    cols = st.columns(3)
                with cols[i % 3], st.container(key=f"voice-{c.index}"):
                    p, s0, e0 = c.sample
                    sample = S.cache.slice(p, s0, e0)
                    chip = '<div><span class="vm-chip moss">통화 녹음 포함</span></div>' if c.phone_linked else ""
                    on = c.index in chosen
                    html_block(
                        f'<div class="vm-voice-name">목소리 {c.index}</div>'
                        f'<div class="vm-voice-meta">파일 {len(c.files)}개에서 약 {mmss(c.seconds)}</div>{chip}'
                        + strip_svg(envelope(sample, 40), [(0, 1)], height=34,
                                    on=MOSS if on else SAMPLE_IDLE, label=f"목소리 {c.index} 들어 보기 파형"))
                    st.audio(audio.wav_bytes(sample), format="audio/wav")
                    if st.checkbox(f"목소리 {c.index} 선택", key=f"pick_{c.index}"):
                        picked.append(c)

        # 긴 녹음에서는 한 사람이 여러 목소리로 나뉘어 목록이 길어진다. 많이 말한 목소리부터 SHOWN개만 펼쳐 두고,
        # 나머지는 접어 둔다(한 사람의 짧은 조각은 카드 하나만 고르면 앱이 따라가 함께 찾는다).
        voice_cards(clusters[:SHOWN])
        if len(clusters) > SHOWN:
            rest = clusters[SHOWN:]
            with st.expander(f"짧게 나온 목소리 {len(rest)}개 더 보기", expanded=any(c.index in chosen for c in rest)):
                voice_cards(rest)
        if st.button("고른 목소리로 찾기", type="primary", disabled=not picked):
            S.target, S.results = target_from_clusters(picked, S.clusters, engine, S.cache), {}
            S.accepted, S.rejected, S.overlap_regions = [], [], set()
            S.ref_wav, S.same, S.not_same = None, set(), set()

with tab_ref:
    html_block('<p class="vm-text">찾는 사람이 말하는 녹음을 올려 주세요. 5초면 되지만, 30초쯤 되면 멀리서 녹음한 목소리까지 더 많이 찾아요. 다른 사람 목소리가 '
               '조금 섞여 있어도 가장 많이 말한 사람을 기준으로 삼아요.</p>')
    up = st.file_uploader("참고 녹음", type=[e.strip(".") for e in sorted(audio.MEDIA_EXTS)],
                          label_visibility="collapsed")
    b1, b2, _ = st.columns([1.2, 1.4, 2])
    use_up = b1.button("올린 녹음으로 찾기", disabled=up is None)
    use_demo = b2.button("데모 참고 녹음으로 찾기", disabled=not DEMO_REF.exists())
    if use_up or use_demo:
        try:
            if use_up:
                with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix) as tmp:
                    tmp.write(up.getvalue())
                    tmp.flush()
                    ref = audio.decode(tmp.name)
            else:
                ref = audio.decode(DEMO_REF)
            S.ref_wav, S.same, S.not_same = ref, set(), set()
            S.target, S.results = target_from_reference(engine, ref, S.clusters), {}
            S.accepted, S.rejected, S.overlap_regions = [], [], set()
            if S.target.info.get("mixed"):
                st.info("참고 녹음에 다른 목소리도 섞여 있어서, 가장 많이 말한 사람을 기준으로 삼았어요.")
        except ValueError as exc:
            st.error(str(exc))

    def same_voice(index: int, yes: bool) -> None:
        """애매한 목소리를 사람이 들어 보고 정한다. 같은 사람이면 기준을 다시 만든다."""
        (S.same if yes else S.not_same).add(index)
        if yes:
            S.target = target_from_reference(engine, S.ref_wav, S.clusters, same=S.same)
            S.results, S.accepted, S.rejected, S.overlap_regions = {}, [], [], set()

    unsure = []
    if S.target is not None and S.target.source == "참고 녹음":
        by_index = {c.index: c for c in S.clusters or []}
        unsure = [by_index[i] for i in S.target.info.get("unsure", [])
                  if i in by_index and i not in S.same | S.not_same]
    if unsure:
        with st.container(key="unsure-box"):
            html_block('<div class="vm-review-title">이 목소리도 같은 사람인가요?</div>'
                       '<p class="vm-text">참고 녹음과 비슷하지만 같은 사람인지 확실하지 않은 목소리예요. '
                       '지금은 다른 사람으로 보고 빼 두었어요. 같은 사람이면 알려 주세요. '
                       '시끄러운 곳이나 멀리서 녹음된 목소리를 더 찾을 수 있어요.</p>')
            for c in unsure[:4]:
                p, s0, e0 = c.sample
                st.markdown(f"목소리 {c.index}  \n:gray[파일 {len(c.files)}개에서 약 {mmss(c.seconds)}]")
                b, yes, no = st.columns([3.2, 1.3, 1.3], vertical_alignment="center")
                b.audio(audio.wav_bytes(S.cache.slice(p, s0, e0)), format="audio/wav")
                yes.button("같은 사람", key=f"same_{c.index}", on_click=same_voice, args=(c.index, True),
                           width="stretch")
                no.button("다른 사람", key=f"other_{c.index}", on_click=same_voice, args=(c.index, False),
                          width="stretch")

    def only_picked() -> None:
        """따라간 목소리 없이 사람이 고른 목소리만으로 다시 찾는다."""
        by_index = {c.index: c for c in S.clusters or []}
        picked = [by_index[i] for i in S.target.info["clusters"] if i in by_index]
        S.target = target_from_clusters(picked, S.clusters, engine, S.cache, follow=False)
        S.results, S.accepted, S.rejected, S.overlap_regions = {}, [], [], set()

    followed = S.target.info.get("followed", []) if S.target is not None and S.target.source == "자동 발견" else []
    if followed:
        with st.container(key="followed-box"):
            names = ", ".join(f"목소리 {i}" for i in followed[:8]) + (f" 등 {len(followed)}개" if len(followed) > 8 else "")
            html_block('<div class="vm-review-title">같은 사람으로 보이는 목소리도 함께 찾았어요</div>'
                       f'<p class="vm-text">고른 목소리와 확실히 같은 사람으로 보이는 {html.escape(names)}를 함께 썼어요. '
                       '멀리서 녹음했거나 녹음이 길면 한 사람이 여러 목소리로 나뉘어 보이기 때문이에요. '
                       '다른 사람이 섞여 보이면 고른 목소리만으로 다시 찾을 수 있어요.</p>')
            st.button("고른 목소리만으로 찾기", key="only-picked", on_click=only_picked)

target = S.target
if target is None:
    step(3, f"{person}의 목소리, 해마다", "찾는 사람을 고르면 이 자리에 해마다 모은 목소리가 나와요.", "later")
    step(4, "모아서 내려받기", "해마다 모은 음성 파일과, 어느 녹음의 몇 초에서 가져왔는지 적은 목록을 함께 받을 수 있어요.", "later")
    st.stop()

# ── 3 해마다 ─────────────────────────────────────────────
key = (id(target), threshold, tuple(S.accepted), tuple(S.rejected), bool(use_overlap))
if key not in S.results:
    with st.spinner("녹음마다 목소리를 찾고 한 번 더 확인하는 중이에요…"
                    + (" 겹쳐 말한 구간까지 살피느라 처음에는 더 걸려요." if use_overlap else "")):
        # 겹친 구간은 다른 사람 목소리가 섞여 있어 기준 지문이나 경쟁자로 쓰면 안 된다
        acc = [r for r in S.accepted if r not in S.overlap_regions]
        rej = [r for r in S.rejected if r not in S.overlap_regions]
        reviewed = with_review(target, engine, S.cache, records, acc, rej) if acc or rej else target
        found = apply_review(search(records, reviewed, engine, S.cache, threshold,
                                    get_extractor() if use_overlap else None), records, S.accepted, S.rejected)
        S.results[key] = {"clips": found.clips, "maybe": found.maybe, "texts": {}}
res = S.results[key]
clips = res["clips"]
by_path = {r.path: r for r in records}


def region(c) -> tuple[str, float, float]:
    return (c.path, round(c.start, 2), round(c.end, 2))


def decide(answer: str, c) -> None:
    """'맞아요'/'아니에요' 버튼. 반대쪽 목록에 있던 같은 구간은 지운다."""
    r = region(c)
    if c.overlap:
        S.overlap_regions.add(r)
    keep, drop = (S.accepted, S.rejected) if answer == "yes" else (S.rejected, S.accepted)
    if r in drop:
        drop.remove(r)
    if r not in keep:
        keep.append(r)


def clear_review() -> None:
    S.accepted, S.rejected, S.overlap_regions = [], [], set()


def when(c) -> str:
    return f"{c.date.year}년 {c.date.month}월 {c.date.day}일" if c.date else "날짜 모름"


step(3, f"{person}의 목소리, 해마다", None, "done" if clips else "current")

if res["maybe"]:
    with st.container(key="body-review"), st.container(key="review-box"):
        html_block(f'<div class="vm-review-title">확인해 주세요</div>'
                   f'<p class="vm-text">기준에 조금 못 미치지만, 폴더에 나온 다른 누구보다 '
                   f'{html.escape(josa(person, "과", "와"))} 더 닮은 구간이 {len(res["maybe"])}개 있어요. '
                   f'들어 보고 알려 주시면 다음 찾기에도 반영해요.</p>')
        for i, m in enumerate(res["maybe"][:6]):
            note = " · 다른 목소리와 겹친 구간" if m.overlap else ""
            st.markdown(f"{Path(m.path).name}  \n:gray[{when(m)}, {m.start:.1f}초부터 {m.end:.1f}초까지{note}]")
            b, yes, no = st.columns([3.2, 1, 1.2], vertical_alignment="center")
            b.audio(audio.wav_bytes(S.cache.slice(m.path, m.start, m.end)), format="audio/wav")
            yes.button("맞아요", key=f"yes_{i}_{region(m)}", on_click=decide, args=("yes", m),
                       width="stretch")
            no.button("아니에요", key=f"no_{i}_{region(m)}", on_click=decide, args=("no", m),
                      width="stretch")
        if len(res["maybe"]) > 6:
            st.caption(f"나머지 {len(res['maybe']) - 6}개는 위 구간을 확인하면 이어서 보여 드려요.")

def undo(kind: str, r) -> None:
    (S.accepted if kind == "yes" else S.rejected).remove(r)


if S.accepted or S.rejected:
    with st.container(key="body-reviewed"):
        note = f"직접 확인한 구간: 맞아요 {len(S.accepted)}개, 아니에요 {len(S.rejected)}개. 이 판단을 기준에 더해 다시 찾았어요."
        if S.rejected:
            note += " ‘아니에요’로 고른 목소리와 비슷하게 들리는 구간도 함께 뺐어요."
        st.caption(note)
        with st.expander("확인한 구간 보기와 되돌리기"):
            for kind, items in (("yes", S.accepted), ("no", S.rejected)):
                for j, r in enumerate(list(items)):
                    a, bcol, c = st.columns([2, 3, 1.2], vertical_alignment="center")
                    label = "맞아요" if kind == "yes" else "아니에요"
                    a.markdown(f"{Path(r[0]).name}  \n:gray[{r[1]:.1f}초부터 {r[2]:.1f}초까지, ‘{label}’로 확인함]")
                    bcol.audio(audio.wav_bytes(S.cache.slice(*r)), format="audio/wav")
                    c.button("되돌리기", key=f"undo_{kind}_{j}_{r}", on_click=undo, args=(kind, r),
                             width="stretch")
            st.button("확인한 내용 모두 지우기", on_click=clear_review)

if not clips:
    html_block('<p class="vm-lede">이 기준으로는 목소리를 찾지 못했어요. 왼쪽에서 기준을 조금 낮추거나, '
               '다른 목소리를 함께 골라 보세요.</p>')
    st.stop()

years = build_timeline(clips)
total = sum(c.duration for c in clips)
html_block(f'<p class="vm-summary">{len(years)}개 해에서 모두 {mmss(total)}를 찾았어요.</p>')
with st.container(key="body-view"):
    view = st.segmented_control("보는 방법", ["해마다", "날짜마다"], default="해마다",
                                key="view", label_visibility="collapsed") or "해마다"


def render_group(gkey: str, title: str, gclips: list, highlight: bool) -> None:
    """해마다 또는 날짜마다 한 묶음을 그린다."""
    files = sorted({c.path for c in gclips})
    secs = sum(c.duration for c in gclips)
    full = compile_highlight(all_selections(gclips), S.cache)
    main = compile_highlight(select_highlights(gclips), S.cache) if highlight else full
    with st.container(key=f"year-{gkey}"):
        html_block(f'<div class="vm-year{" vm-date" if not highlight else ""}"><b>{title}</b>'
                   f'<span>파일 {len(files)}개에서 {mmss(secs)}를 찾았어요</span></div>'
                   + strip_svg(envelope(main, 110), [(0, 1)], height=56, label=f"{title} 목소리 파형"))
        if highlight:
            html_block(f'<p class="vm-note">가장 또렷한 구간을 모은 {mmss(len(main) / audio.SR)}예요.</p>')
        st.audio(audio.wav_bytes(main), format="audio/wav")
        if highlight and len(full) > len(main) + audio.SR:
            with st.expander(f"이 해에 찾은 목소리 전부 듣기 ({mmss(len(full) / audio.SR)})"):
                st.audio(audio.wav_bytes(full), format="audio/wav")

        if do_transcribe:
            if gkey not in res["texts"]:
                with st.spinner("한 말을 글로 옮기는 중이에요…"):
                    res["texts"][gkey] = get_transcriber().transcribe(main) or ""
            if res["texts"][gkey]:
                html_block(f'<p class="vm-quote">{html.escape(res["texts"][gkey])}</p>'
                           '<p class="vm-note">자동 받아쓰기라 틀린 글자가 있을 수 있어요.</p>')

        files_html = ['<div class="vm-files-label">녹음에서 찾은 자리</div>']
        for path in files:
            rec = by_path[path]
            spans = [(c.start / rec.duration, c.end / rec.duration) for c in gclips if c.path == path]
            files_html.append(
                f'<div class="vm-file"><div class="vm-file-name">{html.escape(rec.name)}</div>'
                f'<div class="vm-file-meta">{date_text(rec.date, rec.date_source)}</div>'
                + strip_svg(envelope(S.cache.get(path), 120), spans, height=30,
                            label=f"{rec.name} 전체 중 찾은 구간 {len(spans)}곳")
                + "</div>")
        html_block("".join(files_html))
        st.download_button("음성 파일 받기", audio.wav_bytes(main),
                           file_name=f"{person}_{gkey}_목소리.wav", mime="audio/wav", key=f"dl_{gkey}")
        with st.expander(f"찾은 구간 {len(gclips)}개 하나씩 듣기"):
            for j, c in enumerate(sorted(gclips, key=lambda c: (c.path, c.start))):
                a, bcol, no = st.columns([2, 3, 1.2], vertical_alignment="center")
                mark = ", 직접 확인함" if c.confirmed else ""
                a.markdown(f"{Path(c.path).name}  \n:gray[{c.start:.1f}초부터 {c.end:.1f}초까지{mark}]")
                bcol.audio(audio.wav_bytes(S.cache.slice(c.path, c.start, c.end)), format="audio/wav")
                no.button("이 사람 아니에요", key=f"rm_{gkey}_{j}_{region(c)}", on_click=decide,
                          args=("no", c), width="stretch")


if view == "해마다":
    for y in years:
        render_group(str(y.year) if y.year else "unknown", str(y.year) if y.year else "날짜 모름", y.clips, True)
else:
    by_date: dict = {}
    for c in clips:
        by_date.setdefault(c.date, []).append(c)
    for d in sorted(by_date, key=lambda d: (d is None, d or 0)):
        render_group(d.isoformat() if d else "unknown",
                     f"{d.year}년 {d.month}월 {d.day}일" if d else "날짜 모름", by_date[d], False)

# ── 4 모아서 내려받기 ────────────────────────────────────
step(4, "모아서 내려받기", None, "done")
all_rows, zip_buf = [], io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    for y in years:
        ylabel = y.year or "날짜모름"
        zf.writestr(f"{person}_{ylabel}_올해의목소리.wav",
                    audio.wav_bytes(compile_highlight(select_highlights(y.clips), S.cache)))
        rows = source_rows(y, all_selections(y.clips), S.get("root"))
        all_rows += rows
        for sel, row in zip(all_selections(y.clips), rows):
            name = f"{row['녹음 날짜'] or '날짜모름'}_{Path(sel.clip.path).stem}_{sel.start:.1f}-{sel.end:.1f}초.wav"
            zf.writestr(f"구간별/{name}", audio.wav_bytes(S.cache.slice(sel.clip.path, sel.start, sel.end)))
    zf.writestr(f"{person}_구간목록.csv", sources_csv(all_rows))

shown = all_rows[:20]
table = "".join(
    f"<tr><td>{r['해']}</td><td>{html.escape(r['녹음 파일'])}</td><td>{r['녹음 날짜'] or '모름'}</td>"
    f"<td>{r['시작(초)']:.1f}~{r['끝(초)']:.1f}초</td></tr>" for r in shown)
more = (f'<p class="vm-note">전체 {len(all_rows)}개 중 20개만 보여 드려요. 나머지는 zip 안의 목록에 있어요.</p>'
        if len(all_rows) > 20 else "")
html_block(
    '<div class="vm-sources"><h3>찾은 구간</h3>'
    '<p>zip에는 해마다 모은 음성, 찾은 구간 하나하나의 음성, 그리고 각 구간이 어느 녹음의 몇 초에서 '
    '왔는지 적은 목록이 들어 있어요. 원본에서 앞뒤를 더 듣고 싶을 때 찾아가기 좋아요.</p>'
    '<table><thead><tr><th>해</th><th>녹음 파일</th><th>녹음 날짜</th><th>구간</th></tr></thead>'
    f'<tbody>{table}</tbody></table>{more}</div>'
)
with st.container(key="download-all"):
    st.download_button("모두 받기 (zip)", zip_buf.getvalue(),
                       file_name=f"{person}_목소리_모음.zip", mime="application/zip", type="primary")
