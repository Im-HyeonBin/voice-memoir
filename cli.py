"""명령줄로 전체 과정을 돌린다.

    # 1) 폴더에 나오는 목소리들 보기 (목소리별 들어 볼 샘플을 exports/샘플에 저장)
    python cli.py demo_library

    # 2) 고른 목소리(같은 사람이면 여러 개)로 연도별 '올해의 목소리' 만들기
    python cli.py demo_library --pick 4 --person 아버지

    # 또는 참고 음성으로 지정
    python cli.py demo_library --reference demo_sources/참고음성_본인_7초.wav --person 나
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from vm import audio
from vm.engine import DEFAULT_MODEL, MODELS, VoiceEngine
from vm.library import Library
from vm.speakers import (discover_speakers, find_clips,
                         target_from_clusters, target_from_reference)
from vm.timeline import build_timeline, compile_highlight, export_year, select_highlights, source_rows, sources_csv

ROOT = Path(__file__).resolve().parent


def mmss(sec: float) -> str:
    return f"{int(sec // 60)}분 {sec % 60:04.1f}초" if sec >= 60 else f"{sec:.1f}초"


def main() -> None:
    ap = argparse.ArgumentParser(description="흩어진 녹음에서 한 사람의 목소리를 찾아 해마다 모은다")
    ap.add_argument("folder")
    ap.add_argument("--reference", help="찾을 사람의 참고 녹음")
    ap.add_argument("--pick", help="자동 발견 목소리 번호(같은 사람이면 쉼표로 여러 개)")
    ap.add_argument("--person", default="찾은 사람", help="결과물에 붙일 이름")
    ap.add_argument("--threshold", type=float, help="담는 기준(모델마다 크기가 다름, 기본은 모델의 보통 기준)")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                    help="화자 지문 모델. ecapa는 상업적으로 써도 되는 라이선스")
    ap.add_argument("--out", default=str(ROOT / "exports"))
    ap.add_argument("--transcribe", action="store_true", help="하이라이트 받아쓰기(처음 한 번 모델 내려받음)")
    args = ap.parse_args()

    out = Path(args.out)
    engine = VoiceEngine(ROOT / "models", device="auto", model=args.model)
    t0 = time.time()
    records = Library(engine, ROOT / ".cache").scan(
        args.folder, progress=lambda i, n, name: print(f"\r스캔 {i}/{n} {name[:40]:40s}", end="", flush=True))
    print(f"\r스캔 완료: 파일 {len(records)}개, 녹음 {mmss(sum(r.duration for r in records))} ({time.time() - t0:.1f}초)")
    for r in records:
        tag = " · 전화 음질" if r.narrowband else ""
        err = f" · 읽기 실패: {r.error}" if r.error else ""
        print(f"  {r.name:44s} {r.date or '날짜 모름'} ({r.date_label}){tag}{err}")

    cache = audio.AudioCache()
    clusters = discover_speakers(records, engine, cache)  # 찾을 사람 후보이자, 비교할 경쟁자
    if args.reference:
        target = target_from_reference(engine, audio.decode(args.reference), clusters)
        if target.info["mixed"]:
            print("참고 녹음에 다른 목소리도 섞여 있어, 가장 많이 말한 사람을 기준으로 삼았어요.")
    else:
        if not args.pick:
            sample_dir = out / "샘플"
            sample_dir.mkdir(parents=True, exist_ok=True)
            print("\n이 폴더에 나오는 목소리들 (말한 시간 순):")
            for c in clusters:
                p, s, e = c.sample
                f = sample_dir / f"목소리{c.index}.wav"
                audio.write_wav(f, cache.slice(p, s, e))
                linked = " · 통화 녹음 포함" if c.phone_linked else ""
                print(f"  목소리 {c.index}: 약 {mmss(c.seconds)}, 파일 {len(c.files)}개{linked} → 샘플 {f.name}")
            print(f"\n샘플을 들어 보고 --pick 번호로 고르세요. 샘플 위치: {sample_dir}")
            return
        chosen = [c for c in clusters if c.index in {int(x) for x in args.pick.split(",")}]
        target = target_from_clusters(chosen, clusters, engine, cache)

    clips = find_clips(records, target, engine, cache, args.threshold)
    years = build_timeline(clips)
    transcriber = None
    if args.transcribe:
        from vm.transcribe import Transcriber
        transcriber = Transcriber()

    print(f"\n{args.person}의 목소리: 총 {mmss(sum(c.duration for c in clips))}, {len(years)}개 해")
    rows = []
    for y in years:
        sel = select_highlights(y.clips)
        paths = export_year(y, sel, cache, args.person, out)
        rows += source_rows(y, sel, args.folder)
        print(f"\n■ {y.label}: 파일 {len(y.files)}개에서 {mmss(y.seconds)}")
        for s in sel:
            print(f"  · {Path(s.clip.path).name} {s.start:5.1f}~{s.end:5.1f}초")
        if transcriber:
            print(f"  받아쓰기(자동이라 틀릴 수 있음): {transcriber.transcribe(compile_highlight(sel, cache))}")
        print(f"  → {paths['m4a'].name}")
    if rows:
        csv_path = out / f"{args.person}_구간목록.csv"
        csv_path.write_bytes(sources_csv(rows))
        print(f"\n가져온 구간 목록: {csv_path.name}")

if __name__ == "__main__":
    main()
