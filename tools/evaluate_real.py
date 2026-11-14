"""실제 녹음으로 정확도를 잰다. 합성 시험 세트가 아니라 가족 영상·통화 녹음처럼 진짜 녹음에 쓴다.

준비: 녹음마다 같은 이름의 정답 파일을 옆에 둔다. 둘 중 하나면 된다.
- Audacity 레이블(.txt): 녹음을 Audacity로 열고 찾을 사람이 말하는 곳을 끌어 잡은 뒤 Ctrl+B로 레이블을 달고
  '파일 → 내보내기 → 레이블 내보내기'. 한 줄에 '시작<탭>끝<탭>이름'. --label로 고른 이름(기본: 모든 레이블)만 찾을 사람으로 본다.
- RTTM(.rttm): 화자 분리 연구에서 쓰는 형식. --speaker로 찾을 사람 이름을 준다.
찾을 사람이 없는 녹음은 빈 정답 파일을 둔다(함정 녹음). 정답 파일이 없는 녹음은 채점하지 않는다.

    python tools/evaluate_real.py ~/가족녹음 --reference ~/아버지_참고.m4a
    python tools/evaluate_real.py ~/가족녹음 --auto            # 자동 발견: 정답과 가장 많이 겹치는 목소리를 고른다

채점은 10ms 단위, 정답 경계 앞뒤 0.25초는 빼고 센다(사람이 경계를 정확히 찍기 어려워서, 화자 분리 평가의 관례).
- 찾은 비율: 찾을 사람이 말한 시간 중 담은 시간
- 잘못 담은 시간: 찾을 사람이 말하지 않은 곳(다른 사람·쉼·잡음)을 담은 시간
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FPS = 100
COLLAR = 0.25


def read_labels(path: Path, label: str | None = None, speaker: str | None = None) -> list[tuple[float, float]]:
    """정답 파일에서 찾을 사람 구간을 읽는다. 빈 파일이면 빈 목록(찾을 사람이 없는 녹음)."""
    spans = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts or line.startswith(("\\", "#")):  # Audacity 주파수 줄('\t...')과 주석은 건너뛴다
            continue
        if path.suffix == ".rttm":
            if parts[0] != "SPEAKER" or (speaker and parts[7] != speaker):
                continue
            s, d = float(parts[3]), float(parts[4])
            spans.append((s, s + d))
        else:
            s, e = float(parts[0]), float(parts[1])
            name = " ".join(parts[2:])
            if label is None or name == label:
                spans.append((s, e))
    return sorted(spans)


def truth_file(audio: Path) -> Path | None:
    for ext in (".rttm", ".txt"):
        p = audio.with_suffix(ext)
        if p.exists():
            return p
    return None


def score(spans, picked, duration: float, collar: float = COLLAR) -> dict:
    n = int(duration * FPS) + 1
    truth = np.zeros(n, bool)
    got = np.zeros(n, bool)
    care = np.ones(n, bool)
    for s, e in spans:
        truth[int(s * FPS):int(e * FPS)] = True
        for b in (s, e):
            care[max(0, int((b - collar) * FPS)):int((b + collar) * FPS)] = False
    for s, e in picked:
        got[int(s * FPS):int(e * FPS)] = True
    return {"target": (truth & care).sum() / FPS, "found": (truth & got & care).sum() / FPS,
            "wrong": (~truth & got & care).sum() / FPS}


def main() -> None:
    from vm import audio
    from vm.engine import DEFAULT_MODEL, VoiceEngine
    from vm.library import Library
    from vm.speakers import discover_speakers, search, target_from_clusters, target_from_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--reference", help="찾을 사람 참고 녹음")
    ap.add_argument("--auto", action="store_true", help="자동 발견에서 정답과 가장 많이 겹치는 목소리를 고른다")
    ap.add_argument("--label", help="Audacity 레이블 중 찾을 사람 이름(기본: 모든 레이블)")
    ap.add_argument("--speaker", help="RTTM의 찾을 사람 이름")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    if not (args.reference or args.auto):
        ap.error("--reference 또는 --auto 중 하나를 고르세요")

    eng = VoiceEngine(ROOT / "models", "auto", args.model)
    records = [r for r in Library(eng, ROOT / ".cache").scan(args.folder) if not r.error]
    truth = {r.path: read_labels(t, args.label, args.speaker) for r in records if (t := truth_file(Path(r.path)))}
    if not truth:
        sys.exit("정답 파일(.txt 또는 .rttm)이 있는 녹음이 없어요. 맨 위 설명을 보세요.")
    cache = audio.AudioCache(max_files=8)
    clusters = discover_speakers(records, eng, cache)
    if args.reference:
        target = target_from_reference(eng, audio.decode(args.reference), clusters)
    else:
        def overlap_with_truth(c):
            return sum(min(e, te) - max(s, ts) for p, s, e in c.members if p in truth
                       for ts, te in truth[p] if min(e, te) > max(s, ts))
        best = max(clusters, key=overlap_with_truth)
        target = target_from_clusters([best], clusters, eng, cache)
        print(f"자동 발견: 목소리 {best.index}를 골랐어요(약 {best.seconds:.0f}초)")

    by_path = {r.path: r for r in records}
    print(f"모델 {eng.model}, 정답이 있는 녹음 {len(truth)}개, 경계 앞뒤 {COLLAR}초 제외\n")
    print(f"{'기준':>6s} {'찾은 비율':>8s} {'잘못 담은 시간':>12s}")
    for thr in eng.cal.levels:
        res = search(records, target, eng, cache, thr)
        tot = {"target": 0.0, "found": 0.0, "wrong": 0.0}
        for path, spans in truth.items():
            row = score(spans, [(c.start, c.end) for c in res.clips if c.path == path], by_path[path].duration)
            for k in tot:
                tot[k] += row[k]
        mark = "  ← 기본" if thr == eng.cal.threshold else ""
        rate = f"{tot['found'] / tot['target']:.0%}" if tot["target"] else "-"
        print(f"{thr:6.3f} {rate:>8s} {tot['wrong']:10.1f}초{mark}")

    res = search(records, target, eng, cache)
    print("\n녹음별(기본 기준)")
    for path, spans in sorted(truth.items()):
        row = score(spans, [(c.start, c.end) for c in res.clips if c.path == path], by_path[path].duration)
        rate = f"{row['found'] / row['target']:.0%}" if row["target"] else "  -"
        print(f"  {Path(path).name:40s} 찾을 사람 {row['target']:6.1f}초 중 {rate:>4s} · 잘못 담음 {row['wrong']:5.1f}초")


if __name__ == "__main__":
    main()
