"""실제 녹음 검증: AMI 회의 말뭉치로 '여러 녹음에 흩어진 한 사람 목소리 찾기'를 잰다.

합성 시험 세트와 달리 실제 대화(말 끊기, 맞장구, 겹쳐 말하기)를 방 가운데 마이크 하나로 녹음한 것이다.
같은 네 사람이 다른 날 네 번 회의한 모임 하나를 '녹음 폴더' 하나로 보고, 네 사람을 한 명씩 찾을 사람으로 둔다.
- 참고 녹음 방식: 첫 회의(a)에서 그 사람만 말한 구간을 10초 남짓 잘라 참고 녹음으로 쓴다
- 자동 발견 방식: 폴더 목소리 중 그 사람 말이 절반 넘게 든 것을 모두 고른다(사람이 들어 보고 고르는 상황 흉내)
채점은 참고 녹음을 자른 첫 회의를 빼고 나머지 세 회의로만 한다. 정답 경계 앞뒤 0.25초는 빼고 센다.

    python tools/fetch_data.py ami
    python tools/evaluate_ami.py [--model ecapa_ko] [--groups ES2004 IS1009]
영어 회의라 한국어 학습의 효과가 아니라 '실제 녹음 조건'에서 앱이 어떻게 동작하는지를 본다.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate_real import read_labels, score  # noqa: E402
from fetch_data import AMI_GROUPS  # noqa: E402

REF_SEC = 10.0


def speakers_of(rttm: Path) -> list[str]:
    return sorted({ln.split()[7] for ln in rttm.read_text().splitlines() if ln.startswith("SPEAKER")})


def solo_spans(rttm: Path, who: str, min_len: float = 1.5) -> list[tuple[float, float]]:
    """그 사람만 말한 구간(다른 사람과 겹치지 않는 부분)."""
    mine = read_labels(rttm, speaker=who)
    others = [s for sp in speakers_of(rttm) if sp != who for s in read_labels(rttm, speaker=sp)]
    out = []
    for s, e in mine:
        cuts = sorted((max(s, os_), min(e, oe)) for os_, oe in others if oe > s and os_ < e)
        t = s
        for cs, ce in cuts:
            if cs - t >= min_len:
                out.append((t, cs))
            t = max(t, ce)
        if e - t >= min_len:
            out.append((t, e))
    return out


def reference_clip(wav: np.ndarray, spans, sr: int) -> np.ndarray:
    """긴 구간부터 한 구간에 최대 5초씩, 모두 REF_SEC 남짓 이어 붙인다."""
    pieces, total = [], 0.0
    for s, e in sorted(spans, key=lambda x: x[0] - x[1]):
        take = min(e - s, 5.0)
        mid = (s + e) / 2
        pieces.append(wav[int((mid - take / 2) * sr):int((mid + take / 2) * sr)])
        total += take
        if total >= REF_SEC:
            break
    return np.concatenate(pieces)


def main() -> None:
    from vm import audio
    from vm.engine import DEFAULT_MODEL, VoiceEngine
    from vm.library import Library
    from vm.speakers import discover_speakers, search, target_from_clusters, target_from_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--groups", nargs="+", default=list(AMI_GROUPS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ref-sec", type=float, default=10.0, help="참고 녹음 길이(초)")
    ap.add_argument("--data", default=str(ROOT / "data" / "ami"), help="모임별 폴더(wav·rttm)가 있는 곳")
    ap.add_argument("--only-ref", action="store_true", help="참고 녹음 방식만 잰다(실험용)")
    ap.add_argument("--all-levels", action="store_true", help="다섯 단계 모두 재서 찾은 비율과 다른 사람 비율을 함께 보인다")
    args = ap.parse_args()
    global REF_SEC
    REF_SEC = args.ref_sec
    eng = VoiceEngine(ROOT / "models", args.device, args.model)
    tot = defaultdict(lambda: defaultdict(float))
    rows = []
    n_clusters = {}
    followed = {}
    by_level = defaultdict(lambda: defaultdict(float))
    for g in args.groups:
        folder = Path(args.data).resolve() / g
        records = Library(eng, ROOT / ".cache").scan(folder)
        cache = audio.AudioCache(max_files=4)
        clusters = discover_speakers(records, eng, cache)
        n_clusters[g] = len(clusters)
        meetings = sorted(records, key=lambda r: r.path)
        first, rest = meetings[0], meetings[1:]
        for who in speakers_of(Path(first.path).with_suffix(".rttm")):
            truth = {r.path: read_labels(Path(r.path).with_suffix(".rttm"), speaker=who) for r in rest}
            ref = reference_clip(cache.get(first.path), solo_spans(Path(first.path).with_suffix(".rttm"), who), audio.SR)
            targets = {"참고": target_from_reference(eng, ref, clusters)}
            # 참고 + 확인: '같은 사람인가요?'로 물은 목소리에 사람이 맞게 답했다고 가정
            unsure = targets["참고"].info["unsure"]
            # 자동 발견: 그 사람 말이 절반 넘게 든 목소리를 고른다(모든 회의 기준)
            truth_all = {r.path: read_labels(Path(r.path).with_suffix(".rttm"), speaker=who) for r in meetings}
            chosen = []
            for c in clusters:
                hit = sum(max(0.0, min(e, te) - max(s, ts)) for p, s, e in c.members for ts, te in truth_all[p])
                if hit > 0.5 * sum(e - s for _, s, e in c.members):
                    chosen.append(c)
            if args.only_ref:
                chosen = []
            same = {c.index for c in chosen if c.index in unsure}
            targets["참고+확인"] = target_from_reference(eng, ref, clusters, same=same)
            asked = len(unsure)
            if chosen:
                targets["자동"] = target_from_clusters(chosen, clusters, eng, cache)
                # 카드 하나만 고름: 그 사람 말이 가장 많이 든 목소리 하나만 고르고 나머지는 앱이 따라간다
                one = max(chosen, key=lambda c: c.seconds)
                targets["자동1"] = target_from_clusters([one], clusters, eng, cache)
                followed[(g, who)] = len(targets["자동1"].info.get("followed", []))
            for mode, tgt in targets.items():
                for lv, thr in enumerate(eng.cal.levels if args.all_levels else []):
                    res = search(records, tgt, eng, cache, thr)
                    for r in rest:
                        sc = score(truth[r.path], [(c.start, c.end) for c in res.clips if c.path == r.path], r.duration)
                        for k, v in sc.items():
                            by_level[(mode, lv)][k] += v
                res = search(records, tgt, eng, cache)
                row = defaultdict(float)
                for r in rest:
                    sc = score(truth[r.path], [(c.start, c.end) for c in res.clips if c.path == r.path], r.duration)
                    for k, v in sc.items():
                        row[k] += v
                for k, v in row.items():
                    tot[mode][k] += v
                tot[mode]["people"] += 1
                rows.append((g, who, mode, row["target"], row["found"], row["wrong"], asked if mode == "참고+확인" else len(chosen)))
            print(f"{g} {who} 끝", file=sys.stderr, flush=True)

    hours = sum(r[3] for r in rows if r[2] == "참고") / 3600
    print(f"모델 {eng.model}, AMI 회의 {len(args.groups)}개 모임 × 4회, 찾을 사람 {len(rows) // 3}명. "
          f"채점은 각 모임의 둘째~넷째 회의(찾을 사람 말소리 합 {hours:.1f}시간), 경계 앞뒤 0.25초 제외\n")
    print(f"목소리 묶음 수: {', '.join(f'{g} {n}개' for g, n in n_clusters.items())}")
    print(f"{'모임':8s} {'사람':8s} {'방식':4s} {'말한 시간':>9s} {'찾은 비율':>8s} {'잘못 담은 시간':>12s} {'담은 것 중 다른 사람':>14s}")
    for g, who, mode, t, f, w, n in rows:
        share = f"{w / (f + w):.1%}" if f + w else "-"
        print(f"{g:8s} {who:8s} {mode:4s} {t / 60:7.1f}분 {f / t if t else 0:8.0%} {w:10.1f}초 {share:>14s}"
              + (f"  (목소리 {n}개 고름)" if mode == "자동" else f"  (물은 목소리 {n}개)" if mode == "참고+확인"
                 else f"  (1개 고르고 {followed.get((g, who), 0)}개 따라감)" if mode == "자동1" else ""))
    print()
    for mode, d in tot.items():
        share = d["wrong"] / (d["found"] + d["wrong"]) if d["found"] + d["wrong"] else 0
        print(f"합계 {mode}: {int(d['people'])}명, 찾은 비율 {d['found'] / d['target']:.0%}, "
              f"잘못 담은 시간 {d['wrong'] / 60:.1f}분, 담은 것 중 다른 사람 {share:.1%}")
    if by_level:
        print("\n단계별(0 넉넉하게 … 4 엄격하게): 찾은 비율 / 담은 것 중 다른 사람")
        for mode in tot:
            cells = []
            for lv, thr in enumerate(eng.cal.levels):
                d = by_level[(mode, lv)]
                share = d["wrong"] / (d["found"] + d["wrong"]) if d["found"] + d["wrong"] else 0
                cells.append(f"{thr:.3f} {d['found'] / d['target']:.0%}/{share:.1%}")
            print(f"  {mode:6s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
