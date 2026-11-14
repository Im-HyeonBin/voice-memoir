"""겹쳐 말하기 시험 세트(bench/overlap)로 대상 화자 추출을 붙인 앱 전체를 잰다.

사람 10명 폴더마다 앱과 똑같이 스캔 → 목소리 자동 발견 → 참고 녹음 / 자동 발견으로 찾기를 하고 세 가지를 비교한다.
- 기본: 대상 화자 추출 없이 담은 것
- 기본+확인: '확인해 주세요' 목록을 사람이 맞게 답했다고 가정(찾을 사람이 절반 넘게 든 구간만 '맞아요')
- 추출+확인: 대상 화자 추출로 찾은 겹친 구간까지 목록에 올린 뒤 같은 가정
확인 목록에 올라온 개수와 그중 찾을 사람이 아닌 것(헛짚음)도 센다.

    python tools/evaluate_overlap.py --tse models/tse/td_speakerbeam_ko.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import TARGET, Scorer, owner_share  # noqa: E402
from make_overlap_benchmark import FILES, OUT  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import apply_review, discover_speakers, search, target_from_clusters, target_from_reference  # noqa: E402


def target_share(clip, spans) -> float:
    got = sum(max(0.0, min(clip.end, s["end"]) - max(clip.start, s["start"])) for s in spans if s["speaker"] == TARGET)
    return got / clip.duration


class Counting:
    """대상 화자 추출을 감싸 호출 수와 추출한 소리 길이를 센다."""

    def __init__(self, ext):
        self.ext, self.calls, self.sec = ext, 0, 0.0

    def __call__(self, mix, enroll, key=None):
        self.calls += 1
        self.sec += len(mix) / audio.SR
        return self.ext(mix, enroll, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="voxblink2_samresnet34_ko")
    ap.add_argument("--tse", required=True, help="대상 화자 추출 가중치(.pth)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--floors", default="", help="추출 후보로 삼을 닮은 정도 하한들(쉼표). 비우면 모델 기본값 하나")
    args = ap.parse_args()
    from vm.tse import Extractor
    eng = VoiceEngine(ROOT / "models", args.device, args.model)
    ext = Extractor(args.tse, eng.device)
    floors = [float(v) for v in args.floors.split(",")] if args.floors else [eng.cal.candidate_floor]
    truth_all = json.loads((ROOT / "bench" / "overlap_truth.json").read_text())
    tot = defaultdict(lambda: defaultdict(float))
    asked = defaultdict(lambda: [0, 0])  # (모드, 방식) → [올라온 수, 헛짚은 수]
    cost = defaultdict(lambda: [0, 0.0])  # 방식 → [호출 수, 추출한 소리(초)]
    speech_sec = 0.0
    for folder, truth in truth_all.items():
        records = Library(eng, ROOT / ".cache").scan(OUT / folder)
        cache = audio.AudioCache(max_files=8)
        scorer = Scorer(records, truth)
        speech_sec += sum(r.speech_seconds for r in records)
        clusters = discover_speakers(records, eng, cache)
        chosen = [c for c in clusters
                  if (lambda m: m[TARGET] / (sum(m.values()) or 1) > 0.5)(owner_share(records, truth, c, clusters))]
        targets = {"참고": target_from_reference(eng, audio.decode(OUT / f"{folder}_참고.wav"), clusters)}
        if chosen:
            targets["자동"] = target_from_clusters(chosen, clusters, eng, cache)
        for mode, tgt in targets.items():
            variants = [("기본", None, None)] + [(f"추출{f:.2f}", Counting(ext), f) for f in floors]
            for how, extractor, floor in variants:
                res = search(records, tgt, eng, cache, extractor=extractor, overlap_floor=floor)
                if extractor is not None:
                    cost[(mode, how)][0] += extractor.calls
                    cost[(mode, how)][1] += extractor.sec
                shown = apply_review(res, records, [], []).maybe  # 화면에 실제로 올라오는 확인 목록
                acc = [(m.path, m.start, m.end) for m in shown if target_share(m, truth[Path(m.path).name]) > 0.5]
                rej = [(m.path, m.start, m.end) for m in shown if target_share(m, truth[Path(m.path).name]) <= 0.5]
                asked[(mode, how)][0] += len(shown)
                asked[(mode, how)][1] += len(rej)
                runs = {"기본": res} if how == "기본" else {}
                runs[f"{how}+확인" if how != "기본" else "기본+확인"] = apply_review(res, records, acc, rej)
                for name, r in runs.items():
                    s = scorer.score(r.clips)
                    for fname, row in s["per_file"].items():
                        for key in ("target_speech", "found_speech", "wrong"):
                            tot[(mode, name, fname)][key] += row[key]
        print(f"{folder} 끝", file=sys.stderr, flush=True)
    print(f"모델 {args.model}, 대상 화자 추출 {Path(args.tse).name}. 찾은 비율(말소리) / 잘못 담은 시간, 사람 10명 합")
    for mode in ("참고", "자동"):
        for f in floors:
            how = f"추출{f:.2f}"
            print(f"\n### {mode} 방식, 추출 후보 하한 {f:.2f}")
            print(f"{'녹음':12s} {'기본':>16s} {'기본+확인':>16s} {'추출+확인':>16s}")
            for fname in FILES:
                cells = []
                for name in ("기본", "기본+확인", f"{how}+확인"):
                    r = tot[(mode, name, fname)]
                    rate = f"{r['found_speech'] / r['target_speech']:4.0%}" if r["target_speech"] else "   -"
                    cells.append(f"{rate} {r['wrong']:5.1f}초")
                print(f"{fname[:-4]:12s} " + " ".join(f"{c:>16s}" for c in cells))
            n0, bad0 = asked[(mode, "기본")]
            n, bad = asked[(mode, how)]
            calls, sec = cost[(mode, how)]
            print(f"  확인 목록: 기본 {n0}개(헛짚음 {bad0}), 추출 {n}개(헛짚음 {bad}). 추출 호출 {calls}번, "
                  f"추출한 소리 {sec / 60:.1f}분(말소리 {speech_sec / 60:.1f}분의 {sec / max(speech_sec, 1):.0%})")


if __name__ == "__main__":
    main()
