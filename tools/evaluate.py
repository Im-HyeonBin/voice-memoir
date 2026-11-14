"""정답 파일로 찾기 정확도를 잰다.

    python tools/evaluate.py                       # 데모 폴더
    python tools/evaluate.py --set hard            # 어려운 시험 세트
    python tools/evaluate.py --set hard --model ecapa

채점은 10ms 단위로 한다(겹쳐 말하는 구간이 있어서).
- 찾은 비율: 찾을 사람이 실제로 말한 시간 중 찾아낸 시간. '말소리만'은 문장 사이 쉼을 뺀 값
- 잘못 담은 시간: 찾을 사람이 말하지 않는데 다른 사람·TV·잡음을 담은 시간
- 자동 발견은 사람이 들어 보고 고르는 상황을 흉내 내어, 찾을 사람이 절반 넘게 섞인 목소리를 모두 고른다
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vm import audio  # noqa: E402
from vm.engine import BENCH_MODELS, DEFAULT_MODEL, HOP, MODELS, VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import (discover_speakers, find_clips,  # noqa: E402
                         target_from_clusters, target_from_reference)

TARGET = "찾을사람"
FPS = 100
SETS = {
    "demo": (ROOT / "demo_library", ROOT / "demo_sources" / "ground_truth.json"),
    "hard": (ROOT / "bench" / "hard", ROOT / "bench" / "hard_truth.json"),
}


def frames(n: int, intervals) -> np.ndarray:
    m = np.zeros(n, bool)
    for s, e in intervals:
        m[max(0, int(s * FPS)):max(0, int(e * FPS))] = True
    return m


class Scorer:
    def __init__(self, records, truth):
        self.files = {}
        for r in records:
            spans = truth[r.name]
            n = int(max([r.duration] + [sp["end"] for sp in spans]) * FPS) + 1
            self.files[r.path] = {
                "name": r.name,
                "target": frames(n, [(sp["start"], sp["end"]) for sp in spans if sp["speaker"] == TARGET]),
                "others": {who: frames(n, [(sp["start"], sp["end"]) for sp in spans if sp["speaker"] == who])
                           for who in {sp["speaker"] for sp in spans} - {TARGET}},
                "speech": frames(n, r.windows),
            }

    def score(self, clips) -> dict:
        by_file = defaultdict(list)
        for c in clips:
            by_file[c.path].append((c.start, c.end))
        tot = defaultdict(float)
        per_file = {}
        wrong = defaultdict(float)
        for path, f in self.files.items():
            got = frames(len(f["target"]), by_file.get(path, []))
            other_any = np.zeros_like(got)
            for who, m in f["others"].items():
                wrong[who] += (got & m & ~f["target"]).sum() / FPS
                other_any |= m
            row = {
                "target": f["target"].sum() / FPS,
                "found": (got & f["target"]).sum() / FPS,
                "target_speech": (f["target"] & f["speech"]).sum() / FPS,
                "found_speech": (got & f["target"] & f["speech"]).sum() / FPS,
                "wrong": (got & other_any & ~f["target"]).sum() / FPS,
            }
            per_file[f["name"]] = row
            for k, v in row.items():
                tot[k] += v
        return {
            "found": tot["found"] / tot["target"] if tot["target"] else 0.0,
            "found_speech": tot["found_speech"] / tot["target_speech"] if tot["target_speech"] else 0.0,
            "wrong_sec": tot["wrong"],
            "wrong_by": dict(wrong),
            "per_file": per_file,
        }


def owner_share(records, truth, cluster, clusters) -> dict:
    """군집에 묶인 조각들이 실제로 누구였는지(초)."""
    mix = defaultdict(float)
    for r in records:
        if len(r.embeddings) == 0:
            continue
        protos = np.stack([k.narrow if r.narrowband else k.centroid for k in clusters])
        spans = truth[r.name]
        for wi in np.where((r.embeddings @ protos.T).argmax(1) == cluster.index - 1)[0]:
            s, e = r.windows[wi]
            best, who = 0.0, "쉼"
            for sp in spans:
                o = max(0.0, min(e, sp["end"]) - max(s, sp["start"]))
                if sp["speaker"] == TARGET:
                    o *= 1.01  # 겹친 구간에서는 찾을 사람 쪽으로 센다
                if o > best:
                    best, who = o, sp["speaker"]
            mix[who] += HOP
    return mix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="demo")
    ap.add_argument("--model", choices=[*MODELS, *BENCH_MODELS], default=DEFAULT_MODEL)
    ap.add_argument("--thresholds", help="쉼표로 구분. 기본은 모델의 다섯 단계(넉넉하게…엄격하게)")
    args = ap.parse_args()
    library, truth_path = SETS[args.set]
    truth = json.loads(truth_path.read_text())

    t0 = time.time()
    engine = VoiceEngine(ROOT / "models", device="auto", model=args.model)
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else list(engine.cal.levels)
    default = engine.cal.threshold
    records = Library(engine, ROOT / ".cache").scan(library)
    cache = audio.AudioCache(max_files=16)
    scorer = Scorer(records, truth)
    t1 = time.time()

    print(f"모델: {engine.model}\n")
    print(f"== [{args.set}] 자동 발견: 목소리마다 실제로 섞인 사람(초) ==")
    clusters = discover_speakers(records, engine, cache)
    t2 = time.time()
    chosen = []
    for c in clusters:
        mix = owner_share(records, truth, c, clusters)
        total = sum(mix.values()) or 1.0
        top = sorted(mix.items(), key=lambda kv: -kv[1])
        mark = ""
        if mix[TARGET] / total > 0.5:
            chosen.append(c)
            mark = "  ← 고름"
        tag = " (통화 녹음 연결)" if c.phone_linked else ""
        print(f"  목소리 {c.index}: 약 {c.seconds:.0f}초, 파일 {len(c.files)}개{tag} → "
              + ", ".join(f"{k} {v:.0f}" for k, v in top[:4]) + mark)
    print(f"  찾을 사람이 나뉜 목소리 수: {len(chosen)}")

    # 앱과 똑같이 폴더의 다른 목소리들을 경쟁자로 둔다
    ref_wav = audio.decode(ROOT / "demo_sources" / "참고음성_본인_7초.wav")
    targets = {"참고 녹음 7초": target_from_reference(engine, ref_wav, clusters)}
    if chosen:
        targets[f"자동 발견({len(chosen)}개 고름)"] = target_from_clusters(chosen, clusters, engine, cache)

    print("\n== 기준별 정확도 ==")
    print(f"{'방식':18s} {'기준':>5s} {'찾은 비율':>8s} {'(말소리만)':>9s} {'잘못 담은 시간':>12s}  잘못 담은 상대")
    for name, tgt in targets.items():
        for thr in thresholds:
            s = scorer.score(find_clips(records, tgt, engine, cache, thr))
            wb = ", ".join(f"{k} {v:.1f}초" for k, v in sorted(s["wrong_by"].items(), key=lambda kv: -kv[1]) if v > 0.05)
            print(f"{name:18s} {thr:5.2f} {s['found']:8.0%} {s['found_speech']:9.0%} {s['wrong_sec']:10.1f}초  {wb}")
        print()

    print(f"== 파일별 (기본 기준 {default}) ==")
    for name, tgt in targets.items():
        s = scorer.score(find_clips(records, tgt, engine, cache, default))
        print(f"[{name}]")
        for fname, row in s["per_file"].items():
            rate = f"{row['found_speech'] / row['target_speech']:.0%}" if row["target_speech"] else "  -"
            print(f"  {fname:34s} 찾을 사람 말소리 {row['target_speech']:5.1f}초 중 {rate:>4s} · 잘못 담음 {row['wrong']:4.1f}초")
    t3 = time.time()
    print(f"\n시간: 스캔 {t1 - t0:.1f}s · 자동 발견 {t2 - t1:.1f}s · 평가 {t3 - t2:.1f}s")


if __name__ == "__main__":
    main()
