"""사람이 '확인해 볼 구간'과 잘못 담긴 구간을 한 번 확인해 주면 얼마나 나아지는지 흉내 낸다.

정답 기준으로 확인 후보 중 찾을 사람이 맞는 것은 '맞아요', 아닌 것은 '아니에요'를 누르고,
담긴 구간 중 대부분이 다른 사람인 것도 '아니에요'를 누른다고 가정한다.

    python tools/review_sim.py --set hard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import SETS, TARGET, Scorer, owner_share  # noqa: E402
from vm import audio  # noqa: E402
from vm.engine import DEFAULT_MODEL, MODELS, VoiceEngine  # noqa: E402
from vm.library import Library  # noqa: E402
from vm.speakers import (apply_review, discover_speakers, search, target_from_clusters,  # noqa: E402
                         target_from_reference, with_review)


def target_share(clip, truth) -> float:
    spans = [s for s in truth[Path(clip.path).name] if s["speaker"] == TARGET]
    got = sum(max(0.0, min(clip.end, s["end"]) - max(clip.start, s["start"])) for s in spans)
    return got / clip.duration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="hard")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    args = ap.parse_args()
    library, truth_path = SETS[args.set]
    truth = json.loads(truth_path.read_text())
    engine = VoiceEngine(ROOT / "models", device="auto", model=args.model)
    cache = audio.AudioCache(max_files=16)
    records = Library(engine, ROOT / ".cache").scan(library)
    scorer = Scorer(records, truth)
    clusters = discover_speakers(records, engine, cache)
    chosen = [c for c in clusters
              if (lambda m: m[TARGET] / (sum(m.values()) or 1) > 0.5)(owner_share(records, truth, c, clusters))]
    targets = {"참고 녹음": target_from_reference(engine, audio.decode(ROOT / "demo_sources" / "참고음성_본인_7초.wav"), clusters),
               "자동 발견": target_from_clusters(chosen, clusters, engine, cache)}
    for name, target in targets.items():
        res = search(records, target, engine, cache)
        before = scorer.score(res.clips)
        accepted = [(m.path, m.start, m.end) for m in res.maybe if target_share(m, truth) > 0.5]
        rejected = [(m.path, m.start, m.end) for m in res.maybe if target_share(m, truth) <= 0.5]
        rejected += [(c.path, c.start, c.end) for c in res.clips if target_share(c, truth) < 0.5]
        reviewed = with_review(target, engine, cache, records, accepted, rejected)
        after_res = apply_review(search(records, reviewed, engine, cache), records, accepted, rejected)
        after = scorer.score(after_res.clips)
        print(f"[{args.set} · {name}] 확인 {len(accepted) + len(rejected)}번 (맞아요 {len(accepted)}, 아니에요 {len(rejected)})")
        print(f"   확인 전: 찾은 비율(말소리) {before['found_speech']:.0%}, 잘못 담음 {before['wrong_sec']:.1f}초")
        print(f"   확인 후: 찾은 비율(말소리) {after['found_speech']:.0%}, 잘못 담음 {after['wrong_sec']:.1f}초, "
              f"새로 확인해 볼 구간 {len(after_res.maybe)}개")


if __name__ == "__main__":
    main()
