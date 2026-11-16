"""한국어 실제 녹음 검증: MSDWild(유튜브 실제 영상, 누가 언제 말했는지 정답 RTTM)에서 한국어 영상만 골라 잰다.

MSDWild는 X-LANCE 연구실(상해교통대)의 연구용 자료다. 동의서 조건(연구용, 재배포 금지, 인용)을 따른다.
소리·정답은 저장소에 넣지 않고, 결과는 합계만 싣는다. 출처: Liu et al., "MSDWild: Multi-modal Speaker Diarization
Dataset in the Wild", Interspeech 2022, https://github.com/X-LANCE/MSDWILD

    python tools/evaluate_msdwild.py            # data/msdwild/wav.zip, data/msdwild/rttms/all.rttm 필요

1. 압축을 풀고(처음 한 번), 영상마다 앞 30초로 언어를 가린다(faster-whisper, 결과는 캐시). 한국어(확률 0.8 이상)만 쓴다.
2. 한국어 영상마다 20초 넘게 말하고 혼자 말한 구간이 8초 넘는 사람을 한 명씩 찾을 사람으로 둔다.
   - 참고 녹음 방식: 그 사람이 혼자 말한 구간을 앞에서부터 한 구간에 5초까지, 10초 남짓 잘라 참고 녹음으로 쓴다
   - 자동 발견(카드 1개): 그 사람 말이 절반 넘게 든 목소리 중 가장 길게 말한 것 하나만 고른다(나머지는 앱이 따라감)
3. 영상 하나를 녹음 폴더 하나로 본다. 참고 녹음으로 자른 구간은 채점에서 빼고, 정답 경계 앞뒤 0.25초도 뺀다.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "msdwild"
FPS = 100
COLLAR = 0.25
REF_SEC = 10.0


def read_rttm(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    out: dict = defaultdict(lambda: defaultdict(list))
    for ln in path.read_text().splitlines():
        p = ln.split()
        if p and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            out[p[1]][p[7]].append((s, s + d))
    return out


def solo(spans: dict[str, list], who: str, min_len: float = 1.5) -> list[tuple[float, float]]:
    others = [x for sp, v in spans.items() if sp != who for x in v]
    out = []
    for s, e in sorted(spans[who]):
        t = s
        for os_, oe in sorted(x for x in others if x[1] > s and x[0] < e):
            if os_ - t >= min_len:
                out.append((t, os_))
            t = max(t, oe)
        if e - t >= min_len:
            out.append((t, e))
    return out


def reference(wav: np.ndarray, solo_spans, sr: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    pieces, used, total = [], [], 0.0
    for s, e in solo_spans:  # 앞에서부터
        take = min(e - s, 5.0)
        pieces.append(wav[int(s * sr):int((s + take) * sr)])
        used.append((s, s + take))
        total += take
        if total >= REF_SEC:
            break
    return np.concatenate(pieces), used


def score(truth, picked, excluded, duration: float) -> dict:
    n = int(duration * FPS) + 1
    tr, got, care = np.zeros(n, bool), np.zeros(n, bool), np.ones(n, bool)
    for s, e in truth:
        tr[int(s * FPS):int(e * FPS)] = True
        for b in (s, e):
            care[max(0, int((b - COLLAR) * FPS)):int((b + COLLAR) * FPS)] = False
    for s, e in excluded:
        care[int(s * FPS):int(e * FPS)] = False
    for s, e in picked:
        got[int(s * FPS):int(e * FPS)] = True
    return {"target": (tr & care).sum() / FPS, "found": (tr & got & care).sum() / FPS,
            "wrong": (~tr & got & care).sum() / FPS}


def languages(wav_dir: Path, cache: Path) -> dict:
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted(p for p in wav_dir.glob("*.wav") if p.stem not in done)
    if todo:
        from faster_whisper import WhisperModel
        from vm import audio
        try:
            model = WhisperModel("small", device="cuda", compute_type="int8_float32")
            model.detect_language(np.zeros(audio.SR, np.float32))  # CUDA 라이브러리 문제는 첫 추론에서 드러난다
        except Exception:  # GPU에서 못 돌리면 CPU로
            model = WhisperModel("small", device="cpu", compute_type="int8")
        for i, p in enumerate(todo, 1):
            x = audio.decode(p)[: 30 * audio.SR]
            lang, prob, _ = model.detect_language(x)
            done[p.stem] = [lang, round(float(prob), 3)]
            if i % 200 == 0:
                cache.write_text(json.dumps(done))
                print(f"언어 {i}/{len(todo)}", file=sys.stderr, flush=True)
        cache.write_text(json.dumps(done))
    return done


def main() -> None:
    from vm import audio
    from vm.engine import DEFAULT_MODEL, VoiceEngine
    from vm.library import Library
    from vm.speakers import discover_speakers, search, target_from_clusters, target_from_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--min-prob", type=float, default=0.8)
    ap.add_argument("--all-levels", action="store_true", help="다섯 단계 모두 잰다")
    ap.add_argument("--rows", action="store_true", help="사람별 결과를 보인다")
    ap.add_argument("--data", default=str(DATA), help="wav.zip(또는 풀린 wav/)와 rttms/all.rttm이 있는 곳")
    args = ap.parse_args()
    data = Path(args.data).resolve()  # 임시 폴더에 거는 링크가 깨지지 않게 절대 경로로
    wav_dir = data / "wav"
    if not wav_dir.exists():
        with zipfile.ZipFile(data / "wav.zip") as z:
            z.extractall(data)
    langs = languages(wav_dir, ROOT / "bench" / "msdwild_lang.json")
    clips = sorted(c for c, (lg, pr) in langs.items() if lg == args.lang and pr >= args.min_prob
                   and (wav_dir / f"{c}.wav").exists())
    rttm = read_rttm(data / "rttms" / "all.rttm")
    print(f"영상 {len(langs)}개 중 {args.lang}(확률 {args.min_prob} 이상) {len(clips)}개", flush=True)

    eng = VoiceEngine(ROOT / "models", "auto", args.model)
    tot = defaultdict(lambda: defaultdict(float))
    per_person = defaultdict(list)
    by_level = defaultdict(lambda: defaultdict(float))
    rows = []
    n_people = n_clips = 0
    for clip in clips:
        spans = rttm.get(clip)
        if not spans:
            continue
        people = [w for w in spans if sum(e - s for s, e in spans[w]) >= 20
                  and sum(e - s for s, e in solo(spans, w)) >= 8]
        if not people:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / f"{clip}.wav"
            link.symlink_to(wav_dir / f"{clip}.wav")
            recs = Library(eng, ROOT / ".cache").scan(tmp)
            if not recs or recs[0].error or len(recs[0].windows) == 0:
                continue
            rec = recs[0]
            cache = audio.AudioCache(max_files=2)
            clusters = discover_speakers([rec], eng, cache)
            if not clusters:
                continue
            n_clips += 1
            wav = cache.get(rec.path)
            for who in people:
                ref, used = reference(wav, solo(spans, who), audio.SR)
                targets = {"참고": target_from_reference(eng, ref, clusters)}
                mine = spans[who]
                chosen = [c for c in clusters
                          if sum(max(0.0, min(e, te) - max(s, ts)) for _, s, e in c.members for ts, te in mine)
                          > 0.5 * sum(e - s for _, s, e in c.members)]
                if chosen:
                    targets["자동(카드 1개)"] = target_from_clusters([max(chosen, key=lambda c: c.seconds)], clusters, eng, cache)
                n_people += 1
                for mode, tgt in targets.items():
                    for lv, thr in enumerate(eng.cal.levels if args.all_levels else []):
                        r = search([rec], tgt, eng, cache, thr)
                        for k, v in score(mine, [(c.start, c.end) for c in r.clips], used, rec.duration).items():
                            by_level[(mode, lv)][k] += v
                    res = search([rec], tgt, eng, cache)
                    sc = score(mine, [(c.start, c.end) for c in res.clips], used, rec.duration)
                    rows.append((clip, who, mode, sc, len(spans), len(clusters)))
                    for k, v in sc.items():
                        tot[mode][k] += v
                    tot[mode]["people"] += 1
                    if sc["target"] > 0:
                        per_person[mode].append(sc["found"] / sc["target"])
        print(f"{clip} 끝", file=sys.stderr, flush=True)

    hours = tot["참고"]["target"] / 3600
    print(f"모델 {eng.model}. 한국어 영상 {n_clips}개, 찾을 사람 {n_people}명(찾을 사람 말소리 {hours:.1f}시간, 참고 녹음 구간 제외)\n")
    for mode, d in tot.items():
        share = d["wrong"] / (d["found"] + d["wrong"]) if d["found"] + d["wrong"] else 0.0
        med = np.median(per_person[mode]) if per_person[mode] else 0.0
        if not d["target"]:
            continue
        print(f"{mode}: {int(d['people'])}명, 찾은 비율 {d['found'] / d['target']:.0%} (사람별 가운데값 {med:.0%}), "
              f"잘못 담은 시간 {d['wrong'] / 60:.1f}분, 담은 것 중 다른 사람 {share:.1%}")
    if by_level:
        print("\n단계별(0 넉넉하게 … 4 엄격하게): 찾은 비율 / 담은 것 중 다른 사람")
        for mode in tot:
            cells = []
            for lv, thr in enumerate(eng.cal.levels):
                d = by_level[(mode, lv)]
                sh = d["wrong"] / (d["found"] + d["wrong"]) if d["found"] + d["wrong"] else 0.0
                cells.append(f"{thr:.3f} {d['found'] / d['target'] if d['target'] else 0:.0%}/{sh:.1%}")
            print(f"  {mode:12s} " + "  ".join(cells))
    if args.rows:
        print("\n영상 사람 방식 | 말한 초 찾은 초 잘못 담은 초 | 영상의 화자 수 목소리 묶음 수")
        for clip, who, mode, sc, nspk, ncl in rows:
            print(f"{clip} {who:4s} {mode:10s} | {sc['target']:6.1f} {sc['found']:6.1f} {sc['wrong']:6.1f} | {nspk} {ncl}")


if __name__ == "__main__":
    main()
