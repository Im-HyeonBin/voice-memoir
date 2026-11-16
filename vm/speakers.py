"""누구의 목소리를 찾을지 정하고, 그 목소리가 나오는 구간을 찾는다.

- 자동 발견: 라이브러리 전체의 화자 지문을 군집으로 묶어 "이 폴더에 나오는 사람들"을
  보여준다. 깨끗한 참고 녹음이 없어도(예: 돌아가신 부모님) 고를 수 있다.
  같은 사람이라도 통화 녹음은 따로 묶이기 쉬워서, 일반 녹음 쪽 목소리를 전화 음질로
  바꿔 비교해 같은 사람이면 하나로 잇는다.
- 참고 음성: 짧은 녹음 하나로 지정한다. 참고 녹음에 다른 사람이 섞여 있어도
  가장 많이 말한 사람을 기준으로 삼는다.

구간 찾기는 두 단계다. 1.5초 조각으로 후보를 넉넉히 잡은 뒤, 후보 구간을 2초 안팎
덩어리로 다시 확인한다. 짧은 조각은 점수가 들쭉날쭉해 목소리가 닮은 가족을 잘못
잡기 쉬운데, 덩어리로 다시 보면 차이가 크게 벌어진다.

기준 지문은 여러 개다. 찾을 사람 목소리에 울림·잡음·압축을 입힌 지문을 함께 두고
가장 가까운 것과 비교한다. 대신 폴더에 나온 다른 사람들(경쟁자)보다 확실히 더 닮았을
때만 담아서, 기준을 넓힌 만큼 닮은 가족이 섞이지 않게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .audio import SR, AudioCache, is_narrowband, telephone_band
from .augment import narrow_variants, wide_variants
from .engine import EMB_DIM, HOP, WIN, VoiceEngine
from .library import FileRecord

# 닮은 정도 기준값들은 모델마다 숫자 크기가 달라 engine.cal(vm/engine.py Calibration)에 둔다.
PIECE_SEC = 2.0           # 확인 덩어리 길이. 3초보다 말 주고받는 경계가 덜 번진다
TOP_MEMBERS = 24          # 군집 대표 조각 수
ENROLL_SEC = 10.0         # 대상 화자 추출에 줄 찾을 사람 목소리 길이
OVERLAP_CONTEXT = 4.0     # 대상 화자 추출에 넣는 앞뒤 포함 길이(초)
MERGE_LINKAGE = "average"   # 쪼개진 같은 사람 묶음 합치기 방법
MERGE_OFFSET = 0.0          # 합치는 기준 = same_person_sure + 이 값
SURE_CHAIN = True           # 참고 녹음: '확실히 같은 목소리'와 확실히 같은 목소리도 같은 사람으로
SAMPLE_CONTEXT = 8.0      # 목소리마다 들어 볼 구간을 대표 조각 앞뒤 이만큼 안에서 고른다


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _cluster(emb: np.ndarray, distance: float, linkage: str = "average") -> np.ndarray:
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=linkage,
        distance_threshold=distance,
    )
    return model.fit_predict(emb)


def _merge_fragments(emb: np.ndarray, labels: np.ndarray, narrow: np.ndarray, sure: float) -> np.ndarray:
    """한 사람이 여러 묶음으로 쪼개졌으면 대표 지문끼리 '확실히 같은 사람'(sure)만큼 닮은 묶음을 합쳐 새 대표 지문을 돌려준다.

    짧은 조각끼리 묶으면 멀리서 녹음한 긴 회의처럼 한 사람 목소리가 조금씩 다른 녹음에서는 한 사람이 수십 묶음으로
    쪼개졌다(AMI 회의). 묶음 대표 지문은 조각보다 안정적이라 한 번 더 묶으면 합쳐진다. 통화 음질 묶음과 일반 묶음은
    따로 합친다(둘은 아래에서 전화 음질 지문으로 잇는다).
    """
    groups = [np.where(labels == k)[0] for k in range(labels.max() + 1)]
    cents = np.stack([_normalize(emb[g].mean(0)) for g in groups])
    phone = np.array([narrow[g].mean() >= 0.5 for g in groups])
    out = []
    for kind in (False, True):
        ks = np.where(phone == kind)[0]
        if len(ks) == 0:
            continue
        lab = _cluster(cents[ks], 1 - (sure + MERGE_OFFSET), MERGE_LINKAGE)
        for m in range(lab.max() + 1):
            idx = np.concatenate([groups[k] for k in ks[lab == m]])
            out.append(_normalize(emb[idx].mean(0)))
    return np.stack(out)


def _sim(emb: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """조각마다 여러 기준 지문 중 가장 가까운 것과의 닮은 정도."""
    return (emb @ protos.T).max(axis=1)


def _empty() -> np.ndarray:
    return np.zeros((0, EMB_DIM), np.float32)


@dataclass
class Target:
    """찾을 사람의 기준 지문들. 일반 녹음용과 전화 음질 녹음용을 따로 둔다."""
    wide: np.ndarray
    narrow: np.ndarray
    source: str
    info: dict = field(default_factory=dict)
    rivals_wide: np.ndarray = field(default_factory=_empty)
    rivals_narrow: np.ndarray = field(default_factory=_empty)
    margin: float = 0.0  # 경쟁자보다 이만큼은 더 닮아야 담는다
    enroll: np.ndarray | None = None  # 대상 화자 추출에 줄 찾을 사람 목소리(깨끗한 쪽)

    def __post_init__(self):
        self.wide = np.atleast_2d(self.wide).astype(np.float32)
        self.narrow = np.atleast_2d(self.narrow).astype(np.float32)
        dim = self.wide.shape[1]  # 모델마다 지문 길이가 다르다
        if self.rivals_wide.shape[1] != dim:
            self.rivals_wide = np.zeros((0, dim), np.float32)
        if self.rivals_narrow.shape[1] != dim:
            self.rivals_narrow = np.zeros((0, dim), np.float32)

    def protos(self, rec: FileRecord) -> np.ndarray:
        return self.narrow if rec.narrowband else self.wide

    def rivals(self, rec: FileRecord) -> np.ndarray:
        return self.rivals_narrow if rec.narrowband else self.rivals_wide

    def set_rivals(self, clusters: list["SpeakerCluster"], exclude: set[int] = frozenset()) -> "Target":
        """폴더에 나온 다른 사람들을 경쟁자로 둔다. exclude는 찾을 사람과 같은 사람인 목소리 번호."""
        others = [c for c in clusters if c.index not in exclude]
        empty = np.zeros((0, self.wide.shape[1]), np.float32)
        self.rivals_wide = np.stack([c.wide for c in others if c.wide is not None]) if any(
            c.wide is not None for c in others) else empty
        self.rivals_narrow = np.stack([c.narrow for c in others]) if others else empty
        return self


@dataclass
class SpeakerCluster:
    index: int
    wide: np.ndarray | None       # 일반 녹음 기준 지문(통화 녹음에만 나온 사람은 없음)
    narrow: np.ndarray            # 전화 음질 기준 지문
    seconds: float
    files: list[str]
    members: list[tuple[str, float, float]]  # 대표 조각 (파일, 시작, 끝)
    phone_linked: bool = False
    sample: tuple[str, float, float] | None = None

    @property
    def centroid(self) -> np.ndarray:
        return self.wide if self.wide is not None else self.narrow


def _narrow_proto(engine: VoiceEngine, cache: AudioCache,
                  members: list[tuple[str, float, float]]) -> np.ndarray:
    """일반 녹음 조각들을 전화 음질로 바꿔 지문을 다시 뽑는다."""
    pieces, spans, t = [], [], 0.0
    for path, s, e in members:
        x = telephone_band(cache.slice(path, s, e))
        pieces.append(x)
        spans.append((t, t + len(x) / SR))
        t += len(x) / SR
    emb = engine.embed(np.concatenate(pieces), np.asarray(spans, np.float32))
    return _normalize(emb.mean(0))


def _variant_protos(engine: VoiceEngine, wav: np.ndarray, windows: np.ndarray,
                    variants: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [_normalize(engine.embed(v, windows).mean(0)) for v in variants.values()]


def _augment(engine: VoiceEngine, pieces: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """찾을 사람의 일반 녹음 조각들에 녹음 조건을 입혀 기준 지문을 더 만든다."""
    spans, t = [], 0.0
    for x in pieces:
        spans.append((t, t + len(x) / SR))
        t += len(x) / SR
    wav = np.concatenate(pieces)
    windows = np.asarray(spans, np.float32)
    return (_variant_protos(engine, wav, windows, wide_variants(wav)),
            _variant_protos(engine, wav, windows, narrow_variants(wav)))


def discover_speakers(records: list[FileRecord], engine: VoiceEngine, cache: AudioCache,
                      distance: float | None = None, min_seconds: float = 4.0,
                      max_windows: int = 4000, seed: int = 0, merge: bool = True) -> list[SpeakerCluster]:
    """라이브러리에 나오는 사람들을 말한 시간이 긴 순서로 돌려준다. merge면 쪼개진 같은 사람 묶음을 합친다."""
    owners = [(ri, wi) for ri, r in enumerate(records) for wi in range(len(r.windows))]
    if not owners:
        return []
    emb = np.concatenate([r.embeddings for r in records if len(r.embeddings)])
    narrow_win = np.array([records[ri].narrowband for ri, _ in owners])

    # 창이 너무 많으면 일부만 군집화하고 나머지는 가까운 군집에 붙인다.
    pick = np.arange(len(emb))
    if len(emb) > max_windows:
        pick = np.sort(np.random.default_rng(seed).choice(len(emb), max_windows, replace=False))
    labels = _cluster(emb[pick], engine.cal.cluster_distance if distance is None else distance)
    if merge and engine.cal.merge:
        centroids = _merge_fragments(emb[pick], labels, narrow_win[pick], engine.cal.same_person_sure)
    else:
        centroids = np.stack([_normalize(emb[pick][labels == k].mean(0)) for k in range(labels.max() + 1)])
    assign = (emb @ centroids.T).argmax(1)

    raw = []
    for k in range(len(centroids)):
        idx = np.where(assign == k)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-(emb[idx] @ centroids[k]))][:TOP_MEMBERS]
        members = [(records[owners[m][0]].path, *map(float, records[owners[m][0]].windows[owners[m][1]]))
                   for m in top]
        raw.append({
            "centroid": centroids[k], "idx": idx, "members": members,
            "narrow_share": float(narrow_win[idx].mean()),
            "files": {records[owners[m][0]].path for m in idx},
        })

    wide = [c for c in raw if c["narrow_share"] < 0.5]
    narrow_only = [c for c in raw if c["narrow_share"] >= 0.5]
    for c in wide:
        c["narrow"] = _narrow_proto(engine, cache, c["members"]) if narrow_only else c["centroid"]
        c["linked"] = False

    # 통화 녹음 군집이 일반 녹음의 누군가와 같은 사람이면 하나로 잇는다.
    clusters_out = []
    for n in narrow_only:
        best = max(wide, key=lambda c: float(c["narrow"] @ n["centroid"]), default=None)
        if best is not None and float(best["narrow"] @ n["centroid"]) >= engine.cal.link:
            best["idx"] = np.concatenate([best["idx"], n["idx"]])
            best["files"] |= n["files"]
            best["narrow"] = _normalize(best["narrow"] + n["centroid"])
            best["linked"] = True
        else:
            clusters_out.append(SpeakerCluster(0, None, n["centroid"], 0.0, [], n["members"]))
            clusters_out[-1].seconds = len(n["idx"]) * HOP + (WIN - HOP)
            clusters_out[-1].files = sorted(n["files"])
    for c in wide:
        clusters_out.append(SpeakerCluster(
            0, c["centroid"], c["narrow"], len(c["idx"]) * HOP + (WIN - HOP),
            sorted(c["files"]), c["members"], phone_linked=c["linked"]))

    clusters_out = [c for c in clusters_out if c.seconds >= min_seconds]
    clusters_out.sort(key=lambda c: c.seconds, reverse=True)
    for i, c in enumerate(clusters_out, 1):
        c.index = i
        c.sample = _sample_clip(c, records, engine, cache)
    return clusters_out


def _sample_clip(c: SpeakerCluster, records: list[FileRecord], engine: VoiceEngine,
                 cache: AudioCache, max_len: float = 6.0) -> tuple[str, float, float]:
    """들어 볼 대표 구간: 대표 조각 앞뒤 SAMPLE_CONTEXT초 안에서 가장 확실한 구간을 고른다.

    예전에는 대표 조각이 든 파일 전체를 찾았는데, 몇 시간짜리 녹음에서는 목소리마다 전체 찾기를 한 번씩 하게 되어
    목소리 나누기가 수십 분 걸렸다(AMI 회의 2시간 분량). 대표 조각 둘레만 보면 녹음 길이와 상관없이 빠르다.
    """
    target = Target(c.centroid, c.narrow, "cluster")
    by_path = {r.path: r for r in records}
    best = None
    for path, s, e in c.members[:6]:
        rec = by_path[path]
        near = (rec.windows[:, 1] > s - SAMPLE_CONTEXT) & (rec.windows[:, 0] < e + SAMPLE_CONTEXT)
        local = replace(rec, windows=rec.windows[near], embeddings=rec.embeddings[near])
        for clip in find_clips([local], target, engine, cache):
            if best is None or (clip.score, clip.duration) > (best.score, best.duration):
                best = clip
    if best is None:
        return c.members[0]
    take = min(best.duration, max_len)
    mid = (best.start + best.end) / 2
    return best.path, max(best.start, mid - take / 2), max(best.start, mid - take / 2) + take


def _join_enroll(pieces: list[np.ndarray], cut: bool = True) -> np.ndarray | None:
    """대상 화자 추출에 줄 찾을 사람 목소리: 조각을 ENROLL_SEC까지 이어 붙인다. cut이면 조각마다 WIN초까지만."""
    out, total = [], 0
    for x in pieces:
        x = x[: int(WIN * SR)] if cut else x
        out.append(x)
        total += len(x)
        if total >= ENROLL_SEC * SR:
            break
    return np.concatenate(out).astype(np.float32) if total >= 2 * SR else None


def target_from_clusters(selected: list[SpeakerCluster], clusters: list[SpeakerCluster] | None = None,
                         engine: VoiceEngine | None = None, cache: AudioCache | None = None,
                         augment: bool = True) -> Target:
    """'같은 사람'으로 고른 목소리 군집(하나 이상)으로 기준 지문을 만든다.

    clusters를 주면 고르지 않은 목소리들을 경쟁자로 둔다.
    """
    wides = [c.wide for c in selected if c.wide is not None]
    narrow = [_normalize(np.mean([c.narrow for c in selected], axis=0))]
    wide = [_normalize(np.mean(wides, axis=0))] if wides else list(narrow)
    if augment and wides and engine is not None and cache is not None:
        members = [m for c in selected if c.wide is not None for m in c.members][:TOP_MEMBERS]
        more_wide, more_narrow = _augment(engine, [cache.slice(p, s0, e0) for p, s0, e0 in members])
        wide += more_wide
        narrow += more_narrow
    enroll = None
    if cache is not None:
        pieces = [cache.slice(p, s0, e0) for c in selected if c.wide is not None for p, s0, e0 in c.members]
        enroll = _join_enroll(pieces)
    target = Target(np.stack(wide), np.stack(narrow), "자동 발견",
                    {"clusters": [c.index for c in selected], "seconds": sum(c.seconds for c in selected)},
                    margin=engine.cal.rival_margin if engine is not None else 0.0, enroll=enroll)
    if clusters:
        target.set_rivals(clusters, exclude={c.index for c in selected})
    return target


def target_from_reference(engine: VoiceEngine, wav: np.ndarray,
                          clusters: list[SpeakerCluster] | None = None, augment: bool = True,
                          same: set[int] = frozenset(), sure_protos: bool = True) -> Target:
    """참고 녹음에서 가장 많이 말한 사람의 지문을 기준으로 삼는다.

    clusters를 주면 참고 녹음과 다른 사람으로 보이는 목소리들을 경쟁자로 둔다.
    참고 녹음과 확실히 같은 목소리만 같은 사람으로 보고, 애매한 목소리는 다른 사람(경쟁자)으로 둔다.
    애매한 목소리 번호는 info["unsure"]에 남겨 사람에게 물어본다. same은 사람이 같은 사람이라고
    확인한 목소리 번호로, 경쟁자에서 빼고 기준 지문에 더한다. sure_protos면 확실히 같은 목소리도
    기준 지문에 더한다(시끄러운 곳·멀리서 녹음된 그 사람 목소리를 기준으로 삼아 더 찾는다).
    """
    windows, emb = engine.analyze_wave(wav)
    if len(emb) == 0:
        raise ValueError("참고 녹음에서 말소리를 찾지 못했어요. 1초 넘게 말하는 부분이 있는지 확인해 주세요.")
    cal = engine.cal
    labels = _cluster(emb, cal.cluster_distance) if len(emb) >= 4 else np.zeros(len(emb), int)
    used = labels == np.bincount(labels).argmax()
    phone_ref = is_narrowband(wav)
    wide = [_normalize(emb[used].mean(0))]
    narrow = list(wide) if phone_ref else [_normalize(engine.embed(telephone_band(wav), windows[used]).mean(0))]
    if augment and not phone_ref:
        wide += _variant_protos(engine, wav, windows[used], wide_variants(wav))
        narrow += _variant_protos(engine, wav, windows[used], narrow_variants(wav))
    info = {"windows": int(len(emb)), "used": int(used.sum()),
            "mixed": bool(len(set(labels.tolist())) > 1), "reference_is_phone": phone_ref, "unsure": []}
    if clusters:
        protos = np.vstack([np.stack(wide), np.stack(narrow)])
        sims = {c.index: float(_sim(c.centroid[None], protos)[0]) for c in clusters}
        sure = {i for i, v in sims.items() if v >= cal.same_person_sure}
        if SURE_CHAIN and cal.chain:  # 확실히 같은 목소리와 확실히 같을 만큼 닮은 목소리도 같은 사람으로 본다(쪼개진 묶음을 따라간다)
            cents = {c.index: c.centroid for c in clusters}
            frontier = set(sure)
            while frontier:
                ref = np.stack([cents[j] for j in frontier])
                frontier = set()
                for i, v in cents.items():
                    if i not in sure:
                        best = float((ref @ v).max())
                        sims[i] = max(sims[i], best)
                        if best >= cal.same_person_sure:
                            frontier.add(i)
                sure |= frontier
        info["unsure"] = sorted((i for i, v in sims.items() if cal.same_person <= v < cal.same_person_sure),
                                key=lambda i: -sims[i])
        confirmed = [c for c in clusters if c.index in (set(same) | (sure if sure_protos else set()))]
        wide += [c.wide for c in confirmed if c.wide is not None]
        narrow += [c.narrow for c in confirmed]
    merged: list[list[float]] = []  # 겹치는 창을 이어 붙여 같은 소리가 두 번 들어가지 않게
    for s0, e0 in windows[used]:
        if merged and s0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(e0))
        else:
            merged.append([float(s0), float(e0)])
    enroll = None if phone_ref else _join_enroll([wav[int(s0 * SR):int(e0 * SR)] for s0, e0 in merged], cut=False)
    target = Target(np.stack(wide), np.stack(narrow), "참고 녹음", info, margin=cal.rival_margin, enroll=enroll)
    if clusters:
        target.set_rivals(clusters, exclude=sure | set(same))
    return target


@dataclass
class Clip:
    path: str
    date: date | None
    date_source: str
    start: float
    end: float
    score: float   # 덩어리별 닮은 정도의 평균
    peak: float
    confirmed: bool = False  # 사람이 '맞아요'로 확인한 구간
    overlap: bool = False    # 다른 목소리와 겹쳐 대상 화자 추출로 찾은 구간(사람 확인용)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def year(self) -> int | None:
        return self.date.year if self.date else None


def match_record(rec: FileRecord, protos: np.ndarray, threshold: float,
                 gap: float = 0.4, min_len: float = 1.0) -> list[Clip]:
    """1단계: 조각 점수로 후보 구간을 잡는다."""
    if len(rec.windows) == 0:
        return []
    scores = _sim(rec.embeddings, np.atleast_2d(protos))
    hit = scores >= threshold
    win = rec.windows
    clips: list[Clip] = []
    i, n = 0, len(win)
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hit[j + 1] and win[j + 1][0] <= win[j][1] + gap:
            j += 1
        start, end = float(win[i][0]), float(win[j][1])
        # 앞뒤에 다른 사람 조각이 겹쳐 있으면 겹친 부분의 절반만 가져간다.
        if i > 0 and not hit[i - 1] and win[i - 1][1] > start:
            start = (start + float(win[i - 1][1])) / 2
        if j + 1 < n and not hit[j + 1] and win[j + 1][0] < end:
            end = (end + float(win[j + 1][0])) / 2
        if end - start >= min_len:
            seg = scores[i:j + 1]
            clips.append(Clip(rec.path, rec.date, rec.date_source, start, end,
                              float(seg.mean()), float(seg.max())))
        i = j + 1
    return clips


def _runs(clip: Clip, edges: np.ndarray, sims: np.ndarray, mask: np.ndarray) -> list[Clip]:
    """연달아 조건을 만족한 덩어리를 하나의 구간으로 묶는다."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        seg = sims[i:j + 1]
        out.append(Clip(clip.path, clip.date, clip.date_source, float(edges[i]), float(edges[j + 1]),
                        float(seg.mean()), float(seg.max())))
        i = j + 1
    return [c for c in out if c.duration >= 1.0]


def verify_clip(engine: VoiceEngine, wav: np.ndarray, clip: Clip, protos: np.ndarray,
                threshold: float, rivals: np.ndarray | None = None,
                margin: float = 0.0) -> tuple[list[Clip], list[Clip]]:
    """2단계: 후보 구간을 2초 안팎 덩어리로 나눠 다시 확인한다.

    경쟁자가 있으면 가장 닮은 경쟁자보다 margin 이상 더 닮은 덩어리만 남긴다.
    돌려주는 값: (확정 구간, 확인해 볼 구간). 확인해 볼 구간은 기준에 조금 못 미치지만
    다른 누구보다도 찾을 사람과 더 닮은 덩어리다.
    """
    n = max(1, int(round(clip.duration / PIECE_SEC)))
    edges = np.linspace(clip.start, clip.end, n + 1)
    pieces = np.stack([edges[:-1], edges[1:]], axis=1).astype(np.float32)
    emb = engine.embed(wav, pieces)
    sims = _sim(emb, np.atleast_2d(protos))
    lead = sims - _sim(emb, rivals) if rivals is not None and len(rivals) else np.full(n, np.inf)
    ok = (sims >= threshold) & (lead >= margin)
    maybe = ~ok & (sims >= threshold - engine.cal.maybe_band) & (lead >= 0)
    return _runs(clip, edges, sims, ok), _runs(clip, edges, sims, maybe)


def join_phrases(rec: FileRecord, clips: list[Clip], max_gap: float = 1.2) -> list[Clip]:
    """숨 쉬는 짧은 쉼 때문에 쪼개진 구간을 한 문장으로 다시 잇는다.

    사이에 말소리 조각이 하나라도 있으면 잇지 않는다. 숨 쉬는 쉼에는 말소리가 없고, 1초 남짓한 틈에 든
    말소리는 다른 사람의 짧은 맞장구일 수 있는데, 1.5초 조각에는 양쪽 목소리가 섞여 점수로는 가려내기 어렵다.
    어려운 세트에서 이어 붙인 틈은 모두 말소리가 없는 쉼이었다(예전 규칙과 결과가 같다).
    """
    if not clips:
        return clips
    centers = rec.windows.mean(1) if len(rec.windows) else np.zeros(0)
    out = [clips[0]]
    for c in clips[1:]:
        prev = out[-1]
        gap = c.start - prev.end
        inside = (centers > prev.end) & (centers < c.start)
        if 0 <= gap <= max_gap and not inside.any():
            w = prev.duration + c.duration
            out[-1] = Clip(prev.path, prev.date, prev.date_source, prev.start, c.end,
                           (prev.score * prev.duration + c.score * c.duration) / w,
                           max(prev.peak, c.peak))
        else:
            out.append(c)
    return out


@dataclass
class SearchResult:
    clips: list[Clip]   # 담은 구간
    maybe: list[Clip]   # 사람이 확인해 볼 구간


def find_overlaps(rec: FileRecord, wav: np.ndarray, target: Target, engine: VoiceEngine, extractor,
                  covered: list[Clip], threshold: float, floor: float | None = None) -> list[Clip]:
    """평소 찾기로 못 잡은 말소리 중, 다른 목소리와 겹쳐 묻힌 찾을 사람 목소리를 대상 화자 추출로 찾는다.

    찾을 사람과 조금이라도 닮은 창(floor 이상, 기본 candidate_floor)만 모아 앞뒤를 OVERLAP_CONTEXT까지 넓혀 뽑고,
    뽑은 소리를 2초 덩어리로 다시 확인한다. 기준과 경쟁자 비교를 모두 넘은 구간만 돌려주며, 사람 확인용이다.
    전화 음질 녹음은 추출 모델이 학습하지 않은 조건이라 건너뛴다.
    """
    if extractor is None or target.enroll is None or rec.narrowband or len(rec.windows) == 0:
        return []
    protos = target.protos(rec)
    sims = _sim(rec.embeddings, protos)
    floor = engine.cal.candidate_floor if floor is None else floor
    spans = []
    for (s, e), v in zip(rec.windows, sims):
        if v < floor or any(_overlap(s, e, c.start, c.end) > 0.5 * (e - s) for c in covered):
            continue
        if spans and s <= spans[-1][1] + 0.4:
            spans[-1][1] = max(spans[-1][1], float(e))
        else:
            spans.append([float(s), float(e)])
    out, dur = [], len(wav) / SR
    rivals = target.rivals(rec)
    for s, e in spans:
        if e - s < 1.0:
            continue
        half = max(OVERLAP_CONTEXT, e - s) / 2
        a = max(0.0, min((s + e) / 2 - half, dur - 2 * half))
        b = min(dur, a + 2 * half)
        y = extractor(wav[int(a * SR):int(b * SR)], target.enroll, key=(rec.path, round(a, 2), round(b, 2)))
        n = max(1, int(round((e - s) / PIECE_SEC)))
        edges = np.linspace(s, e, n + 1)
        emb = engine.embed(y, np.stack([edges[:-1] - a, edges[1:] - a], axis=1).astype(np.float32))
        sim = _sim(emb, protos)
        lead = sim - _sim(emb, rivals) if len(rivals) else np.full(n, np.inf)
        base = Clip(rec.path, rec.date, rec.date_source, s, e, 0.0, 0.0)
        out += [replace(c, overlap=True) for c in _runs(base, edges, sim, (sim >= threshold) & (lead >= target.margin))]
    return out


def search(records: list[FileRecord], target: Target, engine: VoiceEngine, cache: AudioCache,
           threshold: float | None = None, extractor=None, overlap_floor: float | None = None) -> SearchResult:
    """라이브러리 전체에서 찾을 사람의 목소리 구간과, 확인해 볼 구간을 찾는다.

    extractor(대상 화자 추출, vm/tse.py)를 주면 다른 목소리와 겹친 구간도 찾아 확인해 볼 구간에 더한다.
    """
    cal = engine.cal
    threshold = cal.threshold if threshold is None else threshold
    # 엄격한 단계도 후보는 기본 단계만큼 넉넉히 잡는다. 후보 기준을 같이 올리면 후보 구간이 달라져 2초 덩어리
    # 경계가 옮겨 가고, 그 바람에 엄격하게 했는데 닮은 가족을 더 담는 일이 있었다(어려운 세트 삼촌 0.9→2.5초).
    # 같은 덩어리에 기준만 높이면 담는 구간은 기본 단계의 일부가 된다.
    candidate = max(cal.candidate_floor, min(threshold, cal.threshold) - cal.candidate_margin)
    clips: list[Clip] = []
    maybe: list[Clip] = []
    for rec in records:
        protos = target.protos(rec)
        cands = match_record(rec, protos, candidate)
        if not cands and extractor is None:
            continue
        wav = cache.get(rec.path)
        kept, unsure = [], []
        for c in cands:
            k, u = verify_clip(engine, wav, c, protos, threshold, target.rivals(rec), target.margin)
            kept += k
            unsure += u
        kept = join_phrases(rec, sorted(kept, key=lambda k: k.start))
        clips += kept
        maybe += unsure
        maybe += find_overlaps(rec, wav, target, engine, extractor, kept + unsure, threshold, overlap_floor)
    return SearchResult(clips, maybe)


def find_clips(records: list[FileRecord], target: Target, engine: VoiceEngine, cache: AudioCache,
               threshold: float | None = None) -> list[Clip]:
    return search(records, target, engine, cache, threshold).clips


# ── 사람이 확인한 내용 반영 ──────────────────────────────

Region = tuple[str, float, float]  # (파일, 시작, 끝)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def region_embedding(engine: VoiceEngine, cache: AudioCache, region: Region) -> np.ndarray:
    path, s0, e0 = region
    n = max(1, int(round((e0 - s0) / PIECE_SEC)))
    edges = np.linspace(s0, e0, n + 1)
    emb = engine.embed(cache.get(path), np.stack([edges[:-1], edges[1:]], axis=1).astype(np.float32))
    return _normalize(emb.mean(0))


def with_review(target: Target, engine: VoiceEngine, cache: AudioCache, records: list[FileRecord],
                accepted: list[Region], rejected: list[Region]) -> Target:
    """'맞아요' 구간은 기준 지문에, '아니에요' 구간은 경쟁자에 더해 다음 찾기에 반영한다."""
    narrow = {r.path: r.narrowband for r in records}
    t = replace(target, wide=target.wide.copy(), narrow=target.narrow.copy(),
                rivals_wide=target.rivals_wide.copy(), rivals_narrow=target.rivals_narrow.copy())
    for region in accepted:
        e = region_embedding(engine, cache, region)[None]
        if narrow.get(region[0]):
            t.narrow = np.vstack([t.narrow, e])
        else:
            t.wide = np.vstack([t.wide, e])
    for region in rejected:
        e = region_embedding(engine, cache, region)[None]
        if narrow.get(region[0]):
            t.rivals_narrow = np.vstack([t.rivals_narrow, e])
        else:
            t.rivals_wide = np.vstack([t.rivals_wide, e])
    return t


def apply_review(result: SearchResult, records: list[FileRecord],
                 accepted: list[Region], rejected: list[Region]) -> SearchResult:
    """사람의 판단을 결과에 그대로 적용한다. '아니에요'는 빼고, '맞아요'는 반드시 넣는다."""
    rec = {r.path: r for r in records}

    def rejected_part(c: Clip) -> bool:
        return any(p == c.path and _overlap(c.start, c.end, s0, e0) > 0.5 * c.duration for p, s0, e0 in rejected)

    clips = [c for c in result.clips if not rejected_part(c)]
    for p, s0, e0 in accepted:
        if p in rec and not any(c.path == p and _overlap(c.start, c.end, s0, e0) > 0.5 * (e0 - s0) for c in clips):
            r = rec[p]
            clips.append(Clip(p, r.date, r.date_source, s0, e0, 1.0, 1.0, confirmed=True))
    decided = accepted + rejected
    maybe = [m for m in result.maybe if not rejected_part(m)
             and not any(m.path == p and _overlap(m.start, m.end, s0, e0) > 0 for p, s0, e0 in decided)
             # 이미 담긴 구간 안에 거의 들어 있는 것만 뺀다. 겹쳐 말한 구간은 담긴 구간 바로 옆에 붙어 있어
             # 경계가 조금 겹치기 쉬운데, 조금이라도 겹치면 빼면 목록에서 다 사라진다.
             and not any(m.path == c.path and _overlap(m.start, m.end, c.start, c.end) > 0.5 * m.duration
                         for c in clips)]
    clips.sort(key=lambda c: (c.path, c.start))
    maybe.sort(key=lambda c: -c.score)
    return SearchResult(clips, maybe)
