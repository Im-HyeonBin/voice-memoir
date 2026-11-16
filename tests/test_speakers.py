"""구간 찾기 흐름 전체(후보 → 덩어리 확인 → 경쟁자 비교 → 겹친 구간 → 사람 확인 반영)를 가짜 엔진으로 시험한다."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vm.audio import SR
from vm.speakers import (Clip, SearchResult, Target, apply_review, join_phrases, match_record, search)

from conftest import FakeCache, compose, fake_embedding, record

ME_EMB = fake_embedding(np.full(SR, 0.5, np.float32))
OTHER_EMB = fake_embedding(np.full(SR, -0.5, np.float32))


def make_target(engine, rivals=True, enroll=True) -> Target:
    return Target(ME_EMB[None], ME_EMB[None], "시험",
                  rivals_wide=OTHER_EMB[None] if rivals else np.zeros((0, 3), np.float32),
                  rivals_narrow=OTHER_EMB[None] if rivals else np.zeros((0, 3), np.float32),
                  margin=engine.cal.rival_margin,
                  enroll=np.full(3 * SR, 0.5, np.float32) if enroll else None)


class StripOthers:
    """가짜 대상 화자 추출: 다른 사람 표본을 지운다. 호출 수를 센다."""

    def __init__(self):
        self.calls = []

    def __call__(self, mix, enroll, key=None):
        self.calls.append(key)
        return np.where(mix < 0, 0, mix).astype(np.float32)


@pytest.fixture
def scene(engine):
    # 0~6초 찾을 사람, 6~12초 다른 사람, 12~18초 다른 사람 위에 찾을 사람이 작게 겹침, 18~24초 다른 사람
    wav = compose(("me", 6), ("other", 6), ("mix", 6), ("other", 6))
    rec = record(engine, "a.wav", wav)
    return rec, FakeCache({"a.wav": wav})


def test_search_finds_target_and_not_others(engine, scene):
    rec, cache = scene
    res = search([rec], make_target(engine), engine, cache)
    assert len(res.clips) == 1
    c = res.clips[0]
    assert c.start <= 0.1 and 5.5 <= c.end <= 6.5
    assert not c.overlap and c.date == date(2020, 5, 1)
    assert all(m.end <= 6.5 or m.start >= 6.5 for m in res.maybe)
    assert not any(m.start < 18 and m.end > 12 for m in res.maybe)  # 겹친 구간은 추출기 없이는 못 찾는다


def test_rival_margin_blocks_mixed_region(engine):
    """경쟁자가 없으면 섞인 구간도 기준만 넘으면 담기지만, 경쟁자와 비교하면 빠진다."""
    wav = compose(("mix", 8))
    rec = record(engine, "m.wav", wav)
    cache = FakeCache({"m.wav": wav})
    lenient = search([rec], make_target(engine, rivals=False), engine, cache, threshold=0.40)
    strict = search([rec], make_target(engine), engine, cache, threshold=0.40)
    assert sum(c.duration for c in lenient.clips) > 6
    assert strict.clips == []


def test_overlap_extraction_goes_to_review_list_only(engine, scene):
    rec, cache = scene
    ext = StripOthers()
    res = search([rec], make_target(engine), engine, cache, extractor=ext)
    assert len(res.clips) == 1  # 담은 구간은 그대로
    ov = [m for m in res.maybe if m.overlap]
    assert ov, "겹친 구간을 확인 목록에 올려야 한다"
    assert all(11 <= m.start and m.end <= 19 for m in ov)
    assert sum(m.duration for m in ov) >= 4
    assert ext.calls and all(k[0] == "a.wav" for k in ext.calls)  # 같은 구간을 다시 뽑지 않게 열쇠를 준다


def test_overlap_skipped_for_phone_recordings_and_without_enroll(engine):
    wav = compose(("mix", 8))
    ext = StripOthers()
    phone = record(engine, "p.wav", wav, narrowband=True)
    search([phone], make_target(engine), engine, FakeCache({"p.wav": wav}), extractor=ext)
    wide = record(engine, "w.wav", wav)
    search([wide], make_target(engine, enroll=False), engine, FakeCache({"w.wav": wav}), extractor=ext)
    assert ext.calls == []


def test_phone_recordings_use_narrow_prototypes(engine):
    t = make_target(engine)
    t.narrow = OTHER_EMB[None]
    wav = compose(("other", 6))
    phone = record(engine, "p.wav", wav, narrowband=True)
    assert np.allclose(t.protos(phone), OTHER_EMB[None])


def test_match_record_trims_neighbouring_other_speaker(engine):
    wav = compose(("other", 3), ("me", 6), ("other", 3))
    rec = record(engine, "b.wav", wav)
    clips = match_record(rec, ME_EMB[None], 0.6)
    assert len(clips) == 1
    assert 2.5 <= clips[0].start <= 3.5 and 8.5 <= clips[0].end <= 9.5


def test_join_phrases_only_across_silence(engine):
    rec = record(engine, "c.wav", compose(("me", 3), ("quiet", 1), ("me", 3)))
    # 쉼 1초에는 말소리 창이 없게 만든다(실제로는 말소리 검출이 쉼을 비운다)
    keep = ~((rec.windows[:, 0] < 4) & (rec.windows[:, 1] > 3))
    rec.windows, rec.embeddings = rec.windows[keep], rec.embeddings[keep]
    a = Clip("c.wav", None, "unknown", 0.0, 3.0, 0.9, 0.9)
    b = Clip("c.wav", None, "unknown", 4.0, 7.0, 0.8, 0.8)
    joined = join_phrases(rec, [a, b])
    assert len(joined) == 1 and (joined[0].start, joined[0].end) == (0.0, 7.0)
    assert joined[0].score == pytest.approx(0.85)

    # 1초 맞장구: 이 틈에 걸친 1.5초 조각은 양쪽 목소리가 섞여 점수로는 다른 사람으로 안 보인다. 그래도 잇지 않는다
    other = record(engine, "d.wav", compose(("me", 3), ("other", 1), ("me", 3)))
    assert len(join_phrases(other, [a, b])) == 2
    assert len(join_phrases(rec, [a, b], max_gap=0.5)) == 2  # 틈이 너무 길면 잇지 않는다


# ── 사람 확인 반영 ──────────────────────────────

def clip(s, e, score=0.7, path="a.wav", **kw):
    return Clip(path, date(2020, 1, 1), "filename", s, e, score, score, **kw)


def test_apply_review_rejected_and_accepted(engine, scene):
    rec, _ = scene
    res = SearchResult([clip(0, 6), clip(10, 12)], [])
    out = apply_review(res, [rec], accepted=[("a.wav", 20.0, 23.0)], rejected=[("a.wav", 9.5, 12.0)])
    spans = [(c.start, c.end, c.confirmed) for c in out.clips]
    assert spans == [(0, 6, False), (20.0, 23.0, True)]  # '아니에요'는 빼고, '맞아요'는 반드시 넣는다


def test_apply_review_small_rejection_does_not_drop_clip(engine, scene):
    rec, _ = scene
    out = apply_review(SearchResult([clip(0, 6)], []), [rec], [], [("a.wav", 5.0, 7.0)])
    assert len(out.clips) == 1  # 끝 1초만 겹치면 남긴다(절반 넘게 겹칠 때만 뺀다)


def test_apply_review_keeps_overlap_items_next_to_kept_clip(engine, scene):
    """겹친 구간은 담긴 구간 바로 옆이라 경계가 조금 겹친다. 조금 겹쳤다고 확인 목록에서 사라지면 안 된다."""
    rec, _ = scene
    res = SearchResult([clip(0, 6)], [clip(5.5, 9.0, overlap=True), clip(1.0, 4.0, score=0.5)])
    out = apply_review(res, [rec], [], [])
    assert [(m.start, m.end) for m in out.maybe] == [(5.5, 9.0)]


def test_apply_review_removes_decided_items_and_sorts(engine, scene):
    rec, _ = scene
    res = SearchResult([], [clip(10, 12, score=0.5), clip(14, 16, score=0.6), clip(18, 20, score=0.4)])
    out = apply_review(res, [rec], accepted=[], rejected=[("a.wav", 11.5, 12.5)])
    assert [(m.start, m.score) for m in out.maybe] == [(14, 0.6), (18, 0.4)]  # 답한 것은 빼고 닮은 순서로


def test_discover_sample_cost_does_not_grow_with_recording_length(engine):
    """목소리마다 들어 볼 구간을 고를 때 녹음 전체가 아니라 대표 조각 둘레만 본다(몇 시간짜리 녹음에서 수십 분 걸리던 문제)."""
    from vm.audio import AudioCache  # noqa: F401  (형식 확인용)
    from vm.speakers import discover_speakers

    def embedded_rows(minutes):
        wav = compose(*[(("me", "other")[i % 2], 5.0) for i in range(minutes * 12)])
        rec = record(engine, "long.wav", wav)
        rows = []
        real = engine.embed
        engine.embed = lambda w, win: rows.append(len(win)) or real(w, win)
        try:
            clusters = discover_speakers([rec], engine, FakeCache({"long.wav": wav}))
        finally:
            engine.embed = real
        return clusters, sum(rows)

    short_clusters, short_rows = embedded_rows(2)
    long_clusters, long_rows = embedded_rows(20)
    assert len(long_clusters) >= 2
    for c in long_clusters:  # 들어 볼 구간은 대표 조각 둘레에서 고른다
        p, s, e = c.sample
        assert any(abs(s - ms) <= 8 + 6 for _, ms, _ in c.members[:6])
    assert long_rows <= short_rows * 2  # 녹음이 10배 길어도 확인할 조각 수는 거의 그대로


def test_merge_fragments_joins_same_person_only():
    """한 사람이 두 묶음으로 쪼개졌으면 합치고, 다른 사람과는 합치지 않는다. 통화 음질 묶음은 따로 합친다."""
    from vm.speakers import _merge_fragments, _normalize

    rng = np.random.default_rng(0)

    def windows(center, n):
        return np.stack([_normalize(center + rng.normal(0, 0.05, 3)) for _ in range(n)])

    me_a, me_b = windows(np.array([1.0, 0.15, 0.1]), 10), windows(np.array([1.0, -0.15, 0.1]), 10)
    other = windows(np.array([0.0, 1.0, 0.1]), 10)
    me_phone = windows(np.array([1.0, 0.0, 0.1]), 10)
    emb = np.vstack([me_a, me_b, other, me_phone])
    labels = np.repeat([0, 1, 2, 3], 10)
    narrow = np.repeat([False, False, False, True], 10)
    cents = _merge_fragments(emb, labels, narrow, sure=0.8)
    assert len(cents) == 3  # me_a + me_b, other, me_phone(따로)
    sims = cents @ np.stack([_normalize(me_a.mean(0) + me_b.mean(0)), _normalize(other.mean(0))]).T
    assert (sims.max(0) > 0.99).all()


def test_reference_follows_sure_chain(engine):
    """참고 녹음과 직접은 덜 닮았어도 '확실히 같은 목소리'와 확실히 같은 묶음은 같은 사람으로 본다(실제 회의 녹음에서
    한 사람이 수십 묶음으로 쪼개져, 자기 묶음이 경쟁자가 되어 자기 목소리를 막던 문제). 다른 사람은 그대로 경쟁자다."""
    from dataclasses import replace as dc_replace

    import vm.speakers as sp
    from vm.speakers import SpeakerCluster, _normalize, target_from_reference
    engine.cal = dc_replace(engine.cal, chain=True)  # 따라가기는 모델 기준값에서 켜는 것(기본 모델만 켬)

    def cluster(i, v):
        v = _normalize(np.array(v, np.float32))
        return SpeakerCluster(i, v, v, 10.0, ["x.wav"], [("x.wav", 0.0, 1.5)])

    a = cluster(1, [0.75, 0.66, 0.1])   # 참고 녹음과 확실히 같음
    b = cluster(2, [0.3, 0.95, 0.1])    # 참고와는 덜 닮았지만 a와는 확실히 같음
    c = cluster(3, [0.0, -1.0, 0.1])    # 다른 사람
    cal = engine.cal
    assert float(b.centroid @ ME_EMB) < cal.same_person and float(b.centroid @ a.centroid) >= cal.same_person_sure
    ref = compose(("me", 4))
    t = target_from_reference(engine, ref, [a, b, c], augment=False)
    assert any(np.allclose(p, b.centroid) for p in t.wide)  # b도 기준 지문에 든다
    assert len(t.rivals_wide) == 1 and np.allclose(t.rivals_wide[0], c.centroid)
    old = sp.SURE_CHAIN
    try:
        sp.SURE_CHAIN = False
        t0 = target_from_reference(engine, ref, [a, b, c], augment=False)
        assert len(t0.rivals_wide) == 2  # 따라가지 않으면 b가 경쟁자가 된다
    finally:
        sp.SURE_CHAIN = old


def test_auto_discovery_follows_same_person(engine):
    """자동 발견에서 목소리 하나만 골라도, 그 목소리와 확실히 같은 묶음(과 또 그 묶음과 확실히 같은 묶음)은 함께 쓴다.
    다른 사람은 경쟁자로 남고, follow=False면 고른 것만 쓴다."""
    from dataclasses import replace as dc_replace

    from vm.speakers import SpeakerCluster, _normalize, follow_same_person, target_from_clusters

    def cluster(i, v):
        v = _normalize(np.array(v, np.float32))
        return SpeakerCluster(i, v, v, 10.0, ["x.wav"], [("x.wav", 0.0, 1.5)])

    a = cluster(1, [1.0, 0.0, 0.1])
    b = cluster(2, [0.75, 0.66, 0.1])   # a와 확실히 같음
    c = cluster(3, [0.3, 0.95, 0.1])    # a와는 덜 닮았지만 b와 확실히 같음
    d = cluster(4, [0.0, -1.0, 0.1])    # 다른 사람
    sure = engine.cal.same_person_sure
    assert follow_same_person({1}, [a, b, c, d], sure) == {2, 3}
    engine.cal = dc_replace(engine.cal, chain=True)
    t = target_from_clusters([a], [a, b, c, d], engine)
    assert t.info["followed"] == [2, 3]
    assert len(t.rivals_wide) == 1 and np.allclose(t.rivals_wide[0], d.centroid)
    t0 = target_from_clusters([a], [a, b, c, d], engine, follow=False)
    assert t0.info["followed"] == [] and len(t0.rivals_wide) == 3


def test_follow_starts_from_longest_pick_only(engine):
    """여러 목소리를 고르면 가장 길게 말한 목소리에서만 따라간다(작은 섞인 묶음에서 다른 사람 쪽으로 번지지 않게)."""
    from dataclasses import replace as dc_replace

    from vm.speakers import SpeakerCluster, _normalize, target_from_clusters

    def cluster(i, v, sec):
        v = _normalize(np.array(v, np.float32))
        return SpeakerCluster(i, v, v, sec, ["x.wav"], [("x.wav", 0.0, 1.5)])

    main = cluster(1, [1.0, 0.0, 0.1], 60.0)
    frag = cluster(2, [0.8, -0.55, 0.1], 20.0)      # main과 확실히 같음 → 따라감(mixed와는 다름)
    mixed = cluster(3, [0.0, 0.8, 0.6], 5.0)        # 사람이 함께 고른 작은 묶음(다른 사람이 섞임)
    other = cluster(4, [0.0, 0.6, 0.8], 30.0)       # mixed와만 확실히 같은 다른 사람
    engine.cal = dc_replace(engine.cal, chain=True)
    t = target_from_clusters([main, mixed], [main, frag, mixed, other], engine)
    assert t.info["followed"] == [2]
    assert any(np.allclose(r, other.centroid) for r in t.rivals_wide)  # 다른 사람은 경쟁자로 남는다
