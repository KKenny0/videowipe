"""Tests for the WipePlan v1 schema, serialization, validation, and builder."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import videowipe.detect as detect_module
from videowipe.detect import (
    CleanCandidate,
    CleanDetectionResult,
    TextBox,
    refine_temporal_presence,
)
from videowipe.errors import InvalidInputError
from videowipe.plan import (
    JSON_FILENAME,
    MASK_FILENAME,
    Segment,
    Source,
    Track,
    WipePlan,
    build_refined_wipe_plan,
    build_wipe_plan,
    compute_source,
    execution_masks,
    is_temporal,
    load_wipe_plan,
    save_wipe_plan,
    segments_from_presence,
    validate_plan,
)
from videowipe.planning import CleanPlanDraft, finalize

# ── fixtures / helpers ───────────────────────────────────────────────────────

def _candidate(
    cid="c1",
    type_="subtitle",
    bbox=(10, 80, 90, 95),
    default_remove=True,
    mask=None,
    presence_frames=None,
    temporal_sample_indices=None,
    confidence=0.9,
):
    h, w = 100, 100
    if mask is None:
        m = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        m[y1 : y2 + 1, x1 : x2 + 1] = 1
        mask = m
    return SimpleNamespace(
        id=cid,
        type=type_,
        label=f"{type_}-{cid}",
        bbox=tuple(int(v) for v in bbox),
        confidence=confidence,
        default_remove=default_remove,
        mask=mask,
        presence_frames=list(presence_frames or []),
        temporal_sample_indices=list(temporal_sample_indices or []),
    )


def _source(frame_count=60, width=100, height=100, fps=10.0, sha256="a" * 64):
    return Source(
        basename="x.mp4",
        sha256=sha256,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def _write_video(path, width=100, height=100, frames=60, fps=10.0, fill=0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), fill, dtype=np.uint8)
        frame[:, :, 0] = (i * 4) % 256  # vary content so different videos hash differently
        writer.write(frame)
    writer.release()


# ── segment construction ─────────────────────────────────────────────────────

def test_segments_nearest_sample_with_midpoint_switch():
    # samples at 0,20,40,59; present only at sample 20 -> frames 11..30
    segs = segments_from_presence([0, 20, 40, 59], {20}, 60)
    assert segs == [Segment(11, 31)]


def test_segments_multiple_runs_compress_to_half_open_intervals():
    samples = list(range(0, 60, 10))  # 0,10,20,30,40,50
    present = {0, 10, 30, 40}  # active on first two and the 30/40 pair
    segs = segments_from_presence(samples, present, 60)
    # rebuild expected via the same nearest rule is circular; assert structural invariants
    assert all(0 <= s.start < s.end <= 60 for s in segs)
    assert segs == sorted(segs, key=lambda s: s.start)
    assert all(segs[i].end <= segs[i + 1].start for i in range(len(segs) - 1))
    # frame 5 (nearest sample 0, present) is inside a segment; frame 25 (nearest 20/30 boundary -> 20 absent) is outside
    inside = {f for s in segs for f in range(s.start, s.end)}
    assert 5 in inside
    assert 25 not in inside


def test_segments_empty_when_no_samples():
    assert segments_from_presence([], {5}, 60) == []
    assert segments_from_presence([5], {5}, 0) == []


def test_refinement_rechecks_each_active_frame_once_and_splits_a_gap(tmp_path):
    video = tmp_path / "gap.mp4"
    _write_video(video, frames=5)
    candidate = _candidate(presence_frames=[0, 4])

    class Detector:
        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            current = self.calls
            self.calls += 1
            if current == 2:
                return []
            return [TextBox(
                points=np.array([[10, 80], [90, 80], [90, 95], [10, 95]]),
                confidence=1.0,
            )]

    detector = Detector()
    result = CleanDetectionResult(
        [candidate], (100, 100), sample_indices=[0, 4], detector=detector,
    )
    progress = []
    cancellation_checks = []
    warnings = refine_temporal_presence(
        str(video), result, {candidate.id: [Segment(0, 5)]}, 5,
        progress=lambda done, total: progress.append((done, total)),
        check_cancelled=lambda: cancellation_checks.append(detector.calls),
    )

    assert warnings == []
    assert detector.calls == 5
    assert cancellation_checks == [0, 1, 2, 3, 4]
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert candidate.temporal_sample_indices == [0, 1, 2, 3, 4]
    assert candidate.presence_frames == [0, 1, 3, 4]
    plan = build_wipe_plan([candidate], [0, 4], 2, _source(5), (100, 100))
    assert plan.tracks[0].segments == [Segment(0, 2), Segment(3, 5)]


def test_refinement_failure_keeps_frame_and_records_warning(tmp_path):
    video = tmp_path / "failure.mp4"
    _write_video(video, frames=3)
    candidate = _candidate(presence_frames=[0, 2])

    class Detector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("detector unavailable")
            return [TextBox(
                points=np.array([[10, 80], [90, 80], [90, 95], [10, 95]]),
                confidence=1.0,
            )]

    result = CleanDetectionResult(
        [candidate], (100, 100), sample_indices=[0, 2], detector=Detector(),
    )
    warnings = refine_temporal_presence(
        str(video), result, {candidate.id: [Segment(0, 3)]}, 3,
    )

    assert candidate.presence_frames == [0, 2]
    assert any("frame 1" in warning and "keeping frame" in warning for warning in warnings)


def test_track_specific_samples_keep_legacy_candidates_unchanged():
    refined = _candidate(
        cid="refined", presence_frames=[0, 1, 3, 4],
        temporal_sample_indices=[0, 1, 2, 3, 4],
    )
    legacy = _candidate(cid="legacy", presence_frames=[0, 4])
    plan = build_wipe_plan([refined, legacy], [0, 4], 2, _source(5), (100, 100))

    tracks = {track.id: track for track in plan.tracks}
    assert tracks["refined"].segments == [Segment(0, 2), Segment(3, 5)]
    assert tracks["refined"].presence_fraction == pytest.approx(0.8)
    assert tracks["legacy"].segments == [Segment(0, 5)]


def test_clean_candidate_positional_mask_compatibility():
    mask = np.ones((100, 100), dtype=np.uint8)
    candidate = CleanCandidate(
        "c1", "subtitle", "bottom subtitle", (10, 80, 90, 95),
        0.9, 1.0, "test", True, [], [0, 2], mask,
    )

    assert candidate.mask is mask
    plan = build_wipe_plan([candidate], [0, 2], 2, _source(3), (100, 100))
    assert plan.remove_tracks[0].mask is not None


def test_refinement_rejects_premature_decode(monkeypatch):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            return (True, frames.pop(0)) if frames else (False, None)

        def release(self):
            pass

    monkeypatch.setattr(detect_module.cv2, "VideoCapture", lambda _path: Capture())
    candidate = _candidate(presence_frames=[0])
    result = CleanDetectionResult(
        [candidate], (100, 100), sample_indices=[0],
        detector=type("Detector", (), {"detect": lambda self, frame: []})(),
    )

    progress = []
    with pytest.raises(ValueError, match=r"decoded 2 frames; expected 3"):
        refine_temporal_presence(
            "input.mp4", result, {candidate.id: [Segment(0, 3)]}, 3,
            progress=lambda done, total: progress.append((done, total)),
        )
    assert progress == [(1, 3), (2, 3)]


def test_refinement_checks_cancellation_before_detector_work(tmp_path):
    video = tmp_path / "cancel.mp4"
    _write_video(video, frames=1)

    class Detector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1

    detector = Detector()
    candidate = _candidate(presence_frames=[0])
    result = CleanDetectionResult(
        [candidate], (100, 100), sample_indices=[0], detector=detector,
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        refine_temporal_presence(
            str(video), result, {candidate.id: [Segment(0, 1)]}, 1,
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
        )
    assert detector.calls == 0


def test_exact_decode_shares_detector_and_leaves_keep_track_unchanged(monkeypatch):
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    class Capture:
        read_calls = 0

        def isOpened(self):
            return True

        def read(self):
            self.read_calls += 1
            return True, frames.pop(0)

        def release(self):
            pass

    capture = Capture()
    monkeypatch.setattr(detect_module.cv2, "VideoCapture", lambda _path: capture)

    class Detector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1
            return [TextBox(
                points=np.array([[10, 80], [90, 80], [90, 95], [10, 95]]),
                confidence=1.0,
            )]

    detector = Detector()
    first = _candidate(cid="c1", presence_frames=[0, 2])
    second = _candidate(cid="c2", presence_frames=[0, 2])
    keep = _candidate(cid="keep", default_remove=False, presence_frames=[0, 2])
    result = CleanDetectionResult(
        [first, second, keep], (100, 100), sample_indices=[0, 2], detector=detector,
    )

    refine_temporal_presence(
        "input.mp4", result,
        {"c1": [Segment(0, 3)], "c2": [Segment(0, 3)]}, 3,
    )

    assert capture.read_calls == 3
    assert detector.calls == 3
    assert result.detector is detector
    assert first.temporal_sample_indices == [0, 1, 2]
    assert second.temporal_sample_indices == [0, 1, 2]
    assert keep.presence_frames == [0, 2]
    assert keep.temporal_sample_indices == []


def test_fast_plan_bypasses_dense_refinement():
    class Detector:
        calls = 0

        def detect(self, _frame):
            self.calls += 1
            raise AssertionError("fast mode must not refine")

    candidate = _candidate(presence_frames=[0, 2])
    detector = Detector()
    result = CleanDetectionResult(
        [candidate], (100, 100), sample_indices=[0, 2], detector=detector,
    )
    plan = finalize(
        CleanPlanDraft("unused.mp4", result, _source(3), [candidate.id], {}, False),
        refine=False,
    )

    assert detector.calls == 0
    assert plan.remove_tracks[0].segments == [Segment(0, 3)]


def test_clean_plan_draft_review_finalizes_complete_selection():
    subtitle = _candidate(cid="subtitle", type_="subtitle")
    watermark = _candidate(cid="watermark", type_="watermark")
    result = CleanDetectionResult(
        [subtitle, watermark], (100, 100), sample_indices=[0],
    )

    draft = CleanPlanDraft(
        "unused.mp4", result, _source(3), [subtitle.id], {}, False,
    ).for_request(targets=["watermark"])
    plan = finalize(draft, refine=False)

    assert draft.proposed_remove_ids == {"watermark"}
    assert not hasattr(draft.candidates[0], "temporal_sample_indices")
    with pytest.raises(ValueError, match="read-only"):
        draft.candidates[0].mask[0, 0, 0] = 0
    assert {track.id: track.action for track in plan.tracks} == {
        "subtitle": "keep",
        "watermark": "remove",
    }


# ── decision priority + safety rule ──────────────────────────────────────────

def test_safety_rule_keeps_persistent_top_overlay():
    # top overlay: cy=10 < 0.30*100=30, present on all samples -> safety keep
    top = _candidate(cid="logo", bbox=(40, 5, 60, 15), default_remove=True, presence_frames=[0, 20, 40, 59])
    plan = build_wipe_plan([top], [0, 20, 40, 59], 4, _source(), (100, 100))
    assert plan.tracks[0].action == "keep"
    assert "safety:persistent-top-overlay" in plan.tracks[0].decision_reason


def test_explicit_selection_overrides_safety_rule():
    top = _candidate(cid="logo", bbox=(40, 5, 60, 15), default_remove=True, presence_frames=[0, 20, 40, 59])
    plan = build_wipe_plan(
        [top], [0, 20, 40, 59], 4, _source(), (100, 100),
        explicit_remove_ids={"logo"},
    )
    assert plan.tracks[0].action == "remove"
    assert plan.tracks[0].decision_reason.startswith("explicit-selection")


def test_loaded_action_overrides_everything():
    top = _candidate(cid="logo", bbox=(40, 5, 60, 15), default_remove=True, presence_frames=[0, 20, 40, 59])
    plan = build_wipe_plan(
        [top], [0, 20, 40, 59], 4, _source(), (100, 100),
        explicit_remove_ids={"logo"}, loaded_actions={"logo": "keep"},
    )
    assert plan.tracks[0].action == "keep"
    assert plan.tracks[0].decision_reason.startswith("loaded-plan")


def test_default_remove_applied_when_no_safety_or_explicit():
    bottom = _candidate(cid="sub", bbox=(10, 80, 90, 95), default_remove=True, presence_frames=[0, 20])
    scene = _candidate(cid="s", bbox=(40, 40, 60, 60), default_remove=False, type_="scene_text")
    plan = build_wipe_plan([bottom, scene], [0, 20, 40, 59], 4, _source(), (100, 100))
    actions = {t.id: t.action for t in plan.tracks}
    assert actions == {"sub": "remove", "s": "keep"}


def test_no_presence_evidence_uses_full_video_segment_and_warns():
    region = _candidate(cid="r1", bbox=(0, 0, 99, 10), default_remove=True, presence_frames=[])
    plan = build_wipe_plan([region], [0, 20, 40, 59], 4, _source(60), (100, 100))
    t = plan.tracks[0]
    assert t.segments == [Segment(0, 60)]
    assert t.full_video
    assert any("no per-frame presence evidence" in w for w in plan.warnings)


def test_coarse_temporal_resolution_adds_warning():
    # max gap 40 frames at fps=10 -> 4s > 2s threshold
    plan = build_wipe_plan(
        [_candidate(presence_frames=[0])], [0, 40], 2, _source(60, fps=10.0), (100, 100),
    )
    assert any("coarse temporal resolution" in w for w in plan.warnings)
    assert plan.temporal_resolution.max_boundary_error_frames == 20


# ── round-trip + byte stability ──────────────────────────────────────────────

def _plan_with_two_tracks():
    a = _candidate(cid="c1", bbox=(10, 80, 90, 95), presence_frames=[0, 20, 40, 59])
    b = _candidate(cid="c2", bbox=(40, 5, 60, 15), presence_frames=[0, 20, 40, 59])  # top -> keep
    return build_wipe_plan(
        [a, b], [0, 20, 40, 59], 4, _source(60),
        (100, 100), request={"intent": "remove subtitles"},
    )


def test_save_load_roundtrip_preserves_plan_and_masks(tmp_path):
    plan = _plan_with_two_tracks()
    json_path, npz_path = save_wipe_plan(plan, str(tmp_path))
    assert os.path.basename(json_path) == JSON_FILENAME
    assert os.path.basename(npz_path) == MASK_FILENAME

    loaded = load_wipe_plan(json_path)
    assert loaded.schema_version == 1
    assert loaded.kind == "wipe_plan"
    assert {t.id for t in loaded.tracks} == {"c1", "c2"}
    by_id = {t.id: t for t in loaded.tracks}
    assert by_id["c1"].action == "remove"
    assert by_id["c2"].action == "keep"
    # masks pixel-identical after round-trip
    for orig, got in zip(plan.tracks, loaded.tracks):
        assert np.array_equal(np.asarray(orig.mask).squeeze(), np.asarray(got.mask).squeeze())


def test_signed_mask_roundtrip_preserves_execution_projection(tmp_path):
    plan = _plan_with_two_tracks()
    plan.tracks[0].mask = np.array(
        [[-1 if (row + column) % 2 else 2 for column in range(100)] for row in range(100)],
        dtype=np.int16,
    )
    before = execution_masks(plan)

    json_path, _ = save_wipe_plan(plan, str(tmp_path))
    loaded = load_wipe_plan(json_path)

    assert all(np.array_equal(a, b) for a, b in zip(before, execution_masks(loaded)))


def test_legacy_refined_plan_entrypoint_delegates_with_loaded_actions(monkeypatch):
    import videowipe.planning as planning

    result = SimpleNamespace(
        candidates=[_candidate(cid="c1", presence_frames=[0, 20, 40, 59])],
        sample_indices=[0, 20, 40, 59],
        frame_shape=(100, 100),
    )
    calls = []
    monkeypatch.setattr(
        planning,
        "refine_temporal_presence",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["refined"],
    )

    plan = build_refined_wipe_plan(
        "x.mp4",
        result,
        _source(),
        refine=True,
        loaded_actions={"c1": "keep"},
        progress="progress",
        check_cancelled="cancel",
    )

    assert plan.tracks[0].action == "keep"
    assert plan.tracks[0].decision_reason.startswith("loaded-plan:keep")
    assert plan.warnings[-1] == "refined"
    assert calls[0][1] == {"progress": "progress", "check_cancelled": "cancel"}


def test_save_rejects_duplicate_mask_keys(tmp_path):
    plan = _plan_with_two_tracks()
    plan.tracks[1].mask_key = plan.tracks[0].mask_key

    with pytest.raises(InvalidInputError, match="duplicate mask_key"):
        save_wipe_plan(plan, str(tmp_path))
    assert not tmp_path.joinpath(MASK_FILENAME).exists()


def test_roundtrip_is_byte_stable(tmp_path):
    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    j1 = (tmp_path / JSON_FILENAME).read_bytes()
    n1 = (tmp_path / MASK_FILENAME).read_bytes()

    other = tmp_path / "again"
    loaded = load_wipe_plan(str(tmp_path / JSON_FILENAME))
    save_wipe_plan(loaded, str(other))
    j2 = (other / JSON_FILENAME).read_bytes()
    n2 = (other / MASK_FILENAME).read_bytes()

    assert j1 == j2
    assert n1 == n2


def test_npz_loaded_with_allow_pickle_false(tmp_path):
    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    npz_path = str(tmp_path / MASK_FILENAME)
    # object arrays would force pickle; confirm our NPZ loads cleanly with the guard
    with np.load(npz_path, allow_pickle=False) as npz:
        assert set(npz.files) == {"c1", "c2"}
        assert npz["c1"].dtype == np.uint8


# ── validation rejections ────────────────────────────────────────────────────

def _valid_plan(**overrides):
    plan = _plan_with_two_tracks()
    if "tracks" in overrides:
        plan.tracks = overrides["tracks"]
    return plan


def test_validate_rejects_wrong_kind():
    plan = _valid_plan()
    plan.kind = "something_else"
    with pytest.raises(InvalidInputError, match="kind"):
        validate_plan(plan)


def test_validate_rejects_wrong_schema_version():
    plan = _valid_plan()
    plan.schema_version = 2
    with pytest.raises(InvalidInputError, match="schema_version"):
        validate_plan(plan)


def test_validate_rejects_duplicate_track_id():
    plan = _valid_plan()
    plan.tracks[1].id = plan.tracks[0].id
    with pytest.raises(InvalidInputError, match="duplicate track id"):
        validate_plan(plan)


def test_validate_rejects_bad_action():
    plan = _valid_plan()
    plan.tracks[0].action = "delete"
    with pytest.raises(InvalidInputError, match="action"):
        validate_plan(plan)


def test_validate_rejects_inverted_bbox():
    plan = _valid_plan()
    plan.tracks[0].bbox = (90, 80, 10, 95)
    with pytest.raises(InvalidInputError, match="inverted"):
        validate_plan(plan)


def test_validate_rejects_wrong_mask_shape():
    plan = _valid_plan()
    plan.tracks[0].mask = np.zeros((50, 50), dtype=np.uint8)  # source is 100x100
    with pytest.raises(InvalidInputError, match="mask shape"):
        validate_plan(plan)


def test_validate_rejects_float_mask_dtype():
    plan = _valid_plan()
    plan.tracks[0].mask = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises(InvalidInputError, match="dtype"):
        validate_plan(plan)


def test_validate_rejects_out_of_bounds_segment():
    plan = _valid_plan()
    plan.tracks[0].segments = [Segment(0, 9999)]
    with pytest.raises(InvalidInputError, match="segment"):
        validate_plan(plan)


def test_validate_rejects_overlapping_segments():
    plan = _valid_plan()
    plan.tracks[0].segments = [Segment(0, 30), Segment(20, 40)]
    with pytest.raises(InvalidInputError, match="non-overlapping"):
        validate_plan(plan)


def test_validate_rejects_unsorted_segments():
    plan = _valid_plan()
    plan.tracks[0].segments = [Segment(20, 40), Segment(0, 10)]
    with pytest.raises(InvalidInputError, match="non-overlapping"):
        validate_plan(plan)


def test_validate_requires_remove_when_executing():
    plan = _valid_plan()
    for t in plan.tracks:
        t.action = "keep"
    with pytest.raises(InvalidInputError, match="no remove track"):
        validate_plan(plan, require_remove=True)


def test_load_rejects_asset_sha_mismatch(tmp_path):
    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    # corrupt the NPZ
    with open(tmp_path / MASK_FILENAME, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(InvalidInputError, match="sha256 mismatch"):
        load_wipe_plan(str(tmp_path / JSON_FILENAME))


def test_load_rejects_mask_key_not_in_npz(tmp_path):
    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    # add a track referencing a missing key directly in JSON
    data = json.loads((tmp_path / JSON_FILENAME).read_text())
    data["tracks"].append({**data["tracks"][0], "id": "ghost", "mask_key": "ghost"})
    (tmp_path / JSON_FILENAME).write_text(json.dumps(data))
    with pytest.raises(InvalidInputError, match="ghost"):
        load_wipe_plan(str(tmp_path / JSON_FILENAME))


def test_load_rejects_path_escape_in_mask_filename(tmp_path):
    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    data = json.loads((tmp_path / JSON_FILENAME).read_text())
    data["mask_asset"]["filename"] = "../evil.npz"
    (tmp_path / JSON_FILENAME).write_text(json.dumps(data))
    # path escape is rejected (by the path-safety guard before the filename check)
    with pytest.raises(InvalidInputError, match="unsafe plan-relative path"):
        load_wipe_plan(str(tmp_path / JSON_FILENAME))


# ── source binding (real video) ──────────────────────────────────────────────

def test_source_roundtrip_matches_original_video(tmp_path):
    video = tmp_path / "a.mp4"
    _write_video(video, frames=40, fps=10.0, fill=10)
    src = compute_source(str(video))
    cand = _candidate(presence_frames=[0, 10, 20, 30])
    plan = build_wipe_plan([cand], [0, 10, 20, 30], 4, src, (src.height, src.width))
    save_wipe_plan(plan, str(tmp_path))
    # loading bound to the same video succeeds
    loaded = load_wipe_plan(str(tmp_path / JSON_FILENAME), video_path=str(video))
    assert loaded.source.sha256 == src.sha256


def test_source_mismatch_rejects_different_video(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _write_video(a, frames=40, fps=10.0, fill=10)
    _write_video(b, frames=40, fps=10.0, fill=200)  # different content -> different sha
    src = compute_source(str(a))
    plan = build_wipe_plan([_candidate(presence_frames=[0])], [0], 1, src, (src.height, src.width))
    save_wipe_plan(plan, str(tmp_path))
    with pytest.raises(InvalidInputError, match="sha256 mismatch"):
        load_wipe_plan(str(tmp_path / JSON_FILENAME), video_path=str(b))


# ── is_temporal ───────────────────────────────────────────────────────────────

def test_is_temporal_detects_segmented_remove_track():
    # remove track present on only some samples -> sub-video segment -> temporal
    cand = _candidate(cid="c1", bbox=(10, 80, 90, 95), presence_frames=[0, 20])
    plan = build_wipe_plan([cand], [0, 20, 40, 59], 4, _source(60), (100, 100))
    assert plan.tracks[0].action == "remove"
    assert is_temporal(plan)


def test_full_video_plan_is_not_temporal():
    region = _candidate(cid="r1", bbox=(0, 0, 99, 10), default_remove=True, presence_frames=[])
    plan = build_wipe_plan([region], [0, 20, 40, 59], 4, _source(60), (100, 100))
    assert not is_temporal(plan)


# ── per-frame prediction helpers (C5) ────────────────────────────────────────

def test_predicted_mask_at_respects_segments():
    from videowipe.plan import predicted_mask_at

    # a genuinely segmented remove track (present on samples 0 and 20 only)
    cand = _candidate(cid="c1", bbox=(10, 80, 90, 95), presence_frames=[0, 20])
    p = build_wipe_plan([cand], [0, 20, 40, 59], 4, _source(60), (100, 100))
    # active around the [0,20] sample region, inactive near frame 50
    assert predicted_mask_at(p, 10).any()
    assert not predicted_mask_at(p, 50).any()


def test_remove_union_mask_is_time_independent():
    from videowipe.plan import remove_union_mask

    cand = _candidate(cid="c1", bbox=(10, 80, 90, 95), presence_frames=[0, 20])
    p = build_wipe_plan([cand], [0, 20, 40, 59], 4, _source(60), (100, 100))
    union = remove_union_mask(p)
    assert union.shape == (100, 100)
    assert union.dtype == bool
    # the union covers the candidate's spatial mask regardless of frame
    assert union[85, 50]


def _temporal_execution_plan(mask_value=1):
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[20:40, 20:40] = mask_value
    return WipePlan(
        kind="wipe_plan",
        schema_version=1,
        source=Source("x.mp4", "a" * 64, 96, 64, 4.0, 60),
        request={},
        temporal_resolution=SimpleNamespace(
            max_gap_frames=15,
            max_gap_seconds=3.75,
            max_boundary_error_frames=7,
        ),
        mask_asset=SimpleNamespace(filename=MASK_FILENAME, sha256=""),
        tracks=[Track(
            id="c1", type="subtitle", label="a", action="remove",
            bbox=(20, 20, 40, 40), confidence=0.9, presence_fraction=0.3,
            decision_reason="x", segments=[Segment(10, 30)], mask_key="c1",
            mask=mask,
        )],
    )


def test_execution_masks_projects_static_and_temporal_masks_consistently():
    static, frame_mask = execution_masks(_temporal_execution_plan(), feather_radius=4)

    assert static.shape == (64, 96, 1)
    assert static.dtype == np.float32
    assert static[30, 30, 0] == 1.0
    assert 0.0 < static[19, 30, 0] < 1.0
    assert frame_mask is not None
    np.testing.assert_array_equal(frame_mask(15), static[:, :, 0])
    assert not frame_mask(0).any()
    assert frame_mask(15) is frame_mask(16)
    assert not frame_mask(15).flags.writeable
    with pytest.raises(ValueError):
        frame_mask(15)[0, 0] = 1


def test_execution_masks_normalizes_255_masks_before_temporal_blend():
    static, frame_mask = execution_masks(_temporal_execution_plan(mask_value=255))

    assert frame_mask is not None
    assert set(np.unique(static)).issubset({0, 1})
    assert set(np.unique(frame_mask(15))).issubset({0, 1})


def test_execution_masks_all_keep_returns_empty_static_without_temporal_projection():
    plan = _temporal_execution_plan()
    plan.tracks[0].action = "keep"

    static, frame_mask = execution_masks(plan, feather_radius=4)

    assert static.shape == (64, 96, 1)
    assert not static.any()
    assert frame_mask is None


def test_execution_masks_rejects_missing_remove_mask():
    plan = _temporal_execution_plan()
    plan.tracks[0].mask = None

    with pytest.raises(InvalidInputError, match="no precise mask"):
        execution_masks(plan)


def test_execution_masks_rejects_empty_remove_segments():
    plan = _temporal_execution_plan()
    plan.tracks[0].segments = []

    with pytest.raises(InvalidInputError, match="at least one segment"):
        execution_masks(plan)


def test_execution_masks_rejects_invalid_mask_channels():
    plan = _temporal_execution_plan()
    plan.tracks[0].mask = np.zeros((64, 96, 0), dtype=np.uint8)

    with pytest.raises(InvalidInputError, match="one channel"):
        execution_masks(plan)


def test_execution_masks_rejects_empty_remove_mask():
    plan = _temporal_execution_plan(mask_value=0)

    with pytest.raises(InvalidInputError, match="mask is empty"):
        execution_masks(plan)


def test_compute_source_rounds_frame_count(monkeypatch, tmp_path):
    """compute_source rounds CAP_PROP_FRAME_COUNT to match read_frame_info (A1).

    Truncating would leave the plan one frame short of STTN's loop bound for
    non-integer frame counts, silently un-inpaintainting the trailing frame.
    """
    from videowipe.plan import compute_source

    _write_video(tmp_path / "v.mp4", frames=10)
    real_cap = cv2.VideoCapture

    class FakeCap:
        def __init__(self, path):
            self._real = real_cap(path)

        def isOpened(self):
            return self._real.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10.6  # fractional codec-metadata value
            return self._real.get(prop)

        def release(self):
            self._real.release()

    monkeypatch.setattr("videowipe.plan.cv2.VideoCapture", FakeCap)
    src = compute_source(str(tmp_path / "v.mp4"))
    assert src.frame_count == 11  # round(10.6), not trunc(10.6)==10


# ── Fix 1: explicit_keep_ids + complete action mapping ──────────────────────

def test_explicit_keep_overrides_default_remove_for_unselected():
    """A default-remove candidate the user did NOT select is kept (Fix 1).

    c2 is bottom-region (not a top overlay) with default_remove=True; only
    explicit_keep keeps it. Before Fix 1 it fell through to default_remove.
    """
    bottom_a = _candidate(cid="c1", bbox=(10, 80, 40, 95), default_remove=True, presence_frames=[0, 20])
    bottom_b = _candidate(cid="c2", bbox=(60, 80, 90, 95), default_remove=True, presence_frames=[0, 20])
    plan = build_wipe_plan(
        [bottom_a, bottom_b], [0, 20, 40, 59], 4, _source(60), (100, 100),
        explicit_remove_ids={"c1"}, explicit_keep_ids={"c2"},
    )
    actions = {t.id: t.action for t in plan.tracks}
    assert actions == {"c1": "remove", "c2": "keep"}
    c2 = next(t for t in plan.tracks if t.id == "c2")
    assert "explicit-keep" in c2.decision_reason


def test_explicit_remove_takes_precedence_over_explicit_keep():
    """When an id is in both sets, remove wins (priority order)."""
    c = _candidate(cid="c1", bbox=(10, 80, 40, 95), default_remove=True, presence_frames=[0])
    plan = build_wipe_plan(
        [c], [0, 20], 2, _source(60), (100, 100),
        explicit_remove_ids={"c1"}, explicit_keep_ids={"c1"},
    )
    assert plan.tracks[0].action == "remove"


def test_no_user_direction_leaves_explicit_sets_empty():
    """Default flow: neither explicit set populated; safety + default decide."""
    top = _candidate(cid="logo", bbox=(40, 5, 60, 15), default_remove=True, presence_frames=[0, 20, 40, 59])
    bottom = _candidate(cid="sub", bbox=(10, 80, 90, 95), default_remove=True, presence_frames=[0, 20])
    plan = build_wipe_plan(
        [top, bottom], [0, 20, 40, 59], 4, _source(60), (100, 100),
        explicit_remove_ids=set(), explicit_keep_ids=set(),
    )
    actions = {t.id: t.action for t in plan.tracks}
    assert actions == {"logo": "keep", "sub": "remove"}  # safety keeps top; default removes bottom


# ── Fix 2: precise masks required on remove tracks at execution ─────────────

def test_remove_track_without_mask_passes_metadata_check():
    """A maskless remove track is fine for metadata inspection (no require_remove)."""
    plan = _plan_with_two_tracks()
    for t in plan.tracks:
        if t.action == "remove":
            t.mask = None
    validate_plan(plan)  # structural only — must not raise


def test_remove_track_without_mask_rejected_for_execution():
    plan = _plan_with_two_tracks()
    for t in plan.tracks:
        if t.action == "remove":
            t.mask = None
    with pytest.raises(InvalidInputError, match="no precise mask"):
        validate_plan(plan, require_remove=True)


def test_union_mask_rejects_missing_remove_mask():
    """Execution raises rather than silently widening a maskless track."""
    plan = _plan_with_two_tracks()
    for t in plan.tracks:
        if t.action == "remove":
            t.mask = None
    with pytest.raises(InvalidInputError, match="no precise mask"):
        execution_masks(plan)


def test_normal_plan_with_precise_masks_passes_execution_check():
    plan = _plan_with_two_tracks()
    validate_plan(plan, require_remove=True)  # must not raise


# ── Fix 3: reject sidecar symlinks escaping the plan directory ──────────────

def test_load_rejects_sidecar_symlink_to_outside(tmp_path):
    """An NPZ replaced by a symlink that escapes the plan dir is rejected."""
    import shutil

    plan = _plan_with_two_tracks()
    save_wipe_plan(plan, str(tmp_path))
    npz = tmp_path / MASK_FILENAME
    outside = tmp_path.parent / "outside_stolen.npz"
    shutil.move(str(npz), str(outside))
    os.symlink(str(outside), str(npz))
    with pytest.raises(InvalidInputError, match="escapes plan directory"):
        load_wipe_plan(str(tmp_path / JSON_FILENAME))


def test_save_rejects_preexisting_escaping_symlink(tmp_path):
    """A pre-existing escaping symlink at the NPZ path blocks save; external file untouched."""
    outside = tmp_path.parent / "external_target.npz"
    outside.write_bytes(b"do-not-overwrite")
    (tmp_path / MASK_FILENAME).symlink_to(str(outside))
    plan = _plan_with_two_tracks()
    with pytest.raises(InvalidInputError, match="escapes plan directory"):
        save_wipe_plan(plan, str(tmp_path))
    assert outside.read_bytes() == b"do-not-overwrite"
