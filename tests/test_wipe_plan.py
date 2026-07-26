"""Tests for the WipePlan v1 schema, serialization, validation, and builder."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.errors import InvalidInputError
from videowipe.plan import (
    JSON_FILENAME,
    MASK_FILENAME,
    Segment,
    Source,
    build_wipe_plan,
    compute_source,
    is_temporal,
    load_wipe_plan,
    save_wipe_plan,
    segments_from_presence,
    validate_plan,
)

# ── fixtures / helpers ───────────────────────────────────────────────────────

def _candidate(
    cid="c1",
    type_="subtitle",
    bbox=(10, 80, 90, 95),
    default_remove=True,
    mask=None,
    presence_frames=None,
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
