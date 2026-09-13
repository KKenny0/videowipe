"""Local subtitle evidence survives review and deterministic plan replay."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from videowipe.errors import InvalidInputError
from videowipe.plan import (MaskAsset, Segment, Source, TemporalResolution, Track,
                           WipePlan, execution_masks, is_temporal, load_wipe_plan,
                           save_wipe_plan, validate_plan, _spatial_mask)
from videowipe.server.review import compile_review
from videowipe.server.app import _trial_key


def local_plan():
    t = Track('c1', 'subtitle', 'subtitle', 'remove', (0, 0, 19, 9), 1, 1,
              'default', [Segment(0, 6)], 'c1', spatial_segments=[
                  {'start': 0, 'end': 2, 'boxes': [[1, 2, 5, 6]]},
                  {'start': 2, 'end': 4, 'boxes': [[12, 2, 16, 6]]}])
    t.mask = _spatial_mask(t, (10, 20))
    return WipePlan('wipe_plan', 3, Source('x.mp4', 'a'*64, 20, 10, 1, 6), {},
                    TemporalResolution(1, 1, 0), MaskAsset('wipe_plan_masks.npz', ''), [t])


def test_local_masks_roundtrip_review_protection_and_cache_key(tmp_path):
    plan = local_plan()
    assert is_temporal(plan)  # Even though the selected interval is full-video.
    _, frame = execution_masks(plan, 1)
    assert frame(0) is frame(1)
    assert frame(0)[3, 3] == 1 and frame(0)[3, 14] == 0
    assert frame(2)[3, 3] == 0 and frame(2)[3, 14] == 1
    assert not frame(4).any()  # No local evidence must not fall back to the union.
    saved, _ = save_wipe_plan(plan, str(tmp_path))
    restored = load_wipe_plan(saved)
    assert restored.schema_version == 3
    assert all(np.array_equal(frame(i), execution_masks(restored, 1)[1](i)) for i in range(6))
    edited = deepcopy(plan)
    edited.tracks[0].spatial_segments[0]['boxes'], edited.tracks[0].spatial_segments[1]['boxes'] = (
        edited.tracks[0].spatial_segments[1]['boxes'], edited.tracks[0].spatial_segments[0]['boxes'])
    assert np.array_equal(edited.tracks[0].mask, plan.tracks[0].mask)
    assert _trial_key(edited, (0, 6), {}) != _trial_key(plan, (0, 6), {})
    protected, _ = compile_review(deepcopy(plan), protections=[
        {'id': 'p_test', 'bbox': [0, 1, 4, 7], 'segments': [[0, 1]]}])
    assert protected.schema_version == 3
    union, alpha = execution_masks(protected, 1)
    assert alpha(0)[3, 3] == 0 and alpha(1)[3, 3] == 1 and union[3, 3] == 1
    shortened, _ = compile_review(deepcopy(plan), segment_overrides={'c1': [[2, 6]]})
    union, alpha = execution_masks(shortened)
    assert union[3, 3] == 0 and union[3, 14] == 1 and not alpha(4).any()
    manual, _ = compile_review(deepcopy(plan), bbox_overrides={'c1': [0, 1, 5, 7]})
    assert manual.schema_version == 1 and manual.tracks[0].spatial_segments is None
    assert execution_masks(manual)[0][3, 3] == 1


@pytest.mark.parametrize('mutation', [
    lambda p: setattr(p, 'schema_version', 2),
    lambda p: p.tracks[0].spatial_segments[0].update(start=True),
    lambda p: p.tracks[0].spatial_segments[1].update(start=1),
    lambda p: p.tracks[0].spatial_segments[0].update(boxes=[[0, 0, 20, 9]]),
    lambda p: p.tracks[0].spatial_segments[0].update(boxes=[]),
    lambda p: p.tracks[0].mask.fill(1),
])
def test_local_mask_rejects_ambiguous_or_unbounded_evidence(mutation):
    plan = local_plan(); mutation(plan)
    with pytest.raises(InvalidInputError): validate_plan(plan)


def test_dense_detections_are_scoped_to_current_frame_and_subtitle():
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    plan = local_plan(); plan.tracks[0].bbox = (1, 3, 18, 8)
    box = lambda x, y: TextBox(np.array([[x,y],[x+2,y],[x+2,y+1],[x,y+1]]), .9)
    result = SimpleNamespace(frame_shape=(10, 20), candidates=[
        SimpleNamespace(id='c1', type='subtitle', detector_backed=True)],
        sampled_frame_boxes={0: [box(2, 4), box(12, 0)], 1: [box(2, 4)], 2: []})
    _add_spatial_segments(plan, result)
    assert [(r['start'], r['end']) for r in plan.tracks[0].spatial_segments] == [(0, 2)]
    assert not execution_masks(plan)[1](2).any()
    assert len(plan.tracks) == 1


def test_height_stabilization_keeps_current_width_and_stops_at_gaps():
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    plan = local_plan(); plan.source = Source('x.mp4', 'a'*64, 400, 200, 8, 6)
    plan.tracks[0].bbox = (30, 90, 180, 145)
    box = lambda x1,y1,x2,y2: TextBox(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]), .9)
    result = SimpleNamespace(frame_shape=(200, 400), candidates=[
        SimpleNamespace(id='c1', type='subtitle', detector_backed=True)],
        sampled_frame_boxes={0: [box(50,100,150,130)], 1: [box(80,105,130,125)],
                             2: [], 3: [box(80,105,130,125)]})
    _add_spatial_segments(plan, result)
    _, mask = execution_masks(plan)
    assert mask(1)[95,100] == 1  # Recover the briefly clipped top of the glyph.
    assert mask(1)[110,40] == 0 and mask(0)[110,40] == 1  # No width propagation.
    assert not mask(2).any()
    assert mask(3)[95,100] == 0  # A gap breaks height propagation too.


def test_web_actual_mask_replays_local_evidence_and_rejects_stale_revision(tmp_path, monkeypatch):
    import cv2
    from fastapi.testclient import TestClient
    from videowipe.server import app as web
    from videowipe.server.jobs import Job
    plan = local_plan(); save_wipe_plan(plan, str(tmp_path))
    job = Job('a'*32, str(tmp_path/'x.mp4'), str(tmp_path),
              reviewed_plan_path=str(tmp_path/'wipe_plan.json'), compiled_review_revision=2,
              review_revision=2, final_plan_ready=True)
    monkeypatch.setattr(web, '_jobs_root', lambda: str(tmp_path))
    monkeypatch.setattr(web, 'get_job', lambda _: job)
    monkeypatch.setattr(web, '_get_engine', lambda: SimpleNamespace(
        _task_impl=SimpleNamespace(feather_radius=1)))
    client = TestClient(web.app)
    for index, point in ((0, (3, 3)), (2, (3, 14))):
        response = client.get(f'/jobs/{job.id}/mask?frame_index={index}&revision=2')
        assert response.status_code == 200, response.text
        actual = cv2.imdecode(np.frombuffer(response.content, np.uint8), 0)
        assert actual[point] == 255
        assert np.array_equal(actual, np.rint(execution_masks(plan, 1)[1](index)*255).astype(np.uint8))
    assert client.get(f'/jobs/{job.id}/mask?frame_index=0&revision=1').status_code == 409


def test_interpolated_boundary_frames_are_checked_but_known_empty_frames_stay_empty(tmp_path):
    import cv2
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    plan = local_plan()
    video = tmp_path/'boundary.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 1, (20, 10))
    for _ in range(6): writer.write(np.zeros((10, 20, 3), np.uint8))
    writer.release()
    box = TextBox(np.array([[3,3],[8,3],[8,6],[3,6]]), .9)
    calls = []
    result = SimpleNamespace(frame_shape=(10,20), candidates=[
        SimpleNamespace(id='c1', type='subtitle', detector_backed=True)],
        sampled_frame_boxes={0:[box],1:[box],3:[]},
        detector=SimpleNamespace(detect=lambda frame: calls.append(frame.shape) or [box]))
    _add_spatial_segments(plan, result, video_path=str(video))
    assert len(calls) == 3  # Only missing frames 2, 4 and 5.
    mask = execution_masks(plan)[1]
    assert mask(2)[4,5] == 1 and mask(4)[4,5] == 1
    assert not mask(3).any()  # Observed negative evidence is never filled in.
