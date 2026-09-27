"""Local subtitle evidence survives review and deterministic plan replay."""
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe.detect import TextBox
from videowipe.errors import InvalidInputError
from videowipe.plan import (MaskAsset, Segment, Source, TemporalResolution, Track,
                           WipePlan, execution_masks, is_temporal, load_wipe_plan,
                           save_wipe_plan, validate_plan, _spatial_mask)
from videowipe.planning import _add_spatial_segments, refine_temporal_presence
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



def test_descender_outline_requires_local_white_glyph_evidence():
    from videowipe.planning import _descender_extensions
    frame = np.full((40,80,3), 50, np.uint8)
    frame[15:25,30:35] = 0  # Narrow black descender crosses lower core y=22.
    frame[15:20,31:34] = 255
    frame[15:25,50:55] = 0  # Equally dark background without white glyph.
    boxes = [[10,10,70,22]]
    extra = _descender_extensions(frame, boxes)
    assert extra == [(30,23,34,24)]
    frame[21:23,20:45] = 0  # Joined letter outlines must not hide a narrow tail.
    assert _descender_extensions(frame, boxes) == [(30,23,34,24)]
    frame[15:20,31:34] = 0
    assert _descender_extensions(frame, boxes) == []


def test_descender_evidence_reaches_saved_plan_core(tmp_path):
    import cv2
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    video = tmp_path/'outline.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 25, (80,40))
    assert writer.isOpened()
    frame = np.full((40,80,3), 50, np.uint8)
    frame[15:25,30:35] = 0
    frame[15:20,31:34] = 255
    for _ in range(6):
        writer.write(frame)
    writer.release()
    plan = local_plan(); plan.source = Source('outline.avi','a'*64,80,40,25,6)
    plan.tracks[0].bbox = (10,10,70,22)
    box = TextBox(np.array([[30,14],[34,14],[34,18],[30,18]]), .9)
    evidence = {i:[] for i in range(6)}; evidence[0] = [box]
    result = SimpleNamespace(frame_shape=(40,80), candidates=[
        SimpleNamespace(id='c1',type='subtitle',detector_backed=True)],sampled_frame_boxes=evidence)
    _add_spatial_segments(plan,result,video_path=str(video))
    save_wipe_plan(plan,str(tmp_path/'plan'))
    replay = load_wipe_plan(str(tmp_path/'plan/wipe_plan.json'))
    alpha = execution_masks(replay,4)[1]
    assert alpha(0)[24,32] == 1
    assert not alpha(1).any()


@pytest.mark.parametrize('color', [(0,255,255), (255,255,0), (0,0,255), (110,110,110)])
def test_descender_recovery_does_not_treat_colored_text_as_white(color):
    from videowipe.planning import _descender_extensions
    frame = np.full((40,80,3),50,np.uint8)
    frame[15:25,30:35] = 0
    frame[15:20,31:34] = color
    assert _descender_extensions(frame, [[10,10,70,22]]) == []


def _gap_plan(segments):
    plan = WipePlan('wipe_plan', 3, Source('x.mp4', 'a'*64, 400, 200, 8, 9), {},
                    TemporalResolution(1, 1, 0), MaskAsset('wipe_plan_masks.npz', ''), [
        Track('c1', 'subtitle', 'subtitle', 'remove', (30, 90, 180, 145), 1, 1,
              'default', segments, 'c1')])
    return plan


def _gap_result(sampled):
    return SimpleNamespace(frame_shape=(200, 400), candidates=[
        SimpleNamespace(id='c1', type='subtitle', detector_backed=True)],
        sampled_frame_boxes=sampled)


def test_positive_short_gap_bridges():
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    plan = _gap_plan([Segment(0, 2), Segment(5, 9)])
    def tb(x1,y1,x2,y2): return TextBox(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]), .9)
    sampled = {0: [tb(50,100,150,130)], 1: [tb(50,100,150,130)],
               2: [tb(50,100,150,130)], 3: [tb(50,100,150,130)], 4: [tb(50,100,150,130)],
               5: [tb(52,100,148,130)], 6: [tb(52,100,148,130)],
               7: [tb(52,100,148,130)], 8: [tb(52,100,148,130)]}
    _add_spatial_segments(plan, _gap_result(sampled))
    assert plan.tracks[0].segments == [Segment(0, 9)]  # The spurious gap merged.
    mask = execution_masks(plan)[1]
    assert mask(3)[110, 100] == 1  # Inherited line geometry covers the hole.
    assert mask(3)[110, 20] == 0  # Only the flanked line, never the full band.


def test_silent_short_gap_stays_open_when_flanks_disagree():
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    plan = _gap_plan([Segment(0, 2), Segment(5, 9)])
    def tb(x1,y1,x2,y2): return TextBox(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]), .9)
    sampled = {0: [tb(50,100,150,130)], 1: [tb(50,100,150,130)],
               2: [], 3: [], 4: [],
               5: [tb(250,100,350,130)], 6: [tb(250,100,350,130)],
               7: [tb(250,100,350,130)], 8: [tb(250,100,350,130)]}
    _add_spatial_segments(plan, _gap_result(sampled))
    assert plan.tracks[0].segments == [Segment(0, 2), Segment(5, 9)]
    assert not execution_masks(plan)[1](3).any()


def test_leading_glyph_extension_recovers_clipped_first_letter():
    from videowipe.planning import _leading_extensions
    frame = np.full((40,80,3), 50, np.uint8)
    frame[15:25,42:47] = 255  # Leading glyph missed by the detector box.
    boxes = [[50,10,70,22]]
    assert _leading_extensions(frame, boxes, 10) == [(42,10,46,22)]
    frame[:, :] = 240  # White-on-white flash: no dark contrast, never qualifies.
    assert _leading_extensions(frame, boxes, 10) == []
    frame = np.full((40,80,3), 50, np.uint8)
    frame[15:25,42:47] = (0,255,255)  # Colored glyph is not neutral white.
    assert _leading_extensions(frame, boxes, 10) == []
    frame[15:25,42:47] = 255
    assert _leading_extensions(frame, boxes, 45) == []  # Past the track bbox edge.


def test_leading_glyph_evidence_reaches_saved_plan_core(tmp_path):
    import cv2
    from videowipe.detect import TextBox
    from videowipe.planning import _add_spatial_segments
    video = tmp_path/'leading.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 25, (80,40))
    assert writer.isOpened()
    frame = np.full((40,80,3), 50, np.uint8)
    frame[14:19,30:35] = 255  # The leading glyph.
    for _ in range(6):
        writer.write(frame)
    writer.release()
    plan = _gap_plan([Segment(0, 6)])
    plan.source = Source('leading.avi','a'*64,80,40,25,6)
    plan.tracks[0].bbox = (10,10,70,22)
    box = TextBox(np.array([[40,14],[60,14],[60,18],[40,18]]), .9)
    result = SimpleNamespace(frame_shape=(40,80), candidates=[
        SimpleNamespace(id='c1',type='subtitle',detector_backed=True)],
        sampled_frame_boxes={i:[box] for i in range(6)})
    _add_spatial_segments(plan,result,video_path=str(video))
    save_wipe_plan(plan,str(tmp_path/'plan'))
    replay = load_wipe_plan(str(tmp_path/'plan/wipe_plan.json'))
    alpha = execution_masks(replay,1)[1]
    for i in range(6):
        assert alpha(i)[16,32] == 1  # The clipped leading glyph stays covered.


def test_white_descender_with_dark_edge_is_not_left_in_feather():
    from videowipe.planning import _descender_extensions
    frame = np.full((40,80,3),110,np.uint8)
    frame[15:25,30:35] = 255
    frame[22:24,30:35] = 150  # Antialiasing must still connect to the bright body.
    frame[15:20,29] = 0  # Dark edge supports the connected white tail.
    extra = _descender_extensions(frame, [[10,10,70,22]])
    assert any(x1 <= 32 <= x2 and y1 <= 24 <= y2 for x1,y1,x2,y2 in extra)
    frame[15:20,29] = 110
    assert _descender_extensions(frame, [[10,10,70,22]]) == []
    frame[15:25,10:71] = 255  # Broad bright background is not a narrow glyph tail.
    frame[15:20,10] = 0
    assert _descender_extensions(frame, [[10,10,70,22]]) == []


def test_white_tail_uses_glyph_evidence_above_shrinking_lower_edge():
    from videowipe.planning import _descender_extensions
    frame = np.full((80,100,3),110,np.uint8)
    frame[20:46,40:45] = 255
    frame[20:26,39] = 0  # Contrast is above the old eight-pixel strip.
    frame[40:46,45:80] = 150  # Weak background must not absorb the glyph.
    for bottom in (43,40):
        extra = _descender_extensions(frame, [[10,10,90,bottom]])
        assert any(a <= 42 <= c and b <= 45 <= d for a,b,c,d in extra)
        assert all(c < 50 for a,b,c,d in extra)


@pytest.mark.parametrize("mode", ["blank", "error", "unreadable"])
def test_short_gap_requires_positive_frame_evidence(tmp_path, monkeypatch, mode):
    import cv2

    from videowipe import planning
    from videowipe.detect import TextBox
    video = tmp_path / "gap.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 8, (400, 200))
    assert writer.isOpened()
    for _ in range(9):
        writer.write(np.zeros((200, 400, 3), np.uint8))
    writer.release()
    box = TextBox(np.array([[50,100],[150,100],[150,130],[50,130]]), .9)
    result = _gap_result({i: [box] for i in (0,1,5,6,7,8)})
    def detect(frame):
        # Contrast-recovery variants of a content-free frame stay uniform, so
        # "no local structure anywhere" still proves the video carries no text.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert np.ptp(gray) == 0  # Uniform input: still the content-free video.
        if mode == "error":
            raise RuntimeError("detector unavailable")
        return []
    result.detector = SimpleNamespace(detect=detect)
    if mode == "unreadable":
        real_capture = cv2.VideoCapture
        class Reader:
            def __init__(self, path): self.reader = real_capture(path)
            def isOpened(self): return self.reader.isOpened()
            def set(self, prop, value): return self.reader.set(prop, value)
            def read(self):
                if 2 <= self.reader.get(cv2.CAP_PROP_POS_FRAMES) <= 4:
                    return False, None
                return self.reader.read()
            def release(self): self.reader.release()
        monkeypatch.setattr(cv2, "VideoCapture", Reader)
        # Outline recovery's sequential decoder correctly raises on unreadable
        # frames; inspect the completed spatial decision before that stage.
        monkeypatch.setattr(planning, "_recover_descenders", lambda *args: None)
    plan = _gap_plan([Segment(0,2), Segment(5,9)])
    planning._add_spatial_segments(plan, result, video_path=str(video))
    assert plan.tracks[0].segments == [Segment(0,2), Segment(5,9)]
    mask = execution_masks(plan)[1]
    assert mask(1).any()
    assert all(not mask(i).any() for i in (2,3,4))
    if mode == "error":
        assert len(plan.warnings) == 3


class _VariantSignaledDetector:
    """Dense detector that only sees text in contrast-restored band crops.

    The synthetic flash frame (background 250, glyphs 180) is invisible to the
    "normal" pass but every contrast restoration leaves a distinct signature:
    highpass re-centers the background at 128, darkened (gamma 2.2) drops the
    background to ~244 with a wide spread. Returned boxes use the crop-local
    coordinates of the band crop around bbox (30, 90, 180, 145); the recovered
    glyph sits at frame coords (60, 100)-(120, 130).
    """

    def __init__(self, variants, box, raise_on_normal=False):
        self.variants = set(variants)
        self.box = box
        self.raise_on_normal = raise_on_normal
        self.normal_calls = 0
        self._manual_only = False

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))
        spread = float(np.ptp(gray))
        if frame.shape[:2] == (200, 400):
            self.normal_calls += 1
            if self.raise_on_normal:
                raise RuntimeError("detector unavailable")
            return []
        restored = median < 248 or spread > 80
        if not restored:
            return []
        if self._flip_on_restore:
            self._manual_only = True
        if 'highpass' in self.variants and 120 <= median <= 140:
            return [self._text_box()]
        if 'darkened' in self.variants and 232 <= median <= 247 and spread >= 100:
            return [self._text_box()]
        return []

    _flip_on_restore = False

    def _text_box(self):
        x1, y1, x2, y2 = self.box
        return TextBox(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64), .9)


def _flash_video(path, frames=6, width=400, height=200):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 8,
                             (width, height))
    assert writer.isOpened()
    for _ in range(frames):
        frame = np.full((height, width, 3), 250, np.uint8)
        frame[100:131, 60:121] = 180
        writer.write(frame)
    writer.release()


def _flash_result(detector, sampled=None):
    return SimpleNamespace(
        frame_shape=(200, 400),
        sample_indices=[0],
        candidates=[SimpleNamespace(id='c1', type='subtitle', detector_backed=True,
                                    bbox=(30, 90, 180, 145),
                                    temporal_sample_indices=[0], presence_frames=[0])],
        sampled_frame_boxes=sampled if sampled is not None else {},
        detector=detector,
    )


def _flash_plan():
    plan = _gap_plan([Segment(0, 6)])
    plan.tracks[0].bbox = (30, 90, 180, 145)
    return plan


def test_white_flash_recovery_restores_per_frame_presence(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (48, 24, 108, 54))
    result = _flash_result(detector)
    warnings = refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert warnings == []
    presence = set(result.candidates[0].presence_frames)
    assert presence == set(range(6))  # Every flash frame has current-frame evidence.
    assert detector.normal_calls == 6  # Normal pass ran first on every frame.
    for index in range(6):
        assert result.sampled_frame_boxes[index], index
        assert detector._manual_only is False  # Retry never flips shared state.


def test_single_variant_recovery_is_not_evidence(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass'}, (48, 24, 108, 54))
    result = _flash_result(detector)
    refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert result.candidates[0].presence_frames == []


def test_recovery_outside_candidate_bbox_is_not_evidence(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (60, 0, 108, 10))
    result = _flash_result(detector)
    refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert result.candidates[0].presence_frames == []


def test_detector_error_never_gets_a_second_chance(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (60, 100, 120, 130),
                                        raise_on_normal=True)
    result = _flash_result(detector)
    warnings = refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert result.candidates[0].presence_frames == []
    assert detector.normal_calls == 6  # One attempt per frame; never retried.
    assert len(warnings) == 6


def test_recovery_covers_cached_sampled_frames_too(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (48, 24, 108, 54))
    result = _flash_result(detector, sampled={0: []})
    refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert 0 in result.candidates[0].presence_frames


def test_recovered_evidence_reaches_spatial_plan(tmp_path):
    video = tmp_path / 'flash.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (48, 24, 108, 54))
    result = _flash_result(detector)
    refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    plan = _flash_plan()
    _add_spatial_segments(plan, result, video_path=str(video))
    mask = execution_masks(plan, 1)[1]
    for index in range(6):
        assert mask(index)[115, 90] == 1  # The recovered line is really wiped.
        assert mask(index)[115, 20] == 0  # Only the evidenced line, never the band.


def test_gap_probe_recovers_flash_hidden_subtitle(tmp_path):
    # The coarse sample landed on the flash peak, so the gap frames were never
    # refined. The gap probe's contrast-recovery retry must still close the
    # split from per-frame positive evidence only.
    video = tmp_path / 'flash-gap.avi'
    _flash_video(video)
    detector = _VariantSignaledDetector({'highpass', 'darkened'}, (48, 24, 108, 54))
    detector.normal_detects_empty = True
    plan = _gap_plan([Segment(0, 2), Segment(5, 9)])
    result = SimpleNamespace(
        frame_shape=(200, 400), sample_indices=[0],
        candidates=[SimpleNamespace(id='c1', type='subtitle', detector_backed=True,
                                    bbox=(30, 90, 180, 145),
                                    temporal_sample_indices=[], presence_frames=[])],
        sampled_frame_boxes={0: [], 1: [], 5: [], 6: [], 7: [], 8: []},
        detector=detector,
    )

    def normal_empty(frame):
        return []

    # Frames 0-1 and 5-8 carry the coarse presence; the probe only needs to
    # decide frames 2-4, whose normal pass finds nothing.
    detector.detect = detector.detect  # variant-signaled behavior stands
    _add_spatial_segments(plan, result, video_path=str(video))
    assert plan.tracks[0].segments == [Segment(0, 9)]
    mask = execution_masks(plan)[1]
    for index in (2, 3, 4):
        assert mask(index)[115, 90] == 1
        assert mask(index)[115, 20] == 0


def test_boundary_zone_probe_extends_late_subtitle_start(tmp_path):
    # A coarse midpoint boundary placed the segment start three frames after
    # the subtitle actually appears. The uncertainty-zone probe must extend
    # the segment across contiguous frames with per-frame positive evidence.
    video = tmp_path / 'late.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 8, (400, 200))
    assert writer.isOpened()
    for i in range(8):
        frame = np.full((200, 400, 3), 60, np.uint8)
        if 1 <= i <= 5:
            frame[100:131, 60:121] = 240
        writer.write(frame)
    writer.release()
    plan = _gap_plan([Segment(3, 6)])
    plan.source = Source('late.avi', 'a' * 64, 400, 200, 8, 8)
    plan.tracks[0].bbox = (30, 90, 180, 145)
    box = TextBox(np.array([[60, 100], [120, 100], [120, 130], [60, 130]], np.float64), .9)
    result = SimpleNamespace(
        frame_shape=(200, 400), sample_indices=[3],
        candidates=[SimpleNamespace(id='c1', type='subtitle', detector_backed=True,
                                    bbox=(30, 90, 180, 145),
                                    temporal_sample_indices=[3, 4, 5], presence_frames=[3, 4, 5])],
        sampled_frame_boxes={3: [box], 4: [box], 5: [box]},
        detector=SimpleNamespace(detect=lambda frame: [box] if np.ptp(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 100 else []),
    )
    _add_spatial_segments(plan, result, video_path=str(video))
    assert plan.tracks[0].segments == [Segment(1, 6)]  # Extended to the real start.
    mask = execution_masks(plan)[1]
    for index in (1, 2, 3):
        assert mask(index)[115, 90] == 1
        assert mask(index)[115, 20] == 0
    assert not mask(0).any()  # The empty frame before the text stays untouched.


def test_boundary_zone_probe_stops_on_empty_frames(tmp_path):
    # Without positive evidence the boundary never moves into the zone.
    video = tmp_path / 'late-empty.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 8, (400, 200))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((200, 400, 3), 60, np.uint8))
    writer.release()
    plan = _gap_plan([Segment(3, 6)])
    plan.source = Source('late-empty.avi', 'a' * 64, 400, 200, 8, 8)
    plan.tracks[0].bbox = (30, 90, 180, 145)
    box = TextBox(np.array([[60, 100], [120, 100], [120, 130], [60, 130]], np.float64), .9)
    result = SimpleNamespace(
        frame_shape=(200, 400), sample_indices=[3],
        candidates=[SimpleNamespace(id='c1', type='subtitle', detector_backed=True,
                                    bbox=(30, 90, 180, 145),
                                    temporal_sample_indices=[3, 4, 5], presence_frames=[3, 4, 5])],
        sampled_frame_boxes={3: [box], 4: [box], 5: [box]},
        detector=SimpleNamespace(detect=lambda frame: [box] if frame.max() > 200 else []),
    )
    _add_spatial_segments(plan, result, video_path=str(video))
    assert plan.tracks[0].segments == [Segment(3, 6)]
    assert not execution_masks(plan)[1](1).any()


def test_failed_frame_stays_negative_through_spatial_recovery(tmp_path):
    video = tmp_path / 'failed.avi'
    _flash_video(video, frames=9)
    class Detector(_VariantSignaledDetector):
        def detect(self, frame):
            if frame.shape[:2] == (200, 400) and self.normal_calls == 2:
                self.normal_calls += 1
                raise RuntimeError('failed frame')
            return super().detect(frame)
    result = _flash_result(Detector({'highpass', 'darkened'}, (48, 24, 108, 54)))
    refine_temporal_presence(str(video), result, {'c1': [Segment(0, 9)]}, 9)
    plan = _gap_plan([Segment(0, 2), Segment(3, 9)])
    _add_spatial_segments(plan, result, video_path=str(video))
    assert not execution_masks(plan)[1](2).any()


def test_recovery_failure_is_negative_evidence(tmp_path):
    video = tmp_path / 'failed.avi'
    _flash_video(video)
    class Detector:
        def detect(self, frame):
            if frame.shape[:2] == (200, 400):
                return []
            raise RuntimeError('crop detector failed')
    result = _flash_result(Detector())
    warnings = refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    assert len(warnings) == 6
    assert result.failed_frame_indices == set(range(6))
    assert result.candidates[0].presence_frames == []


def test_contrast_consensus_requires_meaningful_overlap():
    from videowipe.detect import _rects_agree
    assert _rects_agree((0, 0, 100, 30), (2, 1, 98, 29))
    assert not _rects_agree((0, 0, 100, 30), (100, 30, 200, 60))
    assert not _rects_agree((0, 0, 100, 30), (99, 29, 199, 59))
    assert not _rects_agree((0, 0, 100, 30), (101, 31, 201, 61))


def test_recovery_checks_later_variant_for_other_confirmed_glyphs():
    from videowipe.detect import _contrast_recovered_boxes
    left = TextBox(np.array([[40, 25], [80, 25], [80, 50], [40, 50]]), .9)
    right = TextBox(np.array([[100, 25], [140, 25], [140, 50], [100, 50]]), .9)

    class Detector:
        calls = 0

        def detect(self, frame):
            boxes = ([left, right], [left], [right], [], [], [])[self.calls]
            self.calls += 1
            return boxes

    detector = Detector()
    recovered = _contrast_recovered_boxes(
        detector, np.zeros((200, 400, 3), np.uint8), (30, 90, 180, 145),
    )
    assert len(recovered) == 2  # Both glyphs have two independent confirmations.
    assert detector.calls == 6


@pytest.mark.parametrize('phase', ['refine', 'gap'])
def test_partial_normal_box_still_recovers_confirmed_text(tmp_path, phase):
    video = tmp_path / 'partial.avi'
    _flash_video(video, frames=9)
    fragment = TextBox(np.array([[60, 100], [65, 100], [65, 110], [60, 110]]), .9)

    class Detector(_VariantSignaledDetector):
        def detect(self, frame):
            if frame.shape[:2] == (200, 400):
                return [fragment]
            return super().detect(frame)

    result = _flash_result(Detector({'highpass', 'darkened'}, (48, 24, 108, 54)))
    if phase == 'refine':
        refine_temporal_presence(str(video), result, {'c1': [Segment(0, 9)]}, 9)
    else:
        result.sampled_frame_boxes = {i: [fragment] for i in range(9)}
        _add_spatial_segments(_gap_plan([Segment(0, 2), Segment(5, 9)]), result,
                              video_path=str(video))
    # The complete glyph is supported by two restored views of this frame;
    # a normal-pass fragment must not prevent collecting that evidence.
    assert any(box.points[:, 0].max() >= 120 for box in result.sampled_frame_boxes[2])


def _synthetic_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('similar', 'scripts/verify_similar_subtitles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_glyph_geometry():
    module = _synthetic_module()
    background = np.full((720, 1280, 3), 120, np.uint8)
    frame = background.copy()
    glyph = np.zeros((720, 1280), bool)
    module._draw_subtitle(frame, glyph)
    changed = np.any(frame != background, axis=2)
    assert changed.any()
    assert not (changed & ~glyph).any()


def test_synthetic_truth_is_pristine(tmp_path, monkeypatch):
    module = _synthetic_module()
    background = np.full((720, 1280, 3), 120, np.uint8)
    monkeypatch.setattr(module, 'EMPTY_HEAD', 0)
    monkeypatch.setattr(module, 'PRESENT', 1)
    monkeypatch.setattr(module, '_background_frames', lambda: [background.copy()])
    monkeypatch.setattr(module, '_sha256', lambda path: 'test')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: None)
    module.generate(tmp_path)
    truth = cv2.imread(str(tmp_path / 'truth-frames/000000.png'))
    source = cv2.imread(str(tmp_path / 'source-frames/000000.png'))
    assert np.array_equal(truth, background)
    assert not np.array_equal(truth, source)


def test_gap_recovery_failure_stays_negative_on_repeat(tmp_path):
    video = tmp_path / 'failed-gap.avi'
    _flash_video(video, frames=9)
    class Detector:
        calls = 0
        def detect(self, frame):
            self.calls += 1
            if frame.shape[:2] == (200, 400):
                return []
            raise RuntimeError('crop detector failed')
    detector = Detector()
    box = _VariantSignaledDetector(set(), (60, 100, 120, 130))._text_box()
    result = _flash_result(detector, {index: ([] if index == 2 else [box])
                                     for index in range(9)})
    plan = _gap_plan([Segment(0, 2), Segment(3, 9)])
    _add_spatial_segments(plan, result, video_path=str(video))
    assert result.failed_frame_indices == {2}
    assert len(plan.warnings) == 1
    before = detector.calls
    _add_spatial_segments(plan, result, video_path=str(video))
    assert detector.calls == before
    assert not execution_masks(plan)[1](2).any()


def test_recovery_does_not_swallow_cancellation(tmp_path):
    from videowipe.errors import ProcessingCancelledError
    import pytest
    video = tmp_path / 'cancel.avi'
    _flash_video(video, frames=9)
    class Detector:
        def detect(self, frame):
            if frame.shape[:2] == (200, 400):
                return []
            raise ProcessingCancelledError('cancel')
    result = _flash_result(Detector())
    with pytest.raises(ProcessingCancelledError):
        refine_temporal_presence(str(video), result, {'c1': [Segment(0, 6)]}, 6)
    result = _flash_result(Detector(), {index: [] for index in range(9)})
    with pytest.raises(ProcessingCancelledError):
        _add_spatial_segments(_gap_plan([Segment(0, 2), Segment(3, 9)]), result,
                              video_path=str(video))


def test_faint_glyph_recovery_amplifies_current_frame_contrast():
    from videowipe.detect import _contrast_recovered_boxes
    frame = np.full((120, 600, 3), 240, np.uint8)
    cv2.putText(frame, 'FAINT TEXT', (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (250, 250, 250), 3)
    box = TextBox(np.array([[50, 40], [320, 40], [320, 85], [50, 85]]), .9)

    class Detector:
        def detect(self, image):
            return [box] if np.ptp(image) >= 60 else []

    assert _contrast_recovered_boxes(Detector(), frame, (0, 0, 599, 119))
    assert not _contrast_recovered_boxes(
        Detector(), np.full_like(frame, 240), (0, 0, 599, 119),
    )


def test_same_restoration_family_cannot_confirm_itself(monkeypatch):
    import videowipe.detect as detect
    frame = np.zeros((120, 600, 3), np.uint8)
    box = TextBox(np.array([[50, 40], [320, 40], [320, 85], [50, 85]]), .9)
    monkeypatch.setattr(detect, '_contrast_restored_variants',
                        lambda crop: [('highpass', crop), ('highpass', crop)])

    class Detector:
        def detect(self, image):
            return [box]

    assert not detect._contrast_recovered_boxes(Detector(), frame, (0, 0, 599, 119))
