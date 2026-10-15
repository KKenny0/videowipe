"""Source ordinals stay stable when the decoder's random seeking is inaccurate."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videowipe import WipeEngine, WipeRequest
from videowipe.detect import TextBox, _iter_sample_frames_with_indices
from videowipe.plan import execution_masks


@pytest.fixture
def inaccurate_seek(monkeypatch):
    capture = cv2.VideoCapture

    class Reader:
        def __init__(self, path):
            self.reader = capture(str(path))

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES and value > 2:
                value -= 2  # A successful seek can still return the wrong frame.
            return self.reader.set(prop, value)

    monkeypatch.setattr(cv2, 'VideoCapture', Reader)
    return Reader


def indexed_video(path, count=16):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 8, (160, 100))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((100, 160, 3), index, np.uint8))
    writer.release()
    return str(path)


def test_sampled_images_have_their_claimed_ordinals(tmp_path, inaccurate_seek):
    video = indexed_video(tmp_path / 'indexed.avi')
    sampled = list(_iter_sample_frames_with_indices(video, 6))
    assert [index for index, _ in sampled] == [0, 3, 6, 9, 12, 15]
    assert [int(frame[0, 0, 0]) for _, frame in sampled] == [0, 3, 6, 9, 12, 15]


def test_sdk_plan_uses_current_frame_evidence_and_keeps_blank_frames(tmp_path, inaccurate_seek):
    video = indexed_video(tmp_path / 'indexed.avi')
    present = {0, 1, 2, 5, 6, 7, 10, 11, 12, 15}
    box = TextBox(np.array([[35, 80], [125, 80], [125, 88], [35, 88]]), .95)
    detector = SimpleNamespace(detect=lambda frame: [box] if int(frame[0, 0, 0]) in present else [])
    with WipeEngine(task='clean', detector=detector, ocr='off') as engine:
        plan = engine.plan(WipeRequest(video, output_dir=tmp_path / 'plan'))
    assert len(plan.remove_tracks) == 1
    _, alpha = execution_masks(plan)
    assert {index for index in range(16) if alpha(index).any()} == present


def frame_plan(segments):
    from videowipe.plan import MaskAsset, Source, TemporalResolution, Track, WipePlan
    return WipePlan('wipe_plan', 3, Source('indexed.avi', 'a'*64, 160, 100, 8, 16), {},
                    TemporalResolution(4, 4, 4), MaskAsset('wipe_plan_masks.npz', ''), [
                        Track('c1', 'subtitle', 'subtitle', 'remove', (30, 75, 130, 99),
                              1, 1, 'default', segments, 'c1')])


def test_boundary_probes_extend_only_across_current_positive_frames(tmp_path, inaccurate_seek):
    from videowipe.plan import Segment
    from videowipe.planning import _add_spatial_segments
    video = indexed_video(tmp_path / 'indexed.avi')
    box = TextBox(np.array([[35, 80], [125, 80], [125, 88], [35, 88]]), .95)
    plan = frame_plan([Segment(5, 8)])
    result = SimpleNamespace(
        frame_shape=(100, 160), candidates=[SimpleNamespace(id='c1', type='subtitle', detector_backed=True)],
        sampled_frame_boxes={5: [box], 6: [box], 7: [box], 8: []},
        detector=SimpleNamespace(detect=lambda f: [box] if 4 <= int(f[0, 0, 0]) < 8 else []))
    _add_spatial_segments(plan, result, video_path=video)
    _, alpha = execution_masks(plan)
    assert {i for i in range(16) if alpha(i).any()} == {4, 5, 6, 7}


def test_outline_recovery_starts_at_the_claimed_frame(tmp_path, inaccurate_seek):
    from videowipe.plan import Segment, _spatial_mask
    from videowipe.planning import _recover_descenders
    video = tmp_path / 'outline.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 8, (160, 100))
    assert writer.isOpened()
    for i in range(16):
        frame = np.full((100, 160, 3), 50, np.uint8)
        if i == 5:
            frame[85:95, 60:65] = 0
            frame[85:90, 61:64] = 255
        writer.write(frame)
    writer.release()
    plan = frame_plan([Segment(5, 6)])
    track = plan.tracks[0]
    track.spatial_segments = [dict(start=5, end=6, boxes=[[50, 80, 75, 92]])]
    track.mask = _spatial_mask(track, (100, 160))
    _recover_descenders(plan, [track], str(video), None)
    _, alpha = execution_masks(plan)
    assert alpha(5)[94, 62] == 1
    assert not alpha(4).any() and not alpha(6).any()


def test_trial_preserves_full_run_frame_identity_with_inaccurate_seek(tmp_path, monkeypatch, inaccurate_seek):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('requires ffmpeg')
    from videowipe.inpainters import sttn
    from videowipe.inpainters.base import InpaintJob
    monkeypatch.setattr(sttn, '_VIDEO_CAPTURE_TYPE', inaccurate_seek)
    monkeypatch.setattr(sttn, '_process_segment', lambda frames, *args: [np.zeros_like(f) for f in frames])
    video = indexed_video(tmp_path / 'indexed.avi')
    mask = np.zeros((100, 160, 1), np.uint8)
    mask[80:90, 35:125] = 1
    pixels = []
    blend = sttn._blend_frame_regions
    def capture(*args):
        out = blend(*args)
        pixels.append(out.copy())
        return out
    monkeypatch.setattr(sttn, '_blend_frame_regions', capture)
    painter = sttn.STTNInpainter()
    painter.backend = object()
    def run(interval):
        reader = cv2.VideoCapture(video)
        try:
            painter.inpaint(InpaintJob(video, mask, str(tmp_path), 8, 16, 160, 100,
                                      reader=reader, gap=4, trial_range=interval))
        finally:
            reader.release()
    run(None)
    full = pixels[:]
    pixels.clear()
    run((5, 9))
    assert len(pixels) == 4
    assert all(np.array_equal(a, b) for a, b in zip(pixels, full[5:9]))
    assert [int(f[0, 0, 0]) for f in pixels] == [5, 6, 7, 8]


def test_sdk_cancellation_stops_between_sparse_sample_frames(tmp_path, monkeypatch):
    from videowipe import CancellationToken, ProcessingCancelledError
    video = indexed_video(tmp_path / 'long-prefix.avi', count=100)
    capture = cv2.VideoCapture
    token = CancellationToken()
    reads = []
    readers = []
    class Reader:
        def __init__(self, path):
            self.reader = capture(str(path))
            self.released = False
            readers.append(self)
        def __getattr__(self, name):
            return getattr(self.reader, name)
        def read(self):
            reads.append(1)
            if len(reads) == 3:
                token.cancel()
            return self.reader.read()
        def release(self):
            self.released = True
            self.reader.release()
    monkeypatch.setattr(cv2, 'VideoCapture', Reader)
    detector = SimpleNamespace(detect=lambda frame: [])
    with WipeEngine(task='clean', detector=detector, ocr='off', detect_mode='fast') as engine:
        with pytest.raises(ProcessingCancelledError):
            engine.plan(WipeRequest(video, tmp_path / 'plan'), cancellation=token)
    assert len(reads) == 3
    assert readers and all(reader.released for reader in readers)


def test_skip_stops_on_decode_failure_or_cancellation():
    from videowipe.errors import ProcessingCancelledError
    from videowipe.video import skip_frames
    calls = []
    reader = SimpleNamespace(read=lambda: calls.append(1) or (len(calls) <= 2, None))
    with pytest.raises(ValueError, match='decoded 2'):
        skip_frames(reader, 4)
    assert len(calls) == 3
    calls.clear()
    def cancelled():
        raise ProcessingCancelledError('cancelled')
    with pytest.raises(ProcessingCancelledError):
        skip_frames(reader, 10, cancelled)
    assert calls == []
