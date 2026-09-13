"""Trial frames must use full-run context, masks, and audio timestamps."""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from videowipe import WipeEngine, WipeRequest
from videowipe.errors import InvalidInputError
from videowipe.inpainters.base import InpaintJob
from videowipe.inpainters import sttn


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
@pytest.mark.parametrize("interval,audio", [((5, 13), True), ((17, 20), False), ((0, 1), False), ((4, 7), False), ((10, 11), False)])
def test_trial_matches_full_run_before_encoding(tmp_path, monkeypatch, interval, audio):
    video = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=96x64:rate=4:duration=5",
        *(["-f", "lavfi", "-i",
           "aevalsrc=if(lt(t\\,1)\\,sin(2*PI*440*t)\\,sin(2*PI*880*t)):s=16000:d=5"] if audio else []),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    contexts = []

    def contextual_prediction(frames, *args):
        contexts.append(np.asarray(frames).copy())
        # Every prediction depends on ALL frames in its segment.
        value = np.mean(frames, axis=0).astype(np.uint8)
        return [value for _ in frames]

    monkeypatch.setattr(sttn, "_process_segment", contextual_prediction)
    original_blend = sttn._blend_frame_regions
    pixels = []

    def capture(*args):
        output = original_blend(*args)
        pixels.append(output.copy())
        return output

    monkeypatch.setattr(sttn, "_blend_frame_regions", capture)
    mask = np.zeros((64, 96, 1), dtype=np.uint8)
    mask[50:60, 10:86] = 1
    indices = []

    def temporal_mask(index):
        indices.append(index)
        return mask if 7 <= index < 11 else np.zeros_like(mask)

    painter = sttn.STTNInpainter()
    painter.backend = object()

    def run(interval=None):
        reader = cv2.VideoCapture(str(video))
        try:
            return painter.inpaint(InpaintJob(
                video_path=str(video), mask=mask, output_dir=str(tmp_path),
                fps=4, frame_count=20, width=96, height=64, reader=reader,
                gap=4, frame_mask=temporal_mask, trial_range=interval,
                dual=interval is not None,
            ))
        finally:
            reader.release()

    run()
    full_pixels = np.asarray(pixels)
    full_contexts = contexts[:]
    pixels.clear()
    contexts.clear()
    indices.clear()
    result = run(interval)
    first, last = interval
    assert np.array_equal(np.asarray(pixels), full_pixels[first:last])
    assert indices == list(range(first, last))
    expected_contexts = [
        context for start, context in zip((4, 8), full_contexts)
        if max(first, start, 7) < min(last, start + 4, 11)
    ]
    assert len(contexts) == len(expected_contexts)
    assert all(np.array_equal(a, b) for a, b in zip(contexts, expected_contexts))
    cap = cv2.VideoCapture(result.output_path)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == last - first
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 128
    cap.release()
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", result.output_path,
    ]))
    assert abs(float(info["format"]["duration"]) - (last - first) / 4) < 0.15
    assert any(s["codec_type"] == "audio" for s in info["streams"]) == audio
    if audio:
        pcm = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", result.output_path,
            "-t", "0.3", "-f", "f32le", "-ar", "16000", "-ac", "1", "-",
        ])
        samples = np.frombuffer(pcm, dtype="<f4")
        freq = np.fft.rfftfreq(len(samples), 1 / 16000)[np.argmax(abs(np.fft.rfft(samples)))]
        assert abs(freq - 880) < 10  # audio from trial timestamp, not source beginning


@pytest.mark.parametrize("interval", [(0, 0), (-1, 2), (False, 2), (1.5, 2), (1,), "1,2"])
def test_sdk_rejects_invalid_trial_before_loading(interval, tmp_path):
    with WipeEngine(task="clean") as engine:
        with pytest.raises(InvalidInputError, match="trial_range"):
            engine.run(WipeRequest(video="missing.mp4", output_dir=tmp_path, trial_range=interval))
        assert not engine._model_loaded


def test_trial_rejects_file_backend(tmp_path):
    with WipeEngine(task="clean", external_command="unused") as engine:
        with pytest.raises(InvalidInputError, match="STTN"):
            engine.run(WipeRequest(video="missing.mp4", output_dir=tmp_path, trial_range=(0, 1)))


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
def test_prediction_reuse_matches_uncached_protected_pixels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from videowipe.inpainters import prediction_cache
    from videowipe.plan import (Source, WipePlan, MaskAsset, TemporalResolution,
                                Track, Segment, execution_masks)
    video = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=96x64:rate=4:duration=3', '-c:v', 'libx264', str(video)], check=True)
    calls, captured = [], []
    def predict(frames, *args):
        calls.append(len(frames))
        return [np.mean(frames, axis=0).astype(np.uint8) for _ in frames]
    monkeypatch.setattr(sttn, '_process_segment', predict)
    blend = sttn._blend_frame_regions
    def record(*args):
        output = blend(*args)
        captured.append(output.copy())
        return output
    monkeypatch.setattr(sttn, '_blend_frame_regions', record)
    mask = np.zeros((64, 96), dtype=np.uint8); mask[52:60, 15:85] = 1
    protect = np.zeros_like(mask); protect[50:62, 30:50] = 1
    plan = WipePlan('wipe_plan', 1, Source('source.mp4', '', 96, 64, 4, 12), {},
                    TemporalResolution(0, 0, 0), MaskAsset('wipe_plan_masks.npz', ''),
                    [Track('r', 'region', 'remove', 'remove', (15, 52, 84, 59), 1, 1,
                           'user', [Segment(0, 12)], 'r', mask)])
    painter = sttn.STTNInpainter()
    painter.backend = SimpleNamespace(benchmark_metadata=lambda: {'device': 'cpu', 'weight_sha256': 'test'})
    cache = tmp_path / 'task' / 'predictions'
    def run(use_cache, interval=None):
        captured.clear(); calls.clear()
        union, temporal = execution_masks(plan, 4)
        reader = cv2.VideoCapture(str(video))
        try:
            painter.inpaint(InpaintJob(str(video), union, str(tmp_path), 4, 12, 96, 64,
                           reader=reader, gap=4, frame_mask=temporal, trial_range=interval,
                           prediction_cache_dir=str(cache) if use_cache else None))
        finally:
            reader.release()
        return np.asarray(captured).copy(), len(calls)
    _, count = run(True)
    assert count == 3
    plan.schema_version = 2
    plan.tracks[0].segments = [Segment(0, 10)]
    plan.tracks.append(Track('p', 'protection', 'protect', 'protect', (30, 50, 49, 61), 1, 1,
                            'user', [Segment(3, 7)], 'p', protect))
    cached, count = run(True)
    assert count == 0
    fresh, count = run(False)
    assert count == 3 and np.array_equal(cached, fresh)
    trial_pixels, count = run(True, (3, 7))
    assert count == 0 and np.array_equal(trial_pixels, fresh[3:7])
    reader = cv2.VideoCapture(str(video))
    for index in range(12):
        ok, frame = reader.read(); assert ok
        if 3 <= index < 7:
            assert np.array_equal(cached[index][protect.astype(bool)], frame[protect.astype(bool)])
    reader.release()
    next(cache.glob('*.npz')).write_bytes(b'corrupt')
    with pytest.warns(RuntimeWarning, match='Corrupt'):
        repaired, count = run(True)
    assert count == 1 and np.array_equal(repaired, fresh)
    painter.backend.benchmark_metadata = lambda: {'device': 'cpu', 'weight_sha256': 'changed'}
    changed, count = run(True)
    assert count == 3 and np.array_equal(changed, fresh)
    moved = np.zeros_like(mask); moved[32:40, 15:85] = 1
    plan.tracks[0].mask = moved
    moved_cached, count = run(True)
    assert count > 0
    moved_fresh, _ = run(False)
    assert np.array_equal(moved_cached, moved_fresh)
    plan.tracks[0].mask = mask
    for path in cache.glob('*.npz'): path.unlink()
    monkeypatch.setattr(prediction_cache, 'TASK_LIMIT', 1)
    bounded, count = run(True)
    assert count == 3 and np.array_equal(bounded, fresh) and not list(cache.glob('*.npz'))


def test_prediction_cache_rejected_by_external_backend(tmp_path):
    with WipeEngine(task='clean', external_command='unused') as engine:
        with pytest.raises(InvalidInputError, match='STTN'):
            engine.run(WipeRequest(video='missing.mp4', output_dir=tmp_path,
                                   prediction_cache_dir=tmp_path / 'predictions'))


def test_prediction_identity_and_shared_capacity(tmp_path, monkeypatch):
    from videowipe.inpainters import prediction_cache as module
    identity = {'source_sha256': 'source', 'weight_sha256': 'weights', 'device': 'cpu',
                'precision': 'float32', 'source_size': [96, 64], 'input_size': [640, 120],
                'backend': 'TorchBackend', 'ref_length': 5, 'neighbor_stride': 5,
                'implementation_sha256': 'code'}
    first = module.PredictionCache(tmp_path / 'one' / 'predictions', identity)
    key = first.key(0, 2, 46, 64)
    for name in identity:
        other = module.PredictionCache(first.directory, {**identity, name: 'changed'})
        assert other.key(0, 2, 46, 64) != key
    assert len({key, first.key(1, 2, 46, 64), first.key(0, 3, 46, 64), first.key(0, 2, 45, 64), first.key(0, 2, 46, 63)}) == 5
    pixels = np.ones((120, 640, 3), np.uint8)
    first.write(key, [pixels, None])
    read = first.read(key, 2)
    assert np.array_equal(read[0], pixels) and read[1] is None
    path = first.directory / f'{key}.npz'
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays['pixels'][0, 0, 0, 0] += 1
    np.savez(path, **arrays)
    with pytest.warns(RuntimeWarning, match='Corrupt'):
        assert first.read(key, 2) is None
    first.write(key, [pixels, None])
    monkeypatch.setattr(module, 'TOTAL_LIMIT', sum(p.stat().st_size for p in first.directory.glob('*.npz')) + 1)
    second = module.PredictionCache(tmp_path / 'two' / 'predictions', identity)
    second.write(key, [pixels, None])
    assert not list(second.directory.glob('*.npz'))
    assert first.read(key, 2) is not None
    monkeypatch.setattr(module, 'TOTAL_LIMIT', 2*1024**3)
    monkeypatch.setattr(module, 'TASK_LIMIT', pixels.nbytes+8192)
    (second.directory / ('.'+'a'*32+'.tmp')).write_bytes(b'x'*pixels.nbytes)
    second.write(key, [pixels])
    assert not list(second.directory.glob('*.npz'))


def test_prediction_cache_rejects_forged_array_header_before_allocation(tmp_path, monkeypatch):
    import io
    import zipfile
    from videowipe.inpainters.prediction_cache import PredictionCache
    cache = PredictionCache(tmp_path, {})
    key = cache.key(0, 1, 0, 120)
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(forged, {'descr': '|u1', 'fortran_order': False,
                                               'shape': (2**50, 120, 640, 3)})
    with zipfile.ZipFile(tmp_path / f'{key}.npz', 'w') as archive:
        archive.writestr('pixels.npy', forged.getvalue())
        archive.writestr('valid.npy', b'')
        archive.writestr('checksum.npy', b'')
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('unvalidated allocation'))
    with pytest.warns(RuntimeWarning, match='Corrupt'):
        assert cache.read(key, 1) is None
    assert cache.corrupt == cache.misses == 1
