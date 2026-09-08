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
@pytest.mark.parametrize("interval,audio", [((5, 13), True), ((17, 20), False), ((0, 1), False)])
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
    expected_contexts = full_contexts[first // 4:(last + 3) // 4]
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
