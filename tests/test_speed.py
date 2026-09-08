"""Device selection and zero-alpha skipping must preserve execution semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from videowipe.backends import TorchBackend
from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob


@pytest.mark.parametrize("cuda,mps,requested,expected", [
    (True, True, "auto", "cuda:0"), (False, True, "auto", "mps"),
    (False, False, "auto", "cpu"), (True, True, "cpu", "cpu"),
])
def test_device_precedence(monkeypatch, cuda, mps, requested, expected):
    torch = pytest.importorskip("torch")
    from videowipe.models import sttn as model
    devices = []

    class FakeModel:
        def to(self, device):
            devices.append(str(device))
            return self

        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"netG": {}})
    monkeypatch.setattr(model, "InpaintGenerator", FakeModel)
    assert str(TorchBackend("unused", requested).device) == expected
    assert devices == [expected]


def test_explicit_unavailable_mps_errors_before_loading(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="use device=cpu"):
        TorchBackend("nonexistent", "mps")


@pytest.mark.parametrize("alpha", [0.0, 0.00001, 1.0])
@pytest.mark.parametrize("storage", ["fresh", "reused", "readonly", "readonly_view"])
def test_skip_per_band_preserves_context_and_original_pixels(tmp_path, monkeypatch, alpha, storage):
    frames = [np.full((64, 96, 3), i * 20, np.uint8) for i in range(5)]
    source = iter(frames)
    reader = SimpleNamespace(read=lambda: (True, next(source).copy()))
    mask = np.ones((64, 96, 1), np.float32)
    calls, indices, pixels = [], [], []
    # Two disjoint bands; only the second may have a soft-alpha target.
    monkeypatch.setattr(sttn, "get_inpaint_mode", lambda *a: [(0, 18), (40, 58)])

    def predict(context, *args):
        calls.append([float(f.mean()) for f in context])
        return [np.full((120, 640, 3), 200, np.uint8) for _ in context]

    buffer = np.zeros_like(mask)
    immutable = np.zeros_like(mask)
    immutable[40:58] = alpha
    immutable.setflags(write=False)
    empty = np.zeros_like(mask)
    empty.setflags(write=False)

    def frame_mask(index):
        indices.append(index)
        if storage == "readonly":
            return immutable if index == 1 else empty
        result = buffer if storage in ("reused", "readonly_view") else np.zeros_like(mask)
        result.fill(0)
        if index == 1:
            result[40:58] = alpha
        if storage == "readonly_view":
            result = result.view()
            result.setflags(write=False)
        return result

    class Sink:
        def write(self, value):
            pixels.append(np.frombuffer(value, np.uint8).reshape(64, 96, 3).copy())

        def close(self):
            pass

    pipe = SimpleNamespace(stdin=Sink(), wait=lambda: 0, poll=lambda: 0, returncode=0)
    monkeypatch.setattr(sttn.subprocess, "Popen", lambda *a, **kw: pipe)
    monkeypatch.setattr(sttn, "_process_segment", predict)
    painter = sttn.STTNInpainter()
    painter.backend = object()
    painter.inpaint(InpaintJob(
        video_path="", reader=reader, mask=mask, output_dir=str(tmp_path),
        width=96, height=64, fps=4, frame_count=5, gap=3,
        frame_mask=frame_mask,
    ))
    assert indices == list(range(5))
    assert calls == ([[0.0, 20.0, 40.0]] if alpha else [])
    for index, output in enumerate(pixels):
        expected = frames[index].copy()
        if index == 1:
            expected[40:58] = np.clip(alpha * 200 + (1 - alpha) * 20, 0, 255).astype(np.uint8)
        assert np.array_equal(output, expected)
