from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from videowipe import InpaintOutcome, WipeEngine, register_inpainter


EXAMPLES = Path(__file__).parents[1] / "examples"


def _load_example(name):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["batch_worker", "custom_inpainter"])
def test_example_help_is_runnable(name):
    module = _load_example(name)
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])
    assert stopped.value.code == 0


def test_batch_example_reuses_one_engine(monkeypatch, tmp_path):
    module = _load_example("batch_worker")
    created = []

    class FakeEngine:
        def __init__(self, **options):
            self.requests = []
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, request, on_progress=None):
            self.requests.append(request)
            return request

    monkeypatch.setattr(module, "WipeEngine", FakeEngine)
    videos = [tmp_path / "a" / "clip.mp4", tmp_path / "b" / "clip.mp4"]

    results = module.process_batch(videos, tmp_path / "mask.png", tmp_path / "out")

    assert len(created) == 1
    assert len(created[0].requests) == 2
    assert created[0].requests[0].output_dir.name == "0001-clip"
    assert created[0].requests[1].output_dir.name == "0002-clip"
    assert created[0].requests[0].output_dir != created[0].requests[1].output_dir
    assert results == created[0].requests


def test_custom_example_runs_through_registry(monkeypatch, tmp_path):
    import videowipe.engine as engine_module

    module = _load_example("custom_inpainter")
    video = tmp_path / "input.mp4"
    mask = tmp_path / "mask.png"
    video.write_bytes(b"example-video")
    mask.write_bytes(b"example-mask")

    class Reader:
        def release(self):
            pass

    monkeypatch.setattr(
        engine_module,
        "read_frame_info",
        lambda path: (Reader(), {"W_ori": 2, "H_ori": 2, "fps": 24.0, "len": 1}),
    )
    monkeypatch.setattr(
        engine_module, "read_mask", lambda path: np.ones((2, 2, 1), dtype=np.uint8)
    )

    with module.build_engine() as engine:
        result = engine.run(module.WipeRequest(
            video=video,
            mask=mask,
            output_dir=tmp_path / "out",
        ))

    assert Path(result.output_path).read_bytes() == b"example-video"
    assert result.backend == module.MODEL_NAME


def test_reused_engine_loads_once_and_isolates_mask_and_metrics(
    monkeypatch, tmp_path
):
    import videowipe.engine as engine_module

    model_name = "recording-embedding-test"
    created = []

    class RecordingInpainter:
        name = model_name

        def __init__(self):
            self.load_calls = 0
            self.weight_paths = []
            self.masks = []
            self.metrics = []
            created.append(self)

        def load(self, weight_path, device="auto"):
            self.load_calls += 1
            self.weight_paths.append(weight_path)

        def inpaint(self, job):
            self.masks.append(job.mask.copy())
            self.metrics.append(job.metrics)
            job.metrics["adapter_run"] = len(self.masks)
            output = Path(job.output_dir) / f"run-{len(self.masks)}.mp4"
            output.write_bytes(b"result")
            return InpaintOutcome(str(output), backend=self.name)

        def cleanup(self):
            pass

    register_inpainter(model_name, RecordingInpainter)

    class Reader:
        def release(self):
            pass

    monkeypatch.setattr(
        engine_module,
        "read_frame_info",
        lambda path: (Reader(), {"W_ori": 2, "H_ori": 2, "fps": 24.0, "len": 1}),
    )
    masks = iter([
        np.ones((2, 2, 1), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
    ])
    monkeypatch.setattr(engine_module, "read_mask", lambda path: next(masks))

    monkeypatch.setattr(
        engine_module,
        "ensure_weight",
        lambda *args: pytest.fail("custom adapters must not resolve STTN weights"),
    )
    monkeypatch.setattr(
        engine_module,
        "ensure_onnx_weights",
        lambda *args: pytest.fail("custom adapters must not resolve STTN weights"),
    )
    engine = WipeEngine(model=model_name)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    engine.process("one.mp4", mask="one.png", output=str(first_dir))
    engine.process("two.mp4", mask="two.png", output=str(second_dir))

    assert len(created) == 1
    assert created[0].load_calls == 1
    assert created[0].weight_paths == [None]
    assert created[0].metrics
    assert np.all(created[0].masks[0] == 1)
    assert np.all(created[0].masks[1] == 0)
    assert created[0].metrics[0] is not created[0].metrics[1]
    first_benchmark = json.loads((first_dir / "benchmark.json").read_text())
    second_benchmark = json.loads((second_dir / "benchmark.json").read_text())
    assert first_benchmark["timing"]["adapter_run"] == 1
    assert second_benchmark["timing"]["adapter_run"] == 2
    engine.cleanup()
