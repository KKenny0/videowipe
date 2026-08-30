import inspect
import json
import os
import pathlib
import re
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import videowipe
from videowipe import agent as agent_module
from videowipe import cli
from videowipe.backends import ONNXBackend, _detect_backend
from videowipe.cli import _build_parser
from videowipe.detect import (
    CleanCandidate,
    DBNetDetector,
    TextBox,
    _classify_region,
    _iou_bbox,
    detect_clean_candidates,
    mask_from_candidates,
    resolve_detect_params,
    select_candidates_by_intent,
    select_clean_candidates,
)
from videowipe.engine import WipeEngine, remove_text
from videowipe.external import ExternalInpainter, ExternalModelError, run_external
from videowipe.inpainters import STTNInpainter, get_registry
from videowipe.inpainters.base import InpaintJob
from videowipe.plan import (
    JSON_FILENAME,
    build_wipe_plan,
    compute_source,
    save_wipe_plan,
)
from videowipe.tasks.base import BaseTask, read_mask, validate_mask_shape


def _write_test_video(path, width=96, height=64, frames=4, draw=None):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4,
        (width, height),
    )
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if draw is not None:
            draw(frame, i)
        writer.write(frame)
    writer.release()


def test_dbnet_latches_manual_only_after_proven_high_level_miss(monkeypatch):
    detector = object.__new__(DBNetDetector)
    detector._adaptive = False
    detector._input_w = detector._input_h = 32
    high_level_model = object()
    detector._hl_model = high_level_model
    detector._manual_only = False
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    box = TextBox(
        points=np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32),
        confidence=0.9,
    )
    calls = {"hl": 0, "manual": 0}
    hl_results = iter([[box], [], [], []])
    manual_results = iter([[], [], [box], [box]])

    def high_level(_frame):
        calls["hl"] += 1
        return next(hl_results)

    def manual(_frame, **_kwargs):
        calls["manual"] += 1
        return next(manual_results)

    monkeypatch.setattr(detector, "_detect_hl", high_level)
    monkeypatch.setattr(detector, "_detect_manual", manual)

    assert detector.detect(frame) == [box]
    assert calls == {"hl": 1, "manual": 0}
    assert detector.detect(frame) == []
    assert detector.detect(frame) == []
    assert detector._manual_only is False
    assert detector.detect(frame) == [box]
    assert detector._manual_only is True
    assert detector._hl_model is high_level_model
    assert detector.detect(frame) == [box]
    assert calls == {"hl": 4, "manual": 4}


def test_cli_exposes_detext_and_clean_commands():
    parser = _build_parser()

    assert parser.parse_args(["detext", "-v", "input.mp4"]).command == "detext"
    clean = parser.parse_args(["clean", "input.mp4", "--target", "watermark"])
    assert clean.command == "clean"
    assert clean.video == "input.mp4"
    assert clean.target == ["watermark"]
    intent = parser.parse_args(
        ["clean", "input.mp4", "--intent", "去掉底部字幕，保留路牌", "--agent", "codex"]
    )
    assert intent.intent == "去掉底部字幕，保留路牌"
    assert intent.agent == "codex"
    region = parser.parse_args(["clean", "input.mp4", "--region", "top-right"])
    assert region.region == ["top-right"]
    with pytest.raises(SystemExit):
        parser.parse_args(["delogo", "-v", "input.mp4", "-m", "mask.png"])


def test_gap_defaults_are_25_and_explicit_cli_value_reaches_engine(monkeypatch):
    parser = _build_parser()
    assert parser.parse_args(["detext", "-v", "input.mp4"]).gap == 25
    assert parser.parse_args(["clean", "input.mp4"]).gap == 25
    assert parser.parse_args(["clean", "input.mp4", "--gap", "17"]).gap == 17

    default_engine = WipeEngine()
    explicit_engine = WipeEngine(gap=17)
    try:
        assert default_engine._task_impl.gap == 25
        assert explicit_engine._task_impl.gap == 17
    finally:
        default_engine.cleanup()
        explicit_engine.cleanup()

    assert inspect.signature(remove_text).parameters["gap"].default == 25
    assert inspect.signature(BaseTask).parameters["gap"].default == 25
    assert InpaintJob.__dataclass_fields__["gap"].default == 25

    captured = {}

    class RecordingEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def process(self, **kwargs):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(cli, "WipeEngine", RecordingEngine)
    monkeypatch.setattr(
        sys, "argv", ["videowipe", "clean", "input.mp4", "--gap", "17"]
    )
    cli.main()
    assert captured["gap"] == 17


# ── Phase B / B1: CLI --plan ─────────────────────────────────────────────────

def _plan_candidate(cid, frame_shape, default_remove=True):
    """Duck-typed candidate with a precise mask, for build_wipe_plan."""
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:60, 8:88] = 1
    return SimpleNamespace(
        id=cid,
        type="subtitle",
        label=f"{cid} subtitle",
        bbox=(8, 50, 88, 60),
        confidence=0.9,
        default_remove=default_remove,
        mask=mask,
        presence_frames=[],  # full-video segment; CLI test does not need temporal math
    )


def _save_two_track_plan(video_path, plan_dir):
    """Build + save a real WipePlan bound to *video_path* (c1 remove, c2 keep)."""
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap.release()
    source = compute_source(str(video_path))
    plan = build_wipe_plan(
        [
            _plan_candidate("c1", (h, w), default_remove=True),
            _plan_candidate("c2", (h, w), default_remove=False),
        ],
        sample_indices=[0, 2, 4, 6],
        n_valid=4,
        source=source,
        frame_shape=(h, w),
    )
    plan_dir.mkdir(parents=True, exist_ok=True)
    save_wipe_plan(plan, str(plan_dir))
    return plan_dir / JSON_FILENAME


class _FakeCleanTask:
    """Stand-in for the STTN task: records nothing, writes a sentinel output."""

    _bm = None
    backend = type("B", (), {"__name__": "FakeBackend"})()
    output_suffix = "clean"
    feather_radius = 0
    frame_mask = None

    def process_video(self, reader, frame_info, mask_arr, output_dir,
                      video_path="", progress=None):
        self._bm["timing"]["inpainting_s"] = 0.001
        out_path = pathlib.Path(output_dir) / "output_clean.mp4"
        out_path.write_bytes(b"clean")
        return str(out_path)

    def cleanup(self):
        pass


def test_cli_clean_exposes_plan_mutually_exclusive_with_mask():
    parser = _build_parser()
    args = parser.parse_args(["clean", "input.mp4", "--plan", "wipe_plan.json"])
    assert args.plan == "wipe_plan.json"
    # --mask and --plan cannot be combined; argparse rejects before the engine.
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["clean", "input.mp4", "--mask", "m.png", "--plan", "p.json"]
        )


def test_cli_clean_plan_round_trip_executes_and_rejects_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)
    monkeypatch.setattr(
        "videowipe.engine._TASK_CLASSES", {"clean": lambda **kw: _FakeCleanTask()}
    )

    def run_cli(*argv):
        monkeypatch.setattr(sys, "argv", ["videowipe", *argv])
        cli.main()

    video = tmp_path / "input.mp4"
    _write_test_video(video)
    plan_path = _save_two_track_plan(video, tmp_path / "plan")

    # Execute the generated plan unchanged.
    out = tmp_path / "run"
    run_cli("clean", str(video), "--plan", str(plan_path), "-o", str(out))
    assert (out / "output_clean.mp4").exists()

    # Execute a modified plan: swap the two tracks' actions in the JSON. The
    # NPZ (and its sha) is untouched, so load_wipe_plan still validates; the
    # newly-remove track's precise mask is already in the NPZ.
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    original = {t["id"]: t["action"] for t in data["tracks"]}
    for track in data["tracks"]:
        track["action"] = "keep" if track["id"] == "c1" else "remove"
    plan_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    out2 = tmp_path / "run2"
    run_cli("clean", str(video), "--plan", str(plan_path), "-o", str(out2))
    assert (out2 / "output_clean.mp4").exists()
    assert original["c1"] == "remove"  # confirm the edit actually flipped it

    # Source mismatch: the plan is bound to `video`; running it against a
    # different video is rejected before any model loads.
    other = tmp_path / "other.mp4"
    _write_test_video(other, width=128)
    with pytest.raises(SystemExit):
        run_cli("clean", str(other), "--plan", str(plan_path),
                "-o", str(tmp_path / "run3"))


def test_version_fields_are_in_sync():
    pyproject = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None, "pyproject version not found"
    assert match.group(1) == videowipe.__version__


def test_gpu_docker_stage_sets_noninteractive_install_and_pythonpath():
    dockerfile = pathlib.Path("Dockerfile").read_text(encoding="utf-8")
    gpu_stage = dockerfile.split(
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu",
        maxsplit=1,
    )[1]

    assert "ENV TZ=Etc/UTC" in gpu_stage
    assert "DEBIAN_FRONTEND=noninteractive" in gpu_stage
    assert "tzdata" in gpu_stage
    assert "dpkg-reconfigure --frontend noninteractive tzdata" in gpu_stage
    assert "PYTHONPATH=/usr/local/lib/python3.11/site-packages" in gpu_stage


def test_engine_rejects_unregistered_task():
    with pytest.raises(ValueError, match="Unknown task"):
        WipeEngine(task="delogo")


def test_read_mask_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot read mask image"):
        read_mask(str(tmp_path / "missing.png"))


def test_validate_mask_shape_rejects_video_size_mismatch():
    mask = np.zeros((10, 20, 1), dtype=np.uint8)
    frame_info = {"H_ori": 11, "W_ori": 20}

    with pytest.raises(ValueError, match="does not match video shape"):
        validate_mask_shape(mask, frame_info)


def test_backend_extension_detection_is_explicit():
    assert _detect_backend("model.onnx") == "onnx"
    assert _detect_backend("model.pth") == "torch"
    assert _detect_backend("model.pt") == "torch"
    with pytest.raises(ValueError, match="Unsupported weight file"):
        _detect_backend("model.bin")


@pytest.mark.parametrize(
    ("system", "machine", "available", "expected"),
    [
        ("Darwin", "arm64", {"onnxruntime", "torch", "torchvision"}, "model.pth"),
        ("Darwin", "arm64", {"onnxruntime"}, "model.onnx"),
        ("Linux", "aarch64", {"onnxruntime", "torch", "torchvision"}, "model.onnx"),
    ],
)
def test_default_backend_selection_matches_platform_benchmark(
    monkeypatch, system, machine, available, expected,
):
    import videowipe.engine as engine_module

    loaded = []

    class FakeInpainter:
        backend = object()

        def load(self, weight_path, device="auto"):
            loaded.append(weight_path)

    monkeypatch.setattr(engine_module.platform, "system", lambda: system)
    monkeypatch.setattr(engine_module.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        engine_module, "_module_available", lambda name: name in available,
    )
    monkeypatch.setattr(engine_module, "ensure_weight", lambda *args: "model.pth")
    monkeypatch.setattr(
        engine_module, "ensure_onnx_weights", lambda *args: "model",
    )
    monkeypatch.setattr(
        engine_module.get_registry(), "create", lambda name: FakeInpainter(),
    )

    WipeEngine()._ensure_model()

    assert loaded == [expected]


@pytest.mark.parametrize(
    ("weight", "available"),
    [
        ("explicit.onnx", {"onnxruntime"}),
        ("explicit.pth", {"torch", "torchvision"}),
    ],
)
def test_explicit_weight_bypasses_default_backend_selection(
    monkeypatch, weight, available,
):
    import videowipe.engine as engine_module

    loaded = []

    class FakeInpainter:
        backend = object()

        def load(self, weight_path, device="auto"):
            loaded.append(weight_path)

    monkeypatch.setattr(
        engine_module, "_module_available", lambda name: name in available,
    )
    monkeypatch.setattr(
        engine_module,
        "ensure_weight",
        lambda *args: pytest.fail("explicit weight must not resolve a default"),
    )
    monkeypatch.setattr(
        engine_module,
        "ensure_onnx_weights",
        lambda *args: pytest.fail("explicit weight must not resolve a default"),
    )
    monkeypatch.setattr(
        engine_module.get_registry(), "create", lambda name: FakeInpainter(),
    )

    WipeEngine(weight=weight)._ensure_model()

    assert loaded == [weight]


def test_onnx_backend_filters_unavailable_providers(tmp_path, monkeypatch):
    calls = []

    class FakeSession:
        def __init__(self, path, providers):
            calls.append((path, providers))

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "CoreMLExecutionProvider", "CPUExecutionProvider",
        ],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    base = tmp_path / "sttn"
    for part in ("encoder", "transformer", "decoder"):
        (tmp_path / f"sttn_{part}.onnx").touch()

    ONNXBackend(str(base) + ".onnx")

    assert [providers for _, providers in calls] == [
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_onnx_backend_requires_cpu_provider(tmp_path, monkeypatch):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CoreMLExecutionProvider"],
        InferenceSession=lambda *args, **kwargs: pytest.fail(
            "session must not load without the CPU provider"
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    base = tmp_path / "sttn"
    for part in ("encoder", "transformer", "decoder"):
        (tmp_path / f"sttn_{part}.onnx").touch()

    with pytest.raises(RuntimeError, match="CPUExecutionProvider is required"):
        ONNXBackend(str(base) + ".onnx")


def test_registry_exposes_sttn():
    """The built-in STTN inpainter is registered under the name 'sttn'."""
    registry = get_registry()
    assert "sttn" in registry.names()
    inpainter = registry.create("sttn")
    assert isinstance(inpainter, STTNInpainter)
    assert inpainter.name == "sttn"


def test_registry_rejects_unknown_model():
    """Creating an unregistered model name raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown inpainter"):
        get_registry().create("nonexistent-model")


def test_registry_exposes_external():
    """The external inpainter is registered under 'external'."""
    registry = get_registry()
    assert "external" in registry.names()
    inpainter = registry.create("external", command="echo noop")
    assert isinstance(inpainter, ExternalInpainter)
    assert inpainter.name == "external"
    assert inpainter.command == "echo noop"


def test_registry_exposes_propainter():
    """ProPainter is registered as an ExternalInpainter named 'propainter'."""
    registry = get_registry()
    assert "propainter" in registry.names()
    inpainter = registry.create("propainter", propainter_dir="/some/path")
    assert isinstance(inpainter, ExternalInpainter)
    assert inpainter.name == "propainter"  # overridden by the factory
    assert "-m videowipe.propainter_wipe" in inpainter.command
    assert "--propainter-dir /some/path" in inpainter.command


def test_propainter_factory_omits_dir_flag_when_unset():
    """Without propainter_dir, the command has no --propainter-dir flag."""
    inpainter = get_registry().create("propainter")
    assert isinstance(inpainter, ExternalInpainter)
    assert inpainter.name == "propainter"
    assert "--propainter-dir" not in inpainter.command
    assert "-m videowipe.propainter_wipe" in inpainter.command


def test_propainter_factory_preserves_directory_with_spaces(tmp_path):
    from videowipe.external import _split_command

    propainter_dir = tmp_path / "Pro Painter"
    inpainter = get_registry().create(
        "propainter", propainter_dir=str(propainter_dir)
    )
    argv = _split_command(inpainter.command)

    assert argv[-2:] == ["--propainter-dir", str(propainter_dir)]


def test_remove_text_cleans_up_when_processing_fails(monkeypatch):
    calls = []

    def fail_process(self, **kwargs):
        raise RuntimeError("boom")

    def track_cleanup(self):
        calls.append("cleanup")

    monkeypatch.setattr(WipeEngine, "process", fail_process)
    monkeypatch.setattr(WipeEngine, "cleanup", track_cleanup)

    with pytest.raises(RuntimeError, match="boom"):
        remove_text(video="input.mp4")

    assert calls == ["cleanup"]


def test_cli_translates_errors_to_stderr(monkeypatch, capsys):
    class FailingEngine:
        def __init__(self, **kwargs):
            pass

        def process(self, **kwargs):
            raise ValueError("bad input")

        def cleanup(self):
            pass

    monkeypatch.setattr(cli, "WipeEngine", FailingEngine)
    monkeypatch.setattr(
        "sys.argv", ["videowipe", "detext", "-v", "input.mp4"]
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "videowipe: bad input" in capsys.readouterr().err


def test_clean_detection_classifies_text_targets(tmp_path):
    video = tmp_path / "input.mp4"
    _write_test_video(video, width=320, height=180)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            boxes = [
                TextBox(
                    points=np.array([[30, 160], [290, 160], [290, 175], [30, 175]]),
                    confidence=0.9,
                    text="hello subtitle",
                ),
                TextBox(
                    points=np.array([[5, 5], [150, 5], [150, 20], [5, 20]]),
                    confidence=0.95,
                    text="2026-05-25 12:30:05",
                ),
                TextBox(
                    points=np.array([[250, 5], [315, 5], [315, 20], [250, 20]]),
                    confidence=0.8,
                    text="@brand",
                ),
            ]
            if self.calls < 3:
                boxes.append(TextBox(
                    points=np.array([[120, 75], [200, 75], [200, 95], [120, 95]]),
                    confidence=0.7,
                    text="Main St",
                ))
            return boxes

    result = detect_clean_candidates(str(video), detector=FakeDetector(), sample_count=3)
    types = {candidate.type for candidate in result.candidates}

    assert {"subtitle", "timestamp", "watermark", "scene_text"} <= types
    scene_text = [candidate for candidate in result.candidates if candidate.type == "scene_text"][0]
    assert scene_text.default_remove is False


def test_clean_timestamp_requires_recognized_text_content(tmp_path):
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[2, 2], [44, 2], [44, 10], [2, 10]]),
                    confidence=0.95,
                )
            ]

    result = detect_clean_candidates(str(video), detector=FakeDetector(), sample_count=3)

    assert all(candidate.type != "timestamp" for candidate in result.candidates)
    assert select_clean_candidates(result.candidates, targets=["timestamp"]) == []


@pytest.mark.parametrize(
    ("bbox", "zone", "presence", "appearance", "expected"),
    [
        ((5, 5, 150, 20), "top-left", 1.0, 1.0, ("watermark", False)),
        ((170, 5, 315, 20), "top-right", 1.0, 1.0, ("logo", False)),
        ((5, 5, 150, 20), "top-left", 1.0, None, ("watermark", False)),
        ((5, 5, 150, 20), "top-left", 0.5, 1.0, ("subtitle", True)),
    ],
)
def test_top_text_semantics_use_persistence_and_side(
    bbox, zone, presence, appearance, expected
):
    target_type, _reason, default_remove = _classify_region(
        bbox, zone, [], 320, 180, presence_fraction=presence,
        appearance_stability=appearance,
    )

    assert (target_type, default_remove) == expected


@pytest.mark.parametrize(
    ("text_template", "expected"),
    [
        ("brand", ("watermark", False)),
        ("headline {index}", ("subtitle", True)),
    ],
)
def test_top_overlay_requires_stable_appearance(
    tmp_path, text_template, expected
):
    video = tmp_path / "input.mp4"

    def draw(frame, index):
        cv2.putText(
            frame, text_template.format(index=index),
            (20, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2,
        )

    _write_test_video(video, width=320, height=180, frames=5, draw=draw)

    class TopSubtitleDetector:
        def detect(self, frame):
            return [TextBox(
                points=np.array([[15, 5], [260, 5], [260, 35], [15, 35]]),
                confidence=0.9,
            )]

    result = detect_clean_candidates(
        str(video), detector=TopSubtitleDetector(), sample_count=5,
    )

    candidate = result.candidates[0]
    assert (candidate.type, candidate.default_remove) == expected


def test_dbnet_rejects_weight_replaced_during_load(tmp_path, monkeypatch):
    weight = tmp_path / "detector.onnx"
    weight.write_bytes(b"first")

    def load(path):
        assert isinstance(path, np.ndarray)
        weight.write_bytes(b"second")
        return object()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)

    with pytest.raises(RuntimeError, match="changed while it was being loaded"):
        DBNetDetector(str(weight))


def test_clean_candidate_selection_and_mask_merge():
    subtitle_mask = np.zeros((20, 30, 1), dtype=np.uint8)
    subtitle_mask[15:18, 2:28] = 1
    timestamp_mask = np.zeros((20, 30, 1), dtype=np.uint8)
    timestamp_mask[1:4, 1:8] = 1
    scene_mask = np.zeros((20, 30, 1), dtype=np.uint8)
    scene_mask[8:12, 10:18] = 1

    candidates = [
        CleanCandidate(
            id="c1", type="subtitle", label="bottom subtitle",
            bbox=(2, 15, 27, 17), confidence=0.9, frame_fraction=1.0,
            reason="wide bottom text", default_remove=True, mask=subtitle_mask,
        ),
        CleanCandidate(
            id="c2", type="timestamp", label="top timestamp",
            bbox=(1, 1, 7, 3), confidence=0.9, frame_fraction=1.0,
            reason="time-like text", default_remove=True, mask=timestamp_mask,
        ),
        CleanCandidate(
            id="c3", type="scene_text", label="center scene text",
            bbox=(10, 8, 17, 11), confidence=0.9, frame_fraction=1.0,
            reason="scene text", default_remove=False, mask=scene_mask,
        ),
    ]

    assert [c.id for c in select_clean_candidates(candidates)] == ["c1", "c2"]
    assert [c.id for c in select_clean_candidates(candidates, targets=["timestamp"])] == ["c2"]

    mask = mask_from_candidates(select_clean_candidates(candidates), (20, 30))
    assert mask[16, 5, 0] == 1
    assert mask[2, 2, 0] == 1
    assert mask[9, 12, 0] == 0


def test_clean_intent_selects_remove_target_and_keeps_scene_text():
    candidates = [
        CleanCandidate(
            id="c1", type="subtitle", label="bottom subtitle",
            bbox=(2, 15, 27, 17), confidence=0.9, frame_fraction=1.0,
            reason="wide bottom text", default_remove=True,
            text_samples=["中文字幕"],
        ),
        CleanCandidate(
            id="c2", type="scene_text", label="center scene text",
            bbox=(10, 8, 17, 11), confidence=0.9, frame_fraction=1.0,
            reason="scene text", default_remove=False,
            text_samples=["路牌"],
        ),
    ]

    selected = select_candidates_by_intent(candidates, "去掉底部中文字幕，保留路牌")

    assert [candidate.id for candidate in selected] == ["c1"]


def test_clean_intent_keep_only_removes_from_default_selection():
    candidates = [
        CleanCandidate(
            id="c1", type="subtitle", label="bottom subtitle",
            bbox=(2, 15, 27, 17), confidence=0.9, frame_fraction=1.0,
            reason="wide bottom text", default_remove=True,
        ),
        CleanCandidate(
            id="c2", type="watermark", label="top-right text watermark",
            bbox=(20, 1, 29, 4), confidence=0.9, frame_fraction=1.0,
            reason="watermark-like text", default_remove=True,
        ),
    ]

    selected = select_clean_candidates(candidates, intent="保留右上角水印")

    assert [candidate.id for candidate in selected] == ["c1"]


def test_clean_preview_writes_artifacts_without_loading_model(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="hello subtitle",
                )
            ]

    def fail_model_load(self):
        raise AssertionError("preview should not load the inpainting model")

    monkeypatch.setattr(WipeEngine, "_ensure_model", fail_model_load)

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        assert engine.process(video=str(video), output=str(output), preview=True) == str(output)
    finally:
        engine.cleanup()

    assert (output / "clean_candidates.json").exists()
    assert (output / "clean_preview.jpg").exists()
    assert (output / "clean_preview_source.jpg").exists()
    assert (output / "auto_mask.png").exists()


def test_clean_agent_selection_overrides_local_rules(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="hello subtitle",
                ),
                TextBox(
                    points=np.array([[70, 4], [94, 4], [94, 12], [70, 12]]),
                    confidence=0.8,
                    text="@brand",
                ),
            ]

    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)
    monkeypatch.setattr(
        WipeEngine,
        "_select_candidates_with_agent",
        staticmethod(lambda agent, candidates, intent: [c for c in candidates if c.type == "watermark"]),
    )

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            intent="去掉底部字幕",
            agent="codex",
        )
    finally:
        engine.cleanup()

    data = (output / "clean_candidates.json").read_text(encoding="utf-8")
    assert '"type": "watermark"' in data
    assert '"selected": true' in data


def test_clean_agent_failure_falls_back_to_local_rules(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="hello subtitle",
                )
            ]

    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)
    monkeypatch.setattr(
        WipeEngine,
        "_select_candidates_with_agent",
        staticmethod(lambda agent, candidates, intent: None),
    )

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            intent="去掉字幕",
            agent="missing-agent",
        )
    finally:
        engine.cleanup()

    data = (output / "clean_candidates.json").read_text(encoding="utf-8")
    assert '"type": "subtitle"' in data
    assert '"selected": true' in data


def test_clean_region_preview_skips_text_detector_and_writes_region(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    class FakeDetector:
        def detect(self, frame):
            return []

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            regions=["top-right"],
        )
    finally:
        engine.cleanup()

    data = (output / "clean_candidates.json").read_text(encoding="utf-8")
    assert '"type": "region"' in data
    assert '"label": "top-right region"' in data
    assert '"selected": true' in data


def test_clean_target_phrase_infers_region_and_logo(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"

    def draw_logo(frame, i):
        cv2.rectangle(frame, (74, 4), (90, 16), (255, 255, 255), 1)
        cv2.line(frame, (76, 14), (88, 6), (255, 255, 255), 1)

    _write_test_video(video, draw=draw_logo)
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    class FakeDetector:
        def detect(self, frame):
            return []

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            targets=["右上角台标"],
        )
    finally:
        engine.cleanup()

    data = (output / "clean_candidates.json").read_text(encoding="utf-8")
    assert '"type": "logo"' in data
    assert '"type": "region"' in data


def test_clean_watermark_target_enables_translucent_watermark_scan(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"

    def draw_watermark(frame, i):
        cv2.putText(
            frame,
            "WATER",
            (28, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    _write_test_video(video, draw=draw_watermark)
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    class FakeDetector:
        def detect(self, frame):
            return []

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            targets=["watermark"],
        )
    finally:
        engine.cleanup()

    data = (output / "clean_candidates.json").read_text(encoding="utf-8")
    assert '"possible translucent center watermark"' in data


def test_stable_center_overlay_is_default_watermark_and_includes_icon(tmp_path):
    video = tmp_path / "input.mp4"

    def draw_overlay(frame, _i):
        cv2.polylines(
            frame,
            [np.array([[78, 91], [96, 78], [105, 99], [81, 106]], np.int32)],
            True,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            frame, "AI", (114, 99), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (220, 220, 220), 2, cv2.LINE_AA,
        )

    _write_test_video(video, width=256, height=192, frames=5, draw=draw_overlay)

    class TextOnlyDetector:
        def detect(self, frame):
            return [TextBox(
                points=np.array([[112, 78], [165, 78], [165, 99], [112, 99]]),
                confidence=0.9,
            )]

    result = detect_clean_candidates(
        str(video), detector=TextOnlyDetector(), sample_count=5,
    )
    candidate = result.candidates[0]

    assert candidate.type == "watermark"
    assert candidate.default_remove is True
    assert select_clean_candidates(result.candidates) == [candidate]
    # The DBNet polygon starts at x=112 and its normal dilation at x=98;
    # pixels left of that prove the adjacent non-text icon expanded the mask.
    assert candidate.mask[76:108, 76:98].sum() > 0
    mask_ys, mask_xs = np.where(candidate.mask[:, :, 0] > 0)
    assert candidate.bbox == (
        int(mask_xs.min()), int(mask_ys.min()),
        int(mask_xs.max()), int(mask_ys.max()),
    )


def test_persistent_center_title_without_adjacent_graphic_stays_scene_text(tmp_path):
    video = tmp_path / "input.mp4"

    def draw_title(frame, _i):
        cv2.putText(
            frame, "TITLE", (40, 53), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (220, 220, 220), 1, cv2.LINE_AA,
        )

    _write_test_video(
        video, width=128, height=96, frames=5, draw=draw_title,
    )

    class PersistentTitleDetector:
        def detect(self, frame):
            return [TextBox(
                points=np.array([[44, 39], [84, 39], [84, 55], [44, 55]]),
                confidence=0.9,
            )]

    result = detect_clean_candidates(
        str(video), detector=PersistentTitleDetector(), sample_count=5,
    )
    candidate = result.candidates[0]

    assert candidate.presence_frames == [0, 1, 2, 3, 4]
    assert candidate.type == "scene_text"
    assert candidate.default_remove is False
    assert select_clean_candidates(result.candidates) == []

    targeted = detect_clean_candidates(
        str(video), detector=PersistentTitleDetector(), sample_count=5,
        include_translucent_watermark=True,
    )
    assert any(item.type == "watermark" for item in targeted.candidates)


def test_clean_timestamp_target_warns_when_detector_has_no_text(tmp_path, capsys):
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[2, 2], [44, 2], [44, 10], [2, 10]]),
                    confidence=0.95,
                )
            ]

    engine = WipeEngine(task="clean", detector=FakeDetector())
    try:
        engine.process(
            video=str(video),
            output=str(output),
            preview=True,
            targets=["timestamp"],
        )
    finally:
        engine.cleanup()

    assert "Timestamp detection requires recognized text content" in capsys.readouterr().out


def test_local_agent_selector_returns_valid_ids(monkeypatch):
    candidate = CleanCandidate(
        id="c1", type="subtitle", label="bottom subtitle",
        bbox=(2, 15, 27, 17), confidence=0.9, frame_fraction=1.0,
        reason="wide bottom text", default_remove=True,
    )

    class Result:
        returncode = 0
        stdout = '{"remove":["c1"]}'

    monkeypatch.setattr(agent_module.shutil, "which", lambda command: command)
    monkeypatch.setattr(agent_module.subprocess, "run", lambda *args, **kwargs: Result())

    assert agent_module.select_with_agent("codex", [candidate], "去掉字幕") == ["c1"]


def test_local_agent_selector_rejects_invalid_ids(monkeypatch):
    candidate = CleanCandidate(
        id="c1", type="subtitle", label="bottom subtitle",
        bbox=(2, 15, 27, 17), confidence=0.9, frame_fraction=1.0,
        reason="wide bottom text", default_remove=True,
    )

    class Result:
        returncode = 0
        stdout = '{"remove":["c9"]}'

    monkeypatch.setattr(agent_module.shutil, "which", lambda command: command)
    monkeypatch.setattr(agent_module.subprocess, "run", lambda *args, **kwargs: Result())

    assert agent_module.select_with_agent("codex", [candidate], "去掉字幕") is None


def test_iou_bbox_overlap_and_disjoint():
    assert _iou_bbox((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou_bbox((0, 0, 5, 5), (5, 5, 10, 10)) == pytest.approx(0.0, abs=0.05)
    assert 0 < _iou_bbox((0, 0, 8, 8), (4, 4, 12, 12)) < 1


def test_band_fallback_catches_missed_bottom_subtitle(tmp_path):
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    class BandOnlyDetector:
        """Returns text only on short (cropped) frames, simulating DBNet miss."""

        def detect(self, frame):
            h = frame.shape[0]
            if h < 50:
                w = frame.shape[1]
                return [
                    TextBox(
                        points=np.array([[2, h - 10], [w - 2, h - 10],
                                         [w - 2, h - 2], [2, h - 2]]),
                        confidence=0.9,
                        text="hidden subtitle",
                    )
                ]
            return []

    # Without fallback: no candidates (main detection sees nothing)
    result_off = detect_clean_candidates(
        str(video), detector=BandOnlyDetector(), sample_count=3, subtitle_fallback="off",
    )
    assert len(result_off.candidates) == 0

    # With light fallback: should find the hidden subtitle
    result_light = detect_clean_candidates(
        str(video), detector=BandOnlyDetector(), sample_count=3, subtitle_fallback="light",
    )
    types = {c.type for c in result_light.candidates}
    assert "subtitle" in types
    assert any("band fallback" in c.reason for c in result_light.candidates)


def test_band_fallback_deduplicates_against_main_detection(tmp_path):
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    class AlwaysDetectBottom:
        """Returns the same bottom text on both full frame and crop."""

        def detect(self, frame):
            h, w = frame.shape[:2]
            return [
                TextBox(
                    points=np.array([[2, h - 10], [w - 2, h - 10],
                                     [w - 2, h - 2], [2, h - 2]]),
                    confidence=0.9,
                    text="subtitle",
                )
            ]

    # Main detection already finds it; fallback should not duplicate
    result = detect_clean_candidates(
        str(video), detector=AlwaysDetectBottom(), sample_count=3, subtitle_fallback="light",
    )
    # Should have exactly one candidate (from main detection), no fallback duplicate
    assert len(result.candidates) == 1
    assert result.candidates[0].id.startswith("c")


# --- Benchmark / timing tests ---


def test_engine_writes_benchmark_json_on_manual_mask(tmp_path, monkeypatch):
    """WipeEngine.process() writes benchmark.json with timing data."""
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    mask = tmp_path / "mask.png"
    _write_test_video(video)

    # Create a valid mask (96x64)
    mask_arr = np.zeros((64, 96), dtype=np.uint8)
    mask_arr[50:60, 10:86] = 255
    cv2.imwrite(str(mask), mask_arr)

    # Stub model loading and processing
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    class FakeTask:
        _bm = None
        backend = type("B", (), {"__name__": "FakeBackend"})()
        output_suffix = "detext"

        def process_video(self, reader, frame_info, mask_arr, output_dir, video_path=""):
            self._bm["timing"]["inpainting_s"] = 0.001
            return os.path.join(output_dir, "output_detext.mp4")

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "videowipe.engine._TASK_CLASSES", {"detext": lambda **kw: FakeTask()}
    )

    engine = WipeEngine(task="detext")
    try:
        engine.process(video=video, mask=mask, output=output)
    finally:
        engine.cleanup()

    bm_path = output / "benchmark.json"
    assert bm_path.exists(), "benchmark.json should be written"
    bm = json.loads(bm_path.read_text(encoding="utf-8"))
    assert bm["mask_source"] == "manual"
    assert "total_s" in bm["timing"]
    assert "model_load_s" in bm["timing"]
    assert "inpainting_s" in bm["timing"]
    assert bm["width"] == 96
    assert bm["height"] == 64
    assert bm["video_path"] == str(video)
    assert bm["output_path"] == str(output / "output_detext.mp4")
    assert bm["error"] is None


def test_engine_writes_benchmark_json_on_error(tmp_path, monkeypatch):
    """benchmark.json is written even when processing fails."""
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    class FailingTask:
        _bm = None
        backend = type("B", (), {"__name__": "FakeBackend"})()

        def process_video(self, reader, frame_info, mask_arr, output_dir, video_path=""):
            raise RuntimeError("inpainting failed")

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "videowipe.engine._TASK_CLASSES", {"detext": lambda **kw: FailingTask()}
    )

    # Create a minimal mask file
    mask = tmp_path / "mask.png"
    mask_arr = np.zeros((64, 96), dtype=np.uint8)
    mask_arr[50:60, 10:86] = 255
    cv2.imwrite(str(mask), mask_arr)

    engine = WipeEngine(task="detext")
    with pytest.raises(RuntimeError, match="inpainting failed"):
        engine.process(video=str(video), mask=str(mask), output=str(output))
    engine.cleanup()

    bm_path = output / "benchmark.json"
    assert bm_path.exists()
    bm = json.loads(bm_path.read_text(encoding="utf-8"))
    assert bm["error"] == "inpainting failed"
    assert "total_s" in bm["timing"]


# --- External model adapter tests ---


def _fake_external_cmd():
    """Return a cross-platform command that copies arg1 to arg3 (video -> output_dir)."""
    return (
        f'{sys.executable} -c "import shutil,sys;'
        f'shutil.copy(sys.argv[1],sys.argv[3])"'
    )


def test_run_external_success(tmp_path):
    """Fake command copies input video to output dir; adapter discovers it."""
    video = tmp_path / "input.mp4"
    _write_test_video(video)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(mask), np.zeros((64, 96), dtype=np.uint8))

    result = run_external(_fake_external_cmd(), str(video), str(mask), str(output_dir))
    assert os.path.exists(result)
    assert result.startswith(str(output_dir))


def test_run_external_nonzero_exit(tmp_path):
    """Non-zero exit raises ExternalModelError with stderr."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # list-form invocation cannot run shell builtins like `exit`; use a real
    # executable that exits non-zero.
    fail_cmd = f'{sys.executable} -c "import sys; sys.exit(1)"'

    with pytest.raises(ExternalModelError, match="exited with code"):
        run_external(fail_cmd, "video.mp4", "mask.png", str(output_dir))


def test_run_external_invokes_subprocess_without_shell(monkeypatch, tmp_path):
    """Injection fix: the command runs as an argv list with shell disabled, so
    shell metacharacters are never interpreted and paths stay verbatim."""
    captured = {}

    class _OK:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["shell"] = kwargs.get("shell", False)
        out_dir = cmd[-1]
        os.makedirs(out_dir, exist_ok=True)
        # run_external only checks the file extension; create a placeholder.
        with open(os.path.join(out_dir, "out.mp4"), "wb"):
            pass
        return _OK()

    monkeypatch.setattr("videowipe.external.subprocess.run", spy)
    run_external(
        "python propainter.py --fp16",
        "my video.mp4",        # space in name
        "m; rm -rf x.png",     # shell metacharacters
        str(tmp_path / "out"),
    )

    assert captured["shell"] is False
    assert isinstance(captured["cmd"], list)
    # Command split by shlex; paths appended verbatim as single argv entries.
    assert captured["cmd"][:3] == ["python", "propainter.py", "--fp16"]
    assert captured["cmd"][-3:] == [
        "my video.mp4", "m; rm -rf x.png", str(tmp_path / "out"),
    ]


def test_run_external_missing_command(tmp_path):
    """A command whose executable does not exist raises ExternalModelError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(ExternalModelError, match="not found"):
        run_external(
            "nonexistent-binary-xyz-12345", "v.mp4", "m.png", str(output_dir)
        )


def test_run_external_no_output(tmp_path):
    """Command succeeds but produces no video file raises ExternalModelError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Python no-op that exits 0 but writes nothing
    noop = f'{sys.executable} -c "pass"'

    with pytest.raises(ExternalModelError, match="no output video"):
        run_external(noop, "video.mp4", "mask.png", str(output_dir))


def test_engine_external_command_skips_model_loading(tmp_path, monkeypatch):
    """Engine with external_command set never calls _ensure_model."""
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    mask = tmp_path / "mask.png"
    mask_arr = np.zeros((64, 96), dtype=np.uint8)
    mask_arr[50:60, 10:86] = 255
    cv2.imwrite(str(mask), mask_arr)

    def fail_model_load(self):
        raise AssertionError("external mode should not load model")

    monkeypatch.setattr(WipeEngine, "_ensure_model", fail_model_load)

    engine = WipeEngine(task="detext", external_command=_fake_external_cmd())
    try:
        out_path = engine.process(
            video=str(video), mask=str(mask), output=str(output)
        )
    finally:
        engine.cleanup()

    assert os.path.exists(out_path)
    assert out_path.startswith(str(output))


def test_engine_external_command_benchmark_json(tmp_path, monkeypatch):
    """benchmark.json has model_type: external and external_s timing."""
    video = tmp_path / "input.mp4"
    output = tmp_path / "result"
    _write_test_video(video)

    mask = tmp_path / "mask.png"
    mask_arr = np.zeros((64, 96), dtype=np.uint8)
    mask_arr[50:60, 10:86] = 255
    cv2.imwrite(str(mask), mask_arr)

    engine = WipeEngine(task="detext", external_command=_fake_external_cmd())
    try:
        engine.process(video=str(video), mask=str(mask), output=str(output))
    finally:
        engine.cleanup()

    bm_path = output / "benchmark.json"
    assert bm_path.exists()
    bm = json.loads(bm_path.read_text(encoding="utf-8"))
    assert bm["model_type"] == "external"
    assert bm["backend"] == "external"
    assert "external_s" in bm["timing"]
    assert bm["error"] is None


# --- Detect mode tests ---


def test_resolve_detect_params_maps_modes():
    fast = resolve_detect_params("fast")
    assert fast["sample_count"] == 24
    assert fast["consistency"] == 0.50
    assert fast["subtitle_fallback"] == "off"

    balanced = resolve_detect_params("balanced")
    assert balanced["sample_count"] == 50
    assert balanced["consistency"] == 0.40
    assert balanced["subtitle_fallback"] == "light"

    sensitive = resolve_detect_params("sensitive")
    assert sensitive["sample_count"] == 80
    assert sensitive["consistency"] == 0.30
    assert sensitive["subtitle_fallback"] == "force"


def test_resolve_detect_params_subtitle_override():
    """When has_subtitle_target is True, subtitle_fallback upgrades to force."""
    fast = resolve_detect_params("fast", has_subtitle_target=True)
    assert fast["subtitle_fallback"] == "force"

    balanced = resolve_detect_params("balanced", has_subtitle_target=True)
    assert balanced["subtitle_fallback"] == "force"


def test_resolve_detect_params_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown detect mode"):
        resolve_detect_params("turbo")


def test_cli_detect_mode_choices():
    parser = _build_parser()

    # Valid modes
    for mode in ("fast", "balanced", "sensitive"):
        args = parser.parse_args(["clean", "input.mp4", "--detect-mode", mode])
        assert args.detect_mode == mode

    # Default is balanced
    args = parser.parse_args(["clean", "input.mp4"])
    assert args.detect_mode == "balanced"

    # Invalid mode rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["clean", "input.mp4", "--detect-mode", "turbo"])


def test_cli_ocr_choices():
    parser = _build_parser()

    for mode in ("auto", "off", "rapidocr"):
        args = parser.parse_args(["clean", "input.mp4", "--ocr", mode])
        assert args.ocr == mode

    # Default is auto
    args = parser.parse_args(["clean", "input.mp4"])
    assert args.ocr == "auto"


def test_detect_mode_fast_uses_fewer_samples(tmp_path, monkeypatch):
    """Fast mode should call detect_clean_candidates with sample_count=24."""
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="test subtitle",
                )
            ]

    captured = {}
    original_detect = detect_clean_candidates

    def spy_detect(*args, **kwargs):
        captured.update(kwargs)
        return original_detect(*args, **kwargs)

    monkeypatch.setattr("videowipe.detect.detect_clean_candidates", spy_detect)
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    engine = WipeEngine(task="clean", detector=FakeDetector(), detect_mode="fast")
    try:
        engine.process(video=str(video), output=str(tmp_path / "result"), preview=True)
    finally:
        engine.cleanup()

    assert captured["sample_count"] == 24
    assert captured["consistency"] == 0.50


def test_detect_mode_sensitive_uses_more_samples(tmp_path, monkeypatch):
    """Sensitive mode should call detect_clean_candidates with sample_count=80."""
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="test subtitle",
                )
            ]

    captured = {}
    original_detect = detect_clean_candidates

    def spy_detect(*args, **kwargs):
        captured.update(kwargs)
        return original_detect(*args, **kwargs)

    monkeypatch.setattr("videowipe.detect.detect_clean_candidates", spy_detect)
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    engine = WipeEngine(task="clean", detector=FakeDetector(), detect_mode="sensitive")
    try:
        engine.process(video=str(video), output=str(tmp_path / "result"), preview=True)
    finally:
        engine.cleanup()

    assert captured["sample_count"] == 80
    assert captured["consistency"] == 0.30


# --- OCR tests ---


def test_ocr_off_builds_no_recognizer():
    """ocr='off' should always return None recognizer."""
    assert WipeEngine._build_recognizer("off") is None


def test_ocr_auto_degrades_gracefully(monkeypatch):
    """ocr='auto' returns None when rapidocr is not installed."""
    import sys
    # Remove ocr module if present to force re-import path
    monkeypatch.setitem(sys.modules, "videowipe.ocr", None)
    # The from import in _build_recognizer will hit videowipe.ocr which is None,
    # causing ImportError. Auto mode should catch and return None.
    result = WipeEngine._build_recognizer("auto")
    assert result is None


def test_ocr_rapidocr_raises_when_missing(monkeypatch):
    """ocr='rapidocr' should raise RuntimeError when rapidocr is not installed."""
    import sys
    monkeypatch.setitem(sys.modules, "videowipe.ocr", None)
    with pytest.raises(RuntimeError, match="rapidocr-onnxruntime"):
        WipeEngine._build_recognizer("rapidocr")


def test_recognizer_fills_text_samples(tmp_path):
    """A fake recognizer should populate text_samples on candidates."""
    video = tmp_path / "input.mp4"
    _write_test_video(video, width=96, height=64)

    def fake_recognizer(crop):
        return "recognized text"

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="",
                )
            ]

    result = detect_clean_candidates(
        str(video),
        detector=FakeDetector(),
        sample_count=3,
        recognizer=fake_recognizer,
    )

    assert len(result.candidates) >= 1
    candidate = result.candidates[0]
    assert "recognized text" in candidate.text_samples


def test_recognizer_skipped_when_text_already_present(tmp_path):
    """When box.text is set, recognizer should not be called."""
    video = tmp_path / "input.mp4"
    _write_test_video(video, width=96, height=64)
    call_count = 0

    def counting_recognizer(crop):
        nonlocal call_count
        call_count += 1
        return "ocr result"

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="already has text",
                )
            ]

    result = detect_clean_candidates(
        str(video),
        detector=FakeDetector(),
        sample_count=3,
        recognizer=counting_recognizer,
    )

    assert call_count == 0
    assert result.candidates[0].text_samples == ["already has text"]


def test_engine_detect_mode_override_in_process(tmp_path, monkeypatch):
    """detect_mode passed to process() overrides the constructor default."""
    video = tmp_path / "input.mp4"
    _write_test_video(video)

    captured = {}

    class FakeDetector:
        def detect(self, frame):
            return [
                TextBox(
                    points=np.array([[10, 52], [86, 52], [86, 60], [10, 60]]),
                    confidence=0.9,
                    text="subtitle",
                )
            ]

    original_detect = detect_clean_candidates

    def spy_detect(*args, **kwargs):
        captured.update(kwargs)
        return original_detect(*args, **kwargs)

    monkeypatch.setattr("videowipe.detect.detect_clean_candidates", spy_detect)
    monkeypatch.setattr(WipeEngine, "_ensure_model", lambda self: None)

    # Constructor says fast, process() says sensitive
    engine = WipeEngine(task="clean", detector=FakeDetector(), detect_mode="fast")
    try:
        engine.process(
            video=str(video),
            output=str(tmp_path / "result"),
            preview=True,
            detect_mode="sensitive",
        )
    finally:
        engine.cleanup()

    assert captured["sample_count"] == 80
    assert captured["consistency"] == 0.30


# --- ExternalInpainter unit tests ---


def test_external_inpainter_requires_nonempty_command():
    """ExternalInpainter rejects an empty command string."""
    with pytest.raises(ValueError, match="non-empty command"):
        ExternalInpainter(command="")


def test_external_inpainter_requires_mask_path():
    """ExternalInpainter.inpaint raises if job.mask_path is missing."""
    from videowipe.inpainters.base import InpaintJob

    inpainter = ExternalInpainter(command="echo noop")
    job = InpaintJob(
        video_path="v.mp4",
        mask=np.zeros((4, 4, 1), dtype=np.uint8),
        output_dir=".",
        fps=0.0,
        frame_count=0,
        width=0,
        height=0,
    )
    with pytest.raises(ValueError, match="mask_path"):
        inpainter.inpaint(job)


# --- A0: soft alpha mask + audio retention + progress callback ---


def test_mask_from_candidates_feather_radius_produces_continuous_alpha():
    """feather_radius > 0 yields a float32 mask with values in [0, 1],
    not the legacy binary uint8 {0, 1}. The bbox interior stays at 1.0;
    the seam outside the bbox contains intermediate values from the
    Gaussian falloff."""
    candidates = [
        CleanCandidate(
            id="c1", type="subtitle", label="bottom subtitle",
            bbox=(5, 15, 25, 18), confidence=0.9, frame_fraction=1.0,
            reason="subtitle band", default_remove=True,
        ),
    ]
    mask = mask_from_candidates(candidates, (20, 30), feather_radius=3)
    # Continuous alpha: float32, contains values strictly between 0 and 1
    assert mask.dtype == np.float32
    assert 0.0 < mask.max() <= 1.0
    # The bbox GEOMETRIC CENTER stays fully opaque (far enough from the
    # Gaussian-blurred edge). bbox y in [15,18], x in [5,25] → center (16, 15).
    # Gaussian blur erodes edges inward, so use a point at least one radius
    # away from every bbox edge: y=16 (1px from top edge 15) is too close;
    # pick the middle of the bbox interior instead.
    assert mask[16, 14, 0] == 1.0 or mask[16, 16, 0] == 1.0, (
        f"bbox interior should stay opaque; got center={mask[16, 14, 0]}"
    )
    # Somewhere outside the bbox there is a soft falloff value in (0, 1)
    flat = mask[:, :, 0]
    soft_pixels = flat[(flat > 0.0) & (flat < 1.0)]
    assert soft_pixels.size > 0


def test_mask_from_candidates_feather_radius_zero_keeps_binary_uint8():
    """feather_radius == 0 keeps the legacy binary uint8 {0, 1} output so
    the eval IoU path continues to compare against binary ground-truth."""
    candidates = [
        CleanCandidate(
            id="c1", type="subtitle", label="bottom subtitle",
            bbox=(5, 15, 25, 18), confidence=0.9, frame_fraction=1.0,
            reason="subtitle band", default_remove=True,
        ),
    ]
    mask = mask_from_candidates(candidates, (20, 30), feather_radius=0)
    assert mask.dtype == np.uint8
    assert mask[16, 15, 0] == 1
    # No intermediate values when feathering is disabled
    flat = mask[:, :, 0]
    soft_pixels = flat[(flat > 0) & (flat < 1)]
    assert soft_pixels.size == 0


def test_mask_from_candidates_feathers_candidate_with_premask():
    """Regression: when a candidate carries a precomputed binary ``mask``
    (as the clean-task detector emits for every candidate), feather_radius > 0
    must still soften the merged mask's outer boundary. The first A0
    implementation only feathered bbox-only candidates and was a no-op here,
    so feather=0 and feather=4 produced byte-identical output. This test pins
    the fix: feathering happens on the final merged mask, not per-candidate.
    """
    h, w = 20, 30
    # Candidate with a full-image binary mask (the real-world detector shape):
    # a solid block at rows 15-18, cols 5-25.
    premask = np.zeros((h, w, 1), dtype=np.uint8)
    premask[15:19, 5:26, 0] = 1
    candidate = CleanCandidate(
        id="c1", type="subtitle", label="bottom subtitle",
        bbox=(5, 15, 25, 18), confidence=0.9, frame_fraction=1.0,
        reason="subtitle band", default_remove=True,
        mask=premask,
    )
    hard = mask_from_candidates([candidate], (h, w), feather_radius=0)
    soft = mask_from_candidates([candidate], (h, w), feather_radius=4)

    # Hard path: binary uint8
    assert hard.dtype == np.uint8
    assert hard[16, 15, 0] == 1

    # Soft path: float32 with intermediate values OUTSIDE the original block,
    # and the interior pinned back to 1.0.
    assert soft.dtype == np.float32
    assert soft[16, 15, 0] == 1.0  # interior unchanged
    flat = soft[:, :, 0]
    soft_pixels = flat[(flat > 0.0) & (flat < 1.0)]
    assert soft_pixels.size > 0, (
        "feather_radius>0 must soften the merged mask boundary even when "
        "the candidate carries a precomputed mask"
    )
    # The hard mask is pure binary {0,1}; the soft mask has intermediate
    # values. They cannot be equal as arrays.
    assert not np.array_equal(soft, hard.astype(np.float32))


def test_inpaint_job_defaults_to_static_mask():
    job = InpaintJob(
        video_path="v.mp4",
        mask=np.zeros((4, 4, 1), dtype=np.uint8),
        output_dir=".",
        fps=0.0,
        frame_count=0,
        width=0,
        height=0,
    )
    assert job.feather_radius == 0
    assert job.frame_mask is None
    assert job.progress is None


def test_engine_sets_nonzero_feather_radius_by_default():
    """WipeEngine applies a non-zero feather_radius to its task impl so the
    STTN blend produces a soft seam in the default run path (the eval path
    opts out by passing feather_radius=0 explicitly)."""
    from videowipe.engine import _DEFAULT_FEATHER_RADIUS

    engine = WipeEngine(task="detext")
    try:
        assert _DEFAULT_FEATHER_RADIUS > 0
        assert engine._task_impl.feather_radius == _DEFAULT_FEATHER_RADIUS
    finally:
        engine.cleanup()


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_sttn_inpaint_preserves_audio_and_reports_progress(tmp_path, monkeypatch):
    """End-to-end A0 verification of the three "finished feel" fixes against
    a real ffmpeg:

    1. The ffmpeg command built by STTNInpainter maps the original video's
       audio stream (``-map 1:a?`` and ``-c:a aac``).
    2. The ``job.progress`` callback is invoked at least once during the
       segment loop.
    3. The soft-alpha blend does not raise on a float32 mask.

    A fake backend returns zero-filled predicted frames so the real STTN
    segment loop and ffmpeg pipe are exercised without loading model weights.
    """
    from videowipe.inpainters.sttn import STTNInpainter

    # Build a tiny test video WITH an audio track so the -map 1:a? path has
    # a real stream to attach. 96x64, 8 frames, silent aac track.
    video = tmp_path / "input.mp4"
    _write_test_video(
        video, width=96, height=64, frames=8,
        draw=lambda frame, i: frame.__setitem__(
            (slice(50, 60), slice(10, 86)), 200),
    )
    # Re-encode with a silent audio track so ffprobe can find a stream.
    import subprocess as _sp
    audio_video = tmp_path / "input_audio.mp4"
    _sp.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video),
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest",
         str(audio_video)],
        check=True,
    )

    # Fake backend: real preprocess, no-op model, zero prediction frames.
    class _FakeBackend:
        __name__ = "FakeBackend"

        def preprocess(self, frames):
            from videowipe.backends import InpaintBackend
            return InpaintBackend.preprocess(self, frames)

        def encode(self, tensor):
            return np.zeros((tensor.shape[0], 1, 1, 1), dtype=np.float32)

        def transform(self, feats):
            return feats

        def decode(self, feats):
            t = feats.shape[0]
            return np.zeros((t, 120, 640, 3), dtype=np.uint8)

        def cleanup(self):
            pass

    inpainter = STTNInpainter()
    inpainter.backend = _FakeBackend()

    captured_cmds = []
    real_popen = _sp.Popen

    def _spy_popen(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr("videowipe.inpainters.sttn.subprocess.Popen", _spy_popen)

    progress_calls = []

    def _progress(done, total):
        progress_calls.append((done, total))

    reader = cv2.VideoCapture(str(audio_video))
    try:
        mask = np.zeros((64, 96, 1), dtype=np.float32)
        mask[50:60, 10:86, 0] = 1.0
        job = InpaintJob(
            video_path=str(audio_video),
            mask=mask,
            output_dir=str(tmp_path),
            fps=4.0,
            frame_count=8,
            width=96,
            height=64,
            reader=reader,
            progress=_progress,
            gap=8,
        )
        outcome = inpainter.inpaint(job)
    finally:
        reader.release()
        inpainter.cleanup()

    # 1. ffmpeg command mapped the original audio stream
    assert captured_cmds, "ffmpeg was not invoked"
    flat_args = [a for cmd in captured_cmds for a in cmd]
    assert "-map" in flat_args and "1:a?" in flat_args
    assert "-c:a" in flat_args and "aac" in flat_args

    # 2. progress callback fired at least once, final call frames_done == total
    assert progress_calls, "progress callback was never invoked"
    assert progress_calls[-1][0] == 8
    assert progress_calls[-1][1] == 8

    # 3. Output mp4 contains an audio stream
    probe = _sp.run(
        ["ffmpeg", "-i", outcome.output_path, "-hide_banner"],
        capture_output=True, text=True, check=False,
    )
    diagnostic = probe.stderr + probe.stdout
    assert "Audio:" in diagnostic, (
        f"output has no audio stream\n{diagnostic}"
    )


def test_get_inpaint_mode_uses_an_unprocessed_frontier():
    from videowipe.inpainters.sttn import get_inpaint_mode

    height, split_h = 1080, 360
    mask = np.zeros((height, 16), dtype=np.uint8)
    mask[355:375] = 1
    mask[715:735] = 1

    modes = get_inpaint_mode(height, split_h, mask)

    assert modes == [(720, 1080), (360, 720), (0, 360)]
    active_rows = np.flatnonzero(mask.any(axis=1))
    assert all(any(start <= row < end for start, end in modes) for row in active_rows)
    assert all(end - start == split_h for start, end in modes)
    assert all(
        max(0, min(first_end, second_end) - max(first_start, second_start)) < split_h // 4
        for first_start, first_end in modes
        for second_start, second_end in modes
        if (first_start, first_end) != (second_start, second_end)
    )


def test_get_inpaint_mode_shifts_only_when_top_edge_is_clear():
    from videowipe.inpainters.sttn import get_inpaint_mode

    mask = np.zeros((1080, 16), dtype=np.uint8)
    mask[715:735] = 1

    assert get_inpaint_mode(1080, 360, mask) == [(720, 1080), (375, 735)]


def test_get_inpaint_mode_clamps_window_when_height_is_less_than_split():
    from videowipe.inpainters.sttn import get_inpaint_mode

    mask = np.ones((120, 16), dtype=np.uint8)

    modes = get_inpaint_mode(120, 360, mask)

    assert modes == [(0, 120)]
    assert all(0 <= start < end <= 120 for start, end in modes)


def test_blend_frame_regions_matches_whole_frame_float_reference():
    from videowipe.inpainters.sttn import _blend_frame_regions

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    original = frame.copy()
    modes = [(2, 12), (8, 18)]
    comps = [
        rng.integers(0, 256, (10, 32, 3), dtype=np.uint8).astype(np.float32),
        rng.integers(0, 256, (10, 32, 3), dtype=np.uint8).astype(np.float32),
    ]
    mask = np.zeros((24, 32, 1), dtype=np.float32)
    mask[2:12, 5:25] = rng.random((10, 20, 1), dtype=np.float32)
    mask[8:18, 8:28] = rng.random((10, 20, 1), dtype=np.float32)

    reference = frame.astype(np.float32)
    for (start, end), comp in zip(modes, comps):
        alpha = mask[start:end]
        reference[start:end] = (
            alpha * comp + (1.0 - alpha) * reference[start:end]
        )
    reference = np.clip(reference, 0, 255).astype(np.uint8)

    actual = _blend_frame_regions(frame, comps, modes, mask)

    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(actual[:2], frame[:2])
    np.testing.assert_array_equal(actual[18:], frame[18:])
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize(
    "temporal,production,output_count",
    [
        (False, False, 3),
        (False, True, 3),
        (True, True, 3),
        (False, True, 2),
    ],
)
def test_sttn_preserves_model_input_before_preprocess(
    tmp_path, monkeypatch, temporal, production, output_count,
):
    """STTN sees source pixels; masks are applied only to the final blend."""
    from videowipe.inpainters.sttn import STTNInpainter

    height, width = 8, 64  # split_h=12 exercises H < split_h too.
    frames = [
        np.full((height, width, 3), (30 + i * 10, 90, 180), np.uint8)
        for i in range(3)
    ]
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[:, 12:52] = 0.5
    mask[1:7, 18:46] = 1.0
    captured = []
    mask_indices = []

    class Reader:
        def __init__(self, source_frames):
            self.frames = source_frames
            self.index = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if self.index == len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()

        def release(self):
            self.released = True

    class SpyBackend:
        def preprocess(self, model_frames):
            captured.extend(frame.copy() for frame in model_frames)
            return np.zeros((len(model_frames), 1, 1, 1), dtype=np.float32)

        def encode(self, tensor):
            return tensor

        def transform(self, feats):
            return feats

        def decode(self, feats):
            return np.zeros((len(feats), 120, 640, 3), dtype=np.uint8)

        def cleanup(self):
            pass

    class Sink:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)
            return len(data)

        def close(self):
            pass

    class Pipe:
        def __init__(self):
            self.returncode = 0
            self.stdin = Sink()

        def wait(self):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

    pipe = Pipe()
    monkeypatch.setattr(
        "videowipe.inpainters.sttn.subprocess.Popen",
        lambda *_args, **_kwargs: pipe,
    )
    output_reader = Reader(frames[:output_count])
    if production:
        monkeypatch.setattr(
            "videowipe.inpainters.sttn._VIDEO_CAPTURE_TYPE", Reader,
        )
        monkeypatch.setattr(
            "videowipe.inpainters.sttn.cv2.VideoCapture",
            lambda _path: output_reader,
        )

    def frame_mask(global_index):
        mask_indices.append(global_index)
        if global_index == 1:
            return mask[:, :, 0]
        return np.zeros((height, width), dtype=np.float32)

    inpainter = STTNInpainter()
    inpainter.backend = SpyBackend()
    inference_reader = Reader(frames)
    job = InpaintJob(
        video_path="input.mp4" if production else "", mask=mask,
        output_dir=str(tmp_path), fps=4.0, frame_count=len(frames),
        width=width, height=height, reader=inference_reader, gap=2,
        frame_mask=frame_mask if temporal else None,
    )
    if output_count < len(frames):
        with pytest.raises(ValueError, match=r"output decoded 2 frames; expected 3"):
            inpainter.inpaint(job)
        assert output_reader.released
        return
    inpainter.inpaint(job)

    assert len(captured) == len(frames)
    assert inference_reader.index == len(frames)
    for source, model_input in zip(frames, captured):
        expected = cv2.resize(
            source, (640, 120), interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        assert np.array_equal(model_input, expected)
    if temporal:
        assert mask_indices == [0, 1, 2]
    if production:
        assert output_reader.index == len(frames)
        assert output_reader.released
    output = np.frombuffer(pipe.stdin.data, dtype=np.uint8).reshape(
        len(frames), height, width, 3,
    )
    for index, source in enumerate(frames):
        active = not temporal or index == 1
        alpha = mask if active else np.zeros_like(mask)
        expected = np.clip((1.0 - alpha) * source, 0, 255).astype(np.uint8)
        assert np.array_equal(output[index], expected)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
def test_sttn_frame_mask_blends_only_active_frames_across_segments(tmp_path, monkeypatch):
    """A temporal frame_mask active on [10,30) blends exactly those frames,
    including across gap-bounded segments (no start_f+j off-by-one).

    FakeBackend returns zero composites, so active frames become black in the
    masked band while inactive frames pass through unchanged.
    """
    from videowipe.inpainters.sttn import STTNInpainter

    H, W, N = 64, 96, 60
    band = (slice(50, 60), slice(10, 86))
    video = tmp_path / "input.mp4"
    _write_test_video(
        video, width=W, height=H, frames=N,
        draw=lambda frame, i: frame.__setitem__(band, 200),
    )

    class _ZeroBackend:
        __name__ = "FakeBackend"

        def preprocess(self, frames):
            from videowipe.backends import InpaintBackend
            return InpaintBackend.preprocess(self, frames)

        def encode(self, tensor):
            return np.zeros((tensor.shape[0], 1, 1, 1), dtype=np.float32)

        def transform(self, feats):
            return feats

        def decode(self, feats):
            return np.zeros((feats.shape[0], 120, 640, 3), dtype=np.uint8)

        def cleanup(self):
            pass

    inpainter = STTNInpainter()
    inpainter.backend = _ZeroBackend()

    def frame_mask(global_idx):
        m = np.zeros((H, W), dtype=np.uint8)
        if 10 <= global_idx < 30:
            m[50:60, 10:86] = 1
        return m

    static = np.zeros((H, W, 1), dtype=np.uint8)
    static[50:60, 10:86, 0] = 1  # lets get_inpaint_mode find the band

    reader = cv2.VideoCapture(str(video))
    try:
        job = InpaintJob(
            video_path=str(video), mask=static, output_dir=str(tmp_path),
            fps=4.0, frame_count=N, width=W, height=H,
            reader=reader, gap=15, frame_mask=frame_mask,
        )
        outcome = inpainter.inpaint(job)
    finally:
        reader.release()
        inpainter.cleanup()

    cap = cv2.VideoCapture(outcome.output_path)
    band_value = {}
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        band_value[idx] = int(f[55, 48].mean())
        idx += 1
    cap.release()
    assert idx == N, f"expected {N} output frames, read {idx}"

    # inactive frames keep the original band (200); active frames blend to black
    for i in (0, 9, 30, 45, 59):
        assert band_value[i] > 100, f"inactive frame {i} should be unchanged, got {band_value[i]}"
    # active frames incl. the segment boundary at 15 and the active tail at 29
    for i in (10, 15, 24, 29):
        assert band_value[i] < 100, f"active frame {i} should be blended black, got {band_value[i]}"


def test_file_based_backend_rejects_temporal_plan(tmp_path):
    """A temporal WipePlan cannot be flattened to a static PNG for file backends."""
    from types import SimpleNamespace

    from videowipe.plan import build_wipe_plan, compute_source

    video = tmp_path / "input.mp4"
    _write_test_video(video, width=96, height=64, frames=30)
    src = compute_source(str(video))
    H, W = src.height, src.width
    band = np.zeros((H, W), dtype=np.uint8)
    band[50:60, 10:86] = 1
    cand = SimpleNamespace(
        id="c1", type="subtitle", label="sub", bbox=(10, 50, 86, 60),
        confidence=0.9, default_remove=True, mask=band, presence_frames=[0, 10],
    )
    plan = build_wipe_plan(
        [cand], sample_indices=[0, 10, 20], n_valid=3, source=src, frame_shape=(H, W),
    )
    from videowipe.plan import is_temporal
    assert is_temporal(plan)

    engine = WipeEngine(task="clean", external_command="echo")
    try:
        with pytest.raises(videowipe.InvalidInputError, match="temporal WipePlan"):
            engine.process(
                video=str(video), output=str(tmp_path / "out"), plan=plan,
            )
    finally:
        engine.cleanup()
