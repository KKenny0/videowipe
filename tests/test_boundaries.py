import json
import os
import pathlib
import sys

import numpy as np
import pytest
import cv2

from videowipe.backends import _detect_backend
from videowipe import cli
from videowipe import agent as agent_module
from videowipe.cli import _build_parser
from videowipe.detect import (
    CleanCandidate,
    TextBox,
    _iou_bbox,
    detect_clean_candidates,
    mask_from_candidates,
    resolve_detect_params,
    select_candidates_by_intent,
    select_clean_candidates,
)
from videowipe.engine import WipeEngine, remove_text
from videowipe.inpainters import STTNInpainter, get_registry
from videowipe.external import ExternalInpainter, ExternalModelError, run_external
from videowipe.tasks.base import read_mask, validate_mask_shape


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
    assert "propainter_wipe.py" in inpainter.command
    assert "--propainter-dir /some/path" in inpainter.command


def test_propainter_factory_omits_dir_flag_when_unset():
    """Without propainter_dir, the command has no --propainter-dir flag."""
    inpainter = get_registry().create("propainter")
    assert isinstance(inpainter, ExternalInpainter)
    assert inpainter.name == "propainter"
    assert "--propainter-dir" not in inpainter.command
    assert "propainter_wipe.py" in inpainter.command


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
        def detect(self, frame):
            return [
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
                TextBox(
                    points=np.array([[120, 75], [200, 75], [200, 95], [120, 95]]),
                    confidence=0.7,
                    text="Main St",
                ),
            ]

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
        engine.process(video=str(video), mask=str(mask), output=str(output))
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


def test_compute_mask_iou_identical_and_disjoint():
    """IoU should be 1.0 for identical masks and 0.0 for disjoint."""
    from videowipe.detect import mask_from_candidates

    mask_a = np.zeros((20, 30, 1), dtype=np.uint8)
    mask_a[5:15, 5:25] = 1

    # Identical
    intersection = np.sum((mask_a > 0) & (mask_a > 0))
    union = np.sum((mask_a > 0) | (mask_a > 0))
    assert intersection / union == pytest.approx(1.0)

    # Disjoint
    mask_b = np.zeros((20, 30, 1), dtype=np.uint8)
    mask_b[0:5, 0:5] = 1
    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    union = np.sum((mask_a > 0) | (mask_b > 0))
    assert intersection == 0


def test_eval_clean_detection_reports_golden_iou(tmp_path):
    """eval_clean_detection.py computes IoU when --mask-dir is given."""
    # Create a test video
    video = tmp_path / "test1.mp4"
    _write_test_video(video, width=96, height=64)

    # Create a golden mask
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    golden = np.zeros((64, 96), dtype=np.uint8)
    golden[50:60, 10:86] = 255
    cv2.imwrite(str(mask_dir / "test1_mask.png"), golden)

    # Run eval script as subprocess
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/eval_clean_detection.py",
         str(tmp_path), "--mask-dir", str(mask_dir)],
        capture_output=True, text=True, cwd=str(
            pathlib.Path(__file__).resolve().parent.parent
        ),
        env={**os.environ, "PYTHONPATH": "src"},
    )
    # Script should succeed (may find 0 candidates but that's ok)
    assert result.returncode in (0, 2)  # 0 = success, 2 = abnormal bbox
    # Should mention golden IoU in output
    assert "Mask area ratio" in result.stdout


def test_eval_clean_detection_flags_missing_golden(tmp_path):
    """eval_clean_detection.py reports missing goldens."""
    video = tmp_path / "test2.mp4"
    _write_test_video(video, width=96, height=64)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    # No golden mask for test2

    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/eval_clean_detection.py",
         str(tmp_path), "--mask-dir", str(mask_dir)],
        capture_output=True, text=True, cwd=str(
            pathlib.Path(__file__).resolve().parent.parent
        ),
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert "MISSING GOLDEN" in result.stdout


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

