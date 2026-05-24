import numpy as np
import pytest

from videowipe.backends import _detect_backend
from videowipe import cli
from videowipe.cli import _build_parser
from videowipe.engine import WipeEngine, remove_text
from videowipe.tasks.base import read_mask, validate_mask_shape


def test_cli_exposes_only_implemented_detext_command():
    parser = _build_parser()

    assert parser.parse_args(["detext", "-v", "input.mp4"]).command == "detext"
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
