import json
from pathlib import Path

import pytest

from videowipe import (
    BackendUnavailableError,
    CancellationToken,
    InvalidInputError,
    ProcessingCancelledError,
    ProcessingError,
    ProgressEvent,
    WipeEngine,
    WipeRequest,
    WipeResult,
)


def _fake_success(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clean.mp4"
    output_path.write_bytes(b"video")
    (output_dir / "auto_mask.png").write_bytes(b"mask")
    (output_dir / "benchmark.json").write_text(
        json.dumps({
            "backend": "FakeBackend",
            "mask_source": "auto",
            "timing": {"total_s": 1.25},
        }),
        encoding="utf-8",
    )
    return str(output_path)


def test_run_returns_structured_result_and_progress(tmp_path, monkeypatch):
    engine = WipeEngine()
    output_dir = tmp_path / "result"
    calls = []

    def fake_process(**kwargs):
        calls.append(kwargs)
        kwargs["progress"](2, 4)
        kwargs["progress"](4, 4)
        return _fake_success(output_dir)

    monkeypatch.setattr(engine, "process", fake_process)
    events = []
    result = engine.run(
        WipeRequest(video=tmp_path / "input.mp4", output_dir=output_dir),
        on_progress=events.append,
    )

    assert isinstance(result, WipeResult)
    assert result.backend == "FakeBackend"
    assert result.mask_source == "auto"
    assert result.timings == {"total_s": 1.25}
    assert result.output_path.endswith("clean.mp4")
    assert {Path(path).name for path in result.artifacts} == {
        "auto_mask.png", "benchmark.json", "clean.mp4",
    }
    assert [event.phase for event in events] == [
        "prepare", "inpaint", "inpaint", "complete",
    ]
    assert events[1].fraction == 0.5
    assert calls[0]["video"] == str(tmp_path / "input.mp4")
    assert result.to_dict()["artifacts"] == list(result.artifacts)


def test_run_preserves_manual_mask_and_preview_contract(tmp_path, monkeypatch):
    engine = WipeEngine()
    output_dir = tmp_path / "preview"
    mask = tmp_path / "mask.png"

    def fake_process(**kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "clean_preview.jpg").write_bytes(b"preview")
        (output_dir / "benchmark.json").write_text(
            json.dumps({"backend": "stale"}), encoding="utf-8"
        )
        return str(output_dir)

    monkeypatch.setattr(engine, "process", fake_process)
    result = engine.run(WipeRequest(
        video=tmp_path / "input.mp4",
        mask=mask,
        output_dir=output_dir,
        preview=True,
    ))

    assert result.preview is True
    assert result.mask_source == "manual"
    assert result.backend is None
    assert [Path(path).name for path in result.artifacts] == ["clean_preview.jpg"]


def test_run_rejects_non_request_and_maps_input_errors(monkeypatch):
    engine = WipeEngine()
    with pytest.raises(InvalidInputError, match="WipeRequest") as wrong_type:
        engine.run({"video": "input.mp4"})
    assert wrong_type.value.code == "INVALID_INPUT"

    def fail_process(**kwargs):
        raise ValueError("bad mask shape")

    monkeypatch.setattr(engine, "process", fail_process)
    with pytest.raises(InvalidInputError, match="bad mask shape") as invalid:
        engine.run(WipeRequest(video="input.mp4"))
    assert isinstance(invalid.value.cause, ValueError)


def test_run_validates_callback_paths_and_sequence_shapes():
    engine = WipeEngine()
    with pytest.raises(InvalidInputError, match="on_progress"):
        engine.run(WipeRequest(video="input.mp4"), on_progress="print")
    with pytest.raises(InvalidInputError, match="targets"):
        engine.run(WipeRequest(video="input.mp4", targets="subtitle"))
    with pytest.raises(InvalidInputError, match="filesystem paths"):
        engine.run(WipeRequest(video=object()))


def test_progress_callback_error_is_not_misclassified(monkeypatch):
    engine = WipeEngine()
    callback_error = ValueError("consumer queue closed")

    def fail_callback(event):
        raise callback_error

    with pytest.raises(ValueError) as raised:
        engine.run(WipeRequest(video="input.mp4"), on_progress=fail_callback)
    assert raised.value is callback_error

    # Callback failure must release the engine admission lock.
    monkeypatch.setattr(engine, "process", lambda **kwargs: "missing-output.mp4")
    result = engine.run(WipeRequest(video="input.mp4"))
    assert result.output_path == "missing-output.mp4"


def test_run_maps_processing_errors(monkeypatch):
    engine = WipeEngine()

    def fail_process(**kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(engine, "process", fail_process)
    with pytest.raises(ProcessingError, match="model crashed") as failed:
        engine.run(WipeRequest(video="input.mp4"))
    assert failed.value.code == "PROCESSING_FAILED"
    assert isinstance(failed.value.cause, RuntimeError)


def test_external_error_message_does_not_expose_stderr(monkeypatch):
    from videowipe.external import ExternalModelError

    engine = WipeEngine()
    secret = "signed-url-secret"
    monkeypatch.setattr(
        engine,
        "process",
        lambda **kwargs: (_ for _ in ()).throw(ExternalModelError(secret)),
    )

    with pytest.raises(ProcessingError) as failed:
        engine.run(WipeRequest(video="input.mp4"))
    assert failed.value.code == "EXTERNAL_MODEL_ERROR"
    assert secret not in str(failed.value)
    assert secret in str(failed.value.cause)


def test_result_excludes_unchanged_artifacts_from_previous_run(tmp_path, monkeypatch):
    engine = WipeEngine(task="clean")
    output_dir = tmp_path / "shared"
    output_dir.mkdir()
    for name in ("auto_mask.png", "clean_candidates.json", "clean_preview.jpg"):
        (output_dir / name).write_bytes(b"previous-request")
    (output_dir / "benchmark.json").write_text(
        json.dumps({"backend": "PreviousBackend"}), encoding="utf-8"
    )

    output_path = output_dir / "current.mp4"

    def fake_process(**kwargs):
        output_path.write_bytes(b"current-request")
        return str(output_path)

    monkeypatch.setattr(engine, "process", fake_process)
    result = engine.run(WipeRequest(video="input.mp4", output_dir=output_dir))

    assert [Path(path).name for path in result.artifacts] == ["current.mp4"]
    assert result.backend is None


def test_cancellation_is_cooperative_and_does_not_poison_engine(tmp_path, monkeypatch):
    engine = WipeEngine()
    output_dir = tmp_path / "result"
    token = CancellationToken()

    def fake_process(**kwargs):
        kwargs["progress"](1, 2)
        return _fake_success(output_dir)

    monkeypatch.setattr(engine, "process", fake_process)

    def cancel_on_inpaint(event):
        if event.phase == "inpaint":
            token.cancel()

    with pytest.raises(ProcessingCancelledError) as cancelled:
        engine.run(
            WipeRequest(video="input.mp4", output_dir=output_dir),
            on_progress=cancel_on_inpaint,
            cancellation=token,
        )
    assert cancelled.value.code == "PROCESSING_CANCELLED"

    # The per-run token is cleared in finally; the same engine remains usable.
    result = engine.run(WipeRequest(video="input.mp4", output_dir=output_dir))
    assert result.output_path.endswith("clean.mp4")


def test_cancellation_during_complete_notification_keeps_success(tmp_path, monkeypatch):
    engine = WipeEngine()
    output_dir = tmp_path / "result"
    token = CancellationToken()
    monkeypatch.setattr(
        engine, "process", lambda **kwargs: _fake_success(output_dir)
    )

    def cancel_on_complete(event):
        if event.phase == "complete":
            token.cancel()

    result = engine.run(
        WipeRequest(video="input.mp4", output_dir=output_dir),
        on_progress=cancel_on_complete,
        cancellation=token,
    )
    assert result.output_path.endswith("clean.mp4")


def test_engine_busy_guard_and_cleanup_are_atomic():
    engine = WipeEngine()
    assert engine._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(ProcessingError) as busy:
            engine.run(WipeRequest(video="input.mp4"))
        assert busy.value.code == "ENGINE_BUSY"
        with pytest.raises(ProcessingError) as cleanup_busy:
            engine.cleanup()
        assert cleanup_busy.value.code == "ENGINE_BUSY"
    finally:
        engine._run_lock.release()


def test_pre_cancelled_request_never_enters_process(monkeypatch):
    engine = WipeEngine()
    token = CancellationToken()
    token.cancel()
    entered = []
    monkeypatch.setattr(engine, "process", lambda **kwargs: entered.append(True))

    with pytest.raises(ProcessingCancelledError):
        engine.run(WipeRequest(video="input.mp4"), cancellation=token)
    assert entered == []


def test_engine_context_manager_cleans_up(monkeypatch):
    engine = WipeEngine()
    calls = []
    monkeypatch.setattr(engine, "cleanup", lambda: calls.append("cleanup"))

    with engine as entered:
        assert entered is engine

    assert calls == ["cleanup"]


def test_missing_backend_fails_before_weight_download(monkeypatch):
    import videowipe.engine as engine_module

    monkeypatch.setattr(engine_module, "_module_available", lambda name: False)
    monkeypatch.setattr(
        engine_module,
        "ensure_weight",
        lambda *args: pytest.fail("missing backend must not download weights"),
    )
    monkeypatch.setattr(
        engine_module,
        "ensure_onnx_weights",
        lambda *args: pytest.fail("missing backend must not download weights"),
    )

    with pytest.raises(BackendUnavailableError) as failed:
        WipeEngine()._ensure_model()
    assert failed.value.code == "BACKEND_UNAVAILABLE"


def test_backend_probe_requires_successful_import(monkeypatch):
    import videowipe.engine as engine_module

    engine_module._module_available.cache_clear()
    monkeypatch.setattr(
        engine_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(OSError("native library missing")),
    )

    assert engine_module._module_available("present_but_broken") is False
    engine_module._module_available.cache_clear()


def test_backend_import_failure_maps_to_backend_unavailable(monkeypatch):
    import videowipe.engine as engine_module

    class MissingRuntimeInpainter:
        def load(self, weight_path, device="auto"):
            raise ImportError("missing native runtime")

    monkeypatch.setattr(engine_module, "_module_available", lambda name: True)
    monkeypatch.setattr(
        engine_module, "ensure_onnx_weights", lambda *args: "fake-model"
    )
    monkeypatch.setattr(
        engine_module.get_registry(),
        "create",
        lambda name: MissingRuntimeInpainter(),
    )

    with pytest.raises(BackendUnavailableError) as failed:
        WipeEngine()._ensure_model()
    assert isinstance(failed.value.cause, ImportError)


def test_progress_event_with_unknown_total_has_no_fraction():
    assert ProgressEvent("detect").fraction is None
