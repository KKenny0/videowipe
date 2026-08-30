"""WipeEngine and convenience functions."""
from __future__ import annotations

import importlib
import json
import os
import platform
import threading
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from videowipe.api import (
    CancellationToken,
    ProgressCallback,
    ProgressEvent,
    WipeRequest,
    WipeResult,
)
from videowipe.errors import (
    BackendUnavailableError,
    InvalidInputError,
    ProcessingCancelledError,
    ProcessingError,
    WipeError,
)
from videowipe.inpainters import InpaintJob, get_registry
from videowipe.plan import (
    WipePlan,
    compute_source,
    execution_masks,
    is_temporal,
    load_wipe_plan,
    save_wipe_plan,
    validate_plan,
)
from videowipe.planning import (
    build_recognizer,
    finalize as finalize_clean_plan,
    prepare as prepare_clean_plan,
)
from videowipe.tasks.base import (
    BaseTask,
    read_frame_info,
    read_mask,
    validate_mask_shape,
)
from videowipe.tasks.detext import DetextTask
from videowipe.weights import ensure_onnx_weights, ensure_weight

if TYPE_CHECKING:
    from videowipe.detect import TextDetector

_TASK_CLASSES = {
    "detext": DetextTask,
    "clean": DetextTask,
}

_DEFAULT_WEIGHTS_PTH = {
    "detext": "detext_trial.pth",
    "clean": "detext_trial.pth",
}
_DEFAULT_WEIGHTS_ONNX = {
    "detext": "sttn",
    "clean": "sttn",
}

# Gaussian alpha radius applied while WipePlan projects execution masks. A
# small default removes the hard seam from STTN's blend without
# softening so much that the filled region bleeds into surrounding detail.
# This is a "finished feel" default, not a user-facing knob.
_DEFAULT_FEATHER_RADIUS = 4


@lru_cache(maxsize=None)
def _module_available(name: str) -> bool:
    """Return whether a backend can actually be imported in this process."""
    try:
        importlib.import_module(name)
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class _ProgressCallbackError(Exception):
    """Keep consumer callback failures outside VideoWipe error mapping."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


class WipeEngine:
    """Reusable engine for video inpainting tasks.

    Create one engine, call process() for each video, then cleanup().

    Args:
        task: Task type, currently "detext".
        weight: Path to model weight file (.pth for PyTorch, .onnx for ONNX).
            For STTN, None auto-resolves the default weights. Custom adapters
            receive None in their ``load()`` method when no weight is set.
        device: "auto", "cuda", or "cpu". Only used with .pth weights.
        gap: Frames per inpainting segment. The conservative default of 25
            balances context and performance; larger values provide more
            context but have superlinear compute and memory cost.
        dual: Show original video side-by-side in output.
        detector: TextDetector for auto mask generation. None to use default.
        external_command: External inpainting command string. When set,
            bypasses built-in STTN and calls the command with
            ``<command> <video> <mask> <output_dir>``.
        model: Inpainter name resolved via the registry. Default "sttn".
            Use "propainter" for the ProPainter external model (requires a
            ProPainter source checkout; see ``propainter_dir``).
        propainter_dir: Path to a ProPainter source checkout, used only when
            ``model == "propainter"``. Falls back to the wrapper script's
            ``VIDEOWIPE_PROPINTER_DIR`` environment variable.
            resolution (``VIDEOWIPE_PROPINTER_DIR`` / default) when None.
        detect_mode: Detection preset: "fast", "balanced", or "sensitive".
            Controls sample count, consistency threshold, and subtitle fallback.
            Only used for the "clean" task. Default is "balanced".
        ocr: OCR mode: "auto", "off", or "rapidocr". "auto" silently degrades
            when OCR is not installed. "off" never imports OCR. "rapidocr"
            raises an error if unavailable.
    """

    def __init__(
        self,
        task: str = "detext",
        weight: Optional[str] = None,
        device: str = "auto",
        gap: int = 25,
        dual: bool = False,
        detector: Optional[TextDetector] = None,
        external_command: Optional[str] = None,
        model: str = "sttn",
        propainter_dir: Optional[str] = None,
        detect_mode: str = "balanced",
        ocr: str = "auto",
    ):
        if task not in _TASK_CLASSES:
            raise ValueError(f"Unknown task: {task}. Choose from: {list(_TASK_CLASSES)}")

        self.task = task
        self._weight = weight
        self._device = device
        self._detector = detector
        self._external_command = external_command
        self._model = model
        self._propainter_dir = propainter_dir
        self._detect_mode = detect_mode
        self._ocr = ocr
        self._task_impl: BaseTask = _TASK_CLASSES[task](gap=gap, dual=dual)
        if task == "clean":
            setattr(self._task_impl, "output_suffix", "clean")
        # Soft alpha seam at mask boundaries: non-zero by default so the STTN
        # blend no longer produces a hard rectangle. This is a "finished feel"
        # default, not a user-facing knob. Set to 0 only for the eval path,
        # which compares against binary ground-truth masks.
        self._task_impl.feather_radius = _DEFAULT_FEATHER_RADIUS
        self._model_loaded = False
        self._active_cancellation: Optional[CancellationToken] = None
        self._active_progress: Optional[ProgressCallback] = None
        self._run_lock = threading.Lock()
        # Warnings from the most recent process() call (e.g. WipePlan warnings),
        # surfaced through WipeResult by run(). Reset at the start of process().
        self._last_warnings: list[str] = []

    def __enter__(self) -> "WipeEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def run(
        self,
        request: WipeRequest,
        on_progress: Optional[ProgressCallback] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> WipeResult:
        """Run one structured SDK request.

        ``process()`` remains the backwards-compatible path-returning API.
        New integrations should use this method for structured progress,
        cancellation, errors, and result metadata.
        """
        if not isinstance(request, WipeRequest):
            raise InvalidInputError("request must be a WipeRequest instance")
        if on_progress is not None and not callable(on_progress):
            raise InvalidInputError("on_progress must be callable")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise InvalidInputError("cancellation must be a CancellationToken")
        if isinstance(request.targets, (str, bytes)):
            raise InvalidInputError("targets must be a sequence of strings, not a string")
        if isinstance(request.regions, (str, bytes)):
            raise InvalidInputError("regions must be a sequence of strings, not a string")
        try:
            video_path = os.fspath(request.video)
            output_dir = os.fspath(request.output_dir)
            mask_path = os.fspath(request.mask) if request.mask is not None else None
            targets = list(request.targets)
            regions = list(request.regions)
        except TypeError as exc:
            raise InvalidInputError(
                "video, mask, and output_dir must be filesystem paths; "
                "targets and regions must be sequences",
                cause=exc,
            ) from exc
        if not video_path or not output_dir:
            raise InvalidInputError("video and output_dir must not be empty")
        if not self._run_lock.acquire(blocking=False):
            raise ProcessingError(
                "This WipeEngine is already processing a request",
                code="ENGINE_BUSY",
                retryable=True,
            )

        token = cancellation or CancellationToken()
        self._active_cancellation = token
        self._active_progress = on_progress
        artifact_before = self._artifact_snapshot(output_dir)

        def legacy_progress(completed: int, total: int) -> None:
            self._emit_progress(ProgressEvent("inpaint", completed, total))

        try:
            self._emit_progress(ProgressEvent("prepare", 0, 1))
            output_path = self.process(
                video=video_path,
                mask=mask_path,
                output=output_dir,
                detector=request.detector,
                targets=targets,
                intent=request.intent,
                agent=request.agent,
                regions=regions,
                preview=request.preview,
                confirm=request.confirm,
                detect_mode=request.detect_mode,
                ocr=request.ocr,
                progress=legacy_progress,
                plan=request.plan,
            )
            result = self._build_result(request, output_path, artifact_before)
            # A final notification cannot retroactively cancel a successful run.
            self._emit_progress(ProgressEvent("complete", 1, 1), check_after=False)
            return result
        except _ProgressCallbackError as exc:
            raise exc.cause
        except ProcessingCancelledError:
            raise
        except WipeError:
            raise
        except ValueError as exc:
            raise InvalidInputError(str(exc), cause=exc) from exc
        except Exception as exc:
            from videowipe.external import ExternalModelError

            code = (
                "EXTERNAL_MODEL_ERROR"
                if isinstance(exc, ExternalModelError)
                else "PROCESSING_FAILED"
            )
            message = (
                "External inpainting backend failed"
                if isinstance(exc, ExternalModelError)
                else str(exc)
            )
            raise ProcessingError(message, code=code, cause=exc) from exc
        finally:
            self._active_cancellation = None
            self._active_progress = None
            self._run_lock.release()

    @staticmethod
    def _artifact_snapshot(output_dir: str) -> dict:
        snapshot = {}
        for name in (
            "auto_mask.png",
            "clean_candidates.json",
            "clean_preview.jpg",
            "benchmark.json",
            "wipe_plan.json",
            "wipe_plan_masks.npz",
        ):
            path = os.path.join(output_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            snapshot[name] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _build_result(
        self,
        request: WipeRequest,
        output_path: str,
        artifact_before: dict,
    ) -> WipeResult:
        output_dir = os.fspath(request.output_dir)
        benchmark_path = os.path.join(output_dir, "benchmark.json")
        benchmark = {}
        artifact_after = self._artifact_snapshot(output_dir)
        benchmark_changed = artifact_after.get("benchmark.json") != artifact_before.get(
            "benchmark.json"
        )
        if not request.preview and benchmark_changed:
            try:
                with open(benchmark_path, "r", encoding="utf-8") as handle:
                    benchmark = json.load(handle)
            except (OSError, ValueError):
                benchmark = {}

        artifacts = []
        for name, signature in artifact_after.items():
            if request.preview and name == "benchmark.json":
                continue
            if signature == artifact_before.get(name):
                continue
            path = os.path.join(output_dir, name)
            artifacts.append(path)
        if os.path.isfile(output_path) and output_path not in artifacts:
            artifacts.append(output_path)

        timings = benchmark.get("timing", {})
        if not isinstance(timings, dict):
            timings = {}
        return WipeResult(
            output_path=output_path,
            backend=benchmark.get("backend"),
            mask_source="manual" if request.mask is not None else "auto",
            artifacts=tuple(artifacts),
            timings=dict(timings),
            warnings=tuple(self._last_warnings),
            preview=request.preview,
        )

    def _check_cancelled(self) -> None:
        if self._active_cancellation is not None:
            self._active_cancellation.raise_if_cancelled()

    def _emit_progress(
        self, event: ProgressEvent, *, check_after: bool = True,
    ) -> None:
        self._check_cancelled()
        if self._active_progress is not None:
            try:
                self._active_progress(event)
            except Exception as exc:
                raise _ProgressCallbackError(exc) from exc
        if check_after:
            self._check_cancelled()

    def _ensure_model(self):
        if self._model_loaded:
            return
        weight_path = self._weight
        if self._model == "sttn":
            if weight_path is None:
                torch_available = _module_available("torch") and _module_available(
                    "torchvision"
                )
                onnx_available = _module_available("onnxruntime")
                # ONNX Runtime's CPU provider is slower than Torch on Apple
                # Silicon; other platforms retain the lightweight ONNX default.
                if _is_apple_silicon() and torch_available:
                    weight_path = ensure_weight(_DEFAULT_WEIGHTS_PTH[self.task])
                elif onnx_available:
                    base = ensure_onnx_weights(_DEFAULT_WEIGHTS_ONNX[self.task])
                    weight_path = base + ".onnx"
                elif torch_available:
                    weight_path = ensure_weight(_DEFAULT_WEIGHTS_PTH[self.task])
                else:
                    raise BackendUnavailableError(
                        "No inference backend found. Install one of:\n"
                        "  pip install videowipe[onnx]   (lightweight, ~200MB)\n"
                        "  pip install videowipe[torch]  (full PyTorch, ~2.5GB)"
                    )
            elif weight_path.endswith(".onnx") and not _module_available("onnxruntime"):
                raise BackendUnavailableError(
                    "ONNX weights require: pip install videowipe[onnx]"
                )
            elif weight_path.endswith((".pth", ".pt")) and not (
                _module_available("torch") and _module_available("torchvision")
            ):
                raise BackendUnavailableError(
                    "PyTorch weights require: pip install videowipe[torch]"
                )
        inpainter = get_registry().create(self._model)
        try:
            inpainter.load(weight_path, device=self._device)
        except ImportError as exc:
            raise BackendUnavailableError(
                f"The {self._model} backend could not load its runtime dependency",
                cause=exc,
            ) from exc
        self._task_impl.inpainter = inpainter
        self._task_impl.backend = getattr(inpainter, "backend", inpainter)
        self._model_loaded = True

    def _resolve_file_inpainter(self):
        """Return a file-based Inpainter for this run, or None for frame-based.

        File-based inpainters (external command, ProPainter) consume a video
        file and a mask file path rather than an in-memory frame stream, and
        share the mask-path preparation + benchmark path in process().
        """
        if self._external_command:
            return get_registry().create(
                "external", command=self._external_command
            )
        if self._model == "propainter":
            return get_registry().create(
                "propainter", propainter_dir=self._propainter_dir
            )
        return None

    def process(self, video: str, mask: str | None = None,
                output: str = "result/",
                detector: Optional[TextDetector] = None,
                targets: list[str] | None = None,
                intent: str | None = None,
                agent: str | None = None,
                regions: list[str] | None = None,
                preview: bool = False,
                confirm: bool = False,
                detect_mode: str | None = None,
                ocr: str | None = None,
                progress=None,
                plan=None) -> str:
        """Process a single video. Returns the output file path.

        Args:
            video: Path to input video.
            mask: Path to mask image. ``None`` to auto-detect subtitle regions.
            output: Output directory.
            detector: Override the text detector for auto-mask generation.
            plan: A :class:`WipePlan` or path to ``wipe_plan.json``. Mutually
                exclusive with *mask*; only supported for the ``clean`` task.
        """
        self._check_cancelled()
        video = os.fspath(video)
        output = os.fspath(output)
        if mask is not None:
            mask = os.fspath(mask)
        if mask is not None and plan is not None:
            raise InvalidInputError("mask and plan are mutually exclusive")
        if plan is not None and self.task != "clean":
            raise InvalidInputError("plan is only supported for the clean task")
        self._last_warnings = []
        self._active_plan: Optional[WipePlan] = None
        frame_mask = None
        os.makedirs(output, exist_ok=True)
        bm: dict = {"video_path": video, "timing": {}}
        bm["mask_source"] = "manual" if mask is not None else "auto"
        t_total_start = time.monotonic()

        if mask is not None:
            mask_arr = read_mask(mask)
        else:
            det = detector or self._detector
            t_detect_start = time.monotonic()
            if self.task == "clean":
                wipe_plan = self._resolve_clean_plan(
                    plan, video, det, targets, intent, agent, regions,
                    detect_mode, ocr, output, confirm,
                )
                self._last_warnings = list(wipe_plan.warnings)
                self._active_plan = wipe_plan
                mask_arr, frame_mask = execution_masks(
                    wipe_plan, self._task_impl.feather_radius
                )
                auto_mask_path = os.path.join(output, "auto_mask.png")
                if not cv2.imwrite(
                    auto_mask_path, (mask_arr * 255).astype(np.uint8),
                ):
                    raise OSError(f"Failed to write image: {auto_mask_path}")
                if plan is None:
                    self._emit_progress(ProgressEvent("persist", 1, 1))
                if preview:
                    self._check_cancelled()
                    print(f"Preview saved to {output}")
                    return output
            else:
                from videowipe.detect import detect_subtitle_mask
                mask_arr = detect_subtitle_mask(video, detector=det)
                # Save auto-detected mask for inspection
                cv2.imwrite(
                    os.path.join(output, "auto_mask.png"),
                    (mask_arr * 255).astype(np.uint8),
                )
            bm["timing"]["detection_s"] = round(time.monotonic() - t_detect_start, 3)

        self._check_cancelled()

        if preview:
            self._check_cancelled()
            print(f"Preview saved to {output}")
            return output

        # A fresh plan must be executable before any backend loads: it needs at
        # least one remove track and a precise mask on each (an all-keep plan
        # from `confirm none`, or a metadata-only plan, stops here without
        # loading the model).
        if self._active_plan is not None:
            validate_plan(self._active_plan, require_remove=True)

        # Resolve mask file path for external command or normal pipeline
        if mask is not None:
            mask_path_saved = mask
        else:
            mask_path_saved = os.path.join(output, "auto_mask.png")

        file_inpainter = self._resolve_file_inpainter()
        if file_inpainter is not None:
            self._check_cancelled()
            if self._active_plan is not None and is_temporal(self._active_plan):
                raise InvalidInputError(
                    "temporal WipePlan (per-frame segments) is not supported by "
                    "file-based backends; a static mask cannot honor time ranges"
                )
            bm["model_type"] = file_inpainter.name
            if self._external_command:
                bm["external_command"] = self._external_command
            t_ext_start = time.monotonic()
            ext_job = InpaintJob(
                video_path=video,
                mask=mask_arr,
                mask_path=mask_path_saved,
                output_dir=output,
                fps=0.0,
                frame_count=0,
                width=0,
                height=0,
                progress=progress,
            )
            out_path = file_inpainter.inpaint(ext_job).output_path
            self._check_cancelled()
            bm["timing"]["external_s"] = round(time.monotonic() - t_ext_start, 3)
            bm["backend"] = file_inpainter.name
            bm["output_path"] = out_path
            bm["error"] = None
            bm["timing"]["total_s"] = round(time.monotonic() - t_total_start, 3)
            try:
                with open(os.path.join(output, "benchmark.json"), "w", encoding="utf-8") as fh:
                    json.dump(bm, fh, indent=2)
            except OSError:
                pass
            return out_path

        t_model_start = time.monotonic()
        self._check_cancelled()
        self._ensure_model()
        self._check_cancelled()
        bm["timing"]["model_load_s"] = round(time.monotonic() - t_model_start, 3)
        bm["backend"] = type(self._task_impl.backend).__name__

        reader = None
        out_path = ""
        bm_error = None
        try:
            reader, frame_info = read_frame_info(video)
            bm.update({
                "width": frame_info["W_ori"],
                "height": frame_info["H_ori"],
                "frame_count": frame_info["len"],
                "fps": round(frame_info["fps"], 2),
            })
            validate_mask_shape(mask_arr, frame_info)
            mask_pixels = float(np.sum(mask_arr > 0))
            total_pixels = frame_info["H_ori"] * frame_info["W_ori"]
            bm["mask_area_ratio"] = round(mask_pixels / total_pixels, 6) if total_pixels else 0.0
            self._task_impl._bm = bm
            self._task_impl.mask_path = mask_path_saved
            process_kwargs = {"video_path": video}
            if progress is not None:
                process_kwargs["progress"] = progress
            if frame_mask is not None:
                process_kwargs["frame_mask"] = frame_mask
            out_path = self._task_impl.process_video(
                reader, frame_info, mask_arr, output, **process_kwargs
            )
            bm["backend"] = getattr(
                self._task_impl, "backend_label", bm["backend"]
            )
            self._check_cancelled()
        except Exception as exc:
            bm_error = str(exc)
            raise
        finally:
            if reader is not None:
                reader.release()
            bm["timing"]["total_s"] = round(time.monotonic() - t_total_start, 3)
            bm["output_path"] = out_path or None
            bm["error"] = bm_error
            try:
                with open(os.path.join(output, "benchmark.json"), "w", encoding="utf-8") as fh:
                    json.dump(bm, fh, indent=2)
            except OSError:
                pass
        return out_path

    def cleanup(self):
        """Release model and GPU memory."""
        if not self._run_lock.acquire(blocking=False):
            raise ProcessingError(
                "Cannot clean up a WipeEngine while a request is running",
                code="ENGINE_BUSY",
                retryable=True,
            )
        try:
            if not self._external_command:
                self._task_impl.cleanup()
            self._model_loaded = False
        finally:
            self._run_lock.release()

    @staticmethod
    def _build_recognizer(ocr_mode: str):
        """Compatibility hook; Clean Planning owns OCR construction."""
        return build_recognizer(ocr_mode)

    @staticmethod
    def _confirm_candidates(candidates, selected):
        """Ask the user which candidates should be removed."""
        selected_ids = {candidate.id for candidate in selected}
        print("Detected clean targets:")
        for candidate in candidates:
            marker = "*" if candidate.id in selected_ids else " "
            print(
                f"{marker} {candidate.id}: {candidate.label} "
                f"({candidate.reason}, confidence {candidate.confidence:.2f})"
            )
        answer = input(
            "Remove candidate ids separated by commas, press Enter to accept, "
            "or type 'none': "
        ).strip()
        if not answer:
            return list(selected)
        if answer.lower() in {"none", "cancel", "no"}:
            return []
        wanted = {item.strip() for item in answer.split(",") if item.strip()}
        return [candidate for candidate in candidates if candidate.id in wanted]

    @staticmethod
    def _select_candidates_with_agent(agent, candidates, intent):
        """Compatibility hook for custom agent-selection adapters."""
        from videowipe.agent import select_with_agent

        selected_ids = select_with_agent(agent, candidates, intent)
        if selected_ids is None:
            return None
        id_set = set(selected_ids)
        return [candidate for candidate in candidates if candidate.id in id_set]

    # ── WipePlan: detection, construction, execution glue ────────────────────

    def plan(
        self,
        request: WipeRequest,
        on_progress: Optional[ProgressCallback] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> WipePlan:
        """Detect, build, and persist a :class:`WipePlan` without loading the model.

        Writes ``wipe_plan.json`` + ``wipe_plan_masks.npz`` (and the overview
        ``auto_mask.png``) into ``request.output_dir``. The plan encodes the
        resolved remove/keep action per track, including the top-overlay safety
        default. Re-running with this plan (via ``WipeRequest(plan=...)``)
        reproduces the same mask deterministically.
        """
        if not isinstance(request, WipeRequest):
            raise InvalidInputError("request must be a WipeRequest instance")
        if on_progress is not None and not callable(on_progress):
            raise InvalidInputError("on_progress must be callable")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):
            raise InvalidInputError("cancellation must be a CancellationToken")
        if isinstance(request.targets, (str, bytes)):
            raise InvalidInputError("targets must be a sequence of strings, not a string")
        if isinstance(request.regions, (str, bytes)):
            raise InvalidInputError("regions must be a sequence of strings, not a string")
        if request.mask is not None:
            raise InvalidInputError("mask and plan are mutually exclusive")
        if self.task != "clean":
            raise InvalidInputError("plan is only supported for the clean task")
        try:
            video_path = os.fspath(request.video)
            output_dir = os.fspath(request.output_dir)
            targets = list(request.targets)
            regions = list(request.regions)
        except TypeError as exc:
            raise InvalidInputError(
                "video and output_dir must be filesystem paths; "
                "targets and regions must be sequences",
                cause=exc,
            ) from exc
        if not video_path or not output_dir:
            raise InvalidInputError("video and output_dir must not be empty")
        if not self._run_lock.acquire(blocking=False):
            raise ProcessingError(
                "This WipeEngine is already processing a request",
                code="ENGINE_BUSY",
                retryable=True,
            )

        self._active_cancellation = cancellation or CancellationToken()
        self._active_progress = on_progress
        try:
            self._emit_progress(ProgressEvent("prepare", 0, 1))
            os.makedirs(output_dir, exist_ok=True)
            draft = self._prepare_clean_draft(
                video_path, request.detector or self._detector, targets,
                request.intent, request.agent, regions, request.detect_mode,
                request.ocr, output_dir, request.confirm,
            )
            wipe_plan = self._finalize_clean_draft(draft)
            self._emit_progress(ProgressEvent("persist", 0, 1))
            self._write_clean_artifacts(
                draft, {track.id for track in wipe_plan.remove_tracks}, output_dir,
            )
            save_wipe_plan(wipe_plan, output_dir)
            mask_arr, _ = execution_masks(
                wipe_plan, self._task_impl.feather_radius
            )
            auto_mask_path = os.path.join(output_dir, "auto_mask.png")
            if not cv2.imwrite(
                auto_mask_path, (mask_arr * 255).astype(np.uint8),
            ):
                raise OSError(f"Failed to write image: {auto_mask_path}")
            self._emit_progress(ProgressEvent("persist", 1, 1))
            self._emit_progress(ProgressEvent("complete", 1, 1))
            return wipe_plan
        except _ProgressCallbackError as exc:
            raise exc.cause
        finally:
            self._active_cancellation = None
            self._active_progress = None
            self._run_lock.release()

    def _resolve_clean_plan(
        self, plan, video, detector, targets, intent, agent, regions,
        detect_mode, ocr, output, confirm,
    ) -> WipePlan:
        """Return an explicit plan or adapt Clean Planning for a fresh run."""
        if plan is not None:
            return self._resolve_plan_argument(plan, video)
        draft = self._prepare_clean_draft(
            video, detector, targets, intent, agent, regions,
            detect_mode, ocr, output, confirm,
        )
        wipe_plan = self._finalize_clean_draft(draft)
        self._emit_progress(ProgressEvent("persist", 0, 1))
        self._write_clean_artifacts(
            draft, {track.id for track in wipe_plan.remove_tracks}, output,
        )
        save_wipe_plan(wipe_plan, output)
        return wipe_plan

    def _prepare_clean_draft(
        self, video, detector, targets, intent, agent, regions,
        detect_mode, ocr, output, confirm,
    ):
        """Adapt progress, confirmation, and preview artifacts around prepare()."""
        effective_mode = detect_mode or self._detect_mode
        effective_ocr = ocr or self._ocr
        self._emit_progress(ProgressEvent("detect", 0, 0))
        draft = prepare_clean_plan(
            video,
            detector=detector,
            targets=targets or (),
            intent=intent,
            agent=agent,
            regions=regions or (),
            detect_mode=effective_mode,
            ocr=effective_ocr,
            recognizer_builder=self._build_recognizer,
            agent_selector=self._select_candidates_with_agent,
        )
        self._write_clean_artifacts(draft, draft.proposed_remove_ids, output)
        if confirm:
            selected = self._confirm_candidates(
                draft.candidates,
                [
                    candidate for candidate in draft.candidates
                    if candidate.id in draft.proposed_remove_ids
                ],
            )
            draft = draft.with_remove_ids(candidate.id for candidate in selected)
            self._write_clean_artifacts(draft, draft.proposed_remove_ids, output)
        self._emit_progress(ProgressEvent("detect", 1, 1))
        return draft

    def _finalize_clean_draft(self, draft) -> WipePlan:
        return finalize_clean_plan(
            draft,
            progress=lambda done, total: self._emit_progress(
                ProgressEvent("refine", done, total)
            ),
            check_cancelled=self._check_cancelled,
        )

    @staticmethod
    def _write_clean_artifacts(draft, selected_ids, output):
        from videowipe.detect import write_clean_artifacts

        candidates = list(draft.candidates)
        write_clean_artifacts(
            SimpleNamespace(
                candidates=candidates,
                preview_frame=draft.preview_frame,
            ),
            [candidate for candidate in candidates if candidate.id in selected_ids],
            output,
        )

    @staticmethod
    def _resolve_plan_argument(plan, video) -> WipePlan:
        """Accept a WipePlan object or a path to wipe_plan.json; bind to video."""
        if isinstance(plan, WipePlan):
            validate_plan(plan, require_remove=True)
            actual = compute_source(video)
            if actual.sha256 != plan.source.sha256:
                raise InvalidInputError(
                    "plan source sha256 does not match the video"
                )
            if (actual.width, actual.height, actual.frame_count) != (
                plan.source.width, plan.source.height, plan.source.frame_count,
            ):
                raise InvalidInputError(
                    "plan source width/height/frame_count do not match the video"
                )
            return plan
        return load_wipe_plan(os.fspath(plan), video_path=video)

def remove_text(
    video: str,
    mask: str | None = None,
    output: str = "result/",
    weight: str | None = None,
    device: str = "auto",
    gap: int = 25,
    dual: bool = False,
    detector: Optional[TextDetector] = None,
) -> str:
    """Remove hardcoded subtitles from a video. Convenience wrapper.

    Backend is auto-detected from the weight file:
      .pth → PyTorch (requires ``pip install videowipe[torch]``)
      .onnx → ONNX Runtime (requires ``pip install videowipe[onnx]``)

    If *mask* is ``None``, subtitle regions are auto-detected.
    """
    engine = WipeEngine(
        task="detext", weight=weight, device=device, gap=gap, dual=dual,
        detector=detector,
    )
    try:
        return engine.process(video=video, mask=mask, output=output)
    finally:
        engine.cleanup()
