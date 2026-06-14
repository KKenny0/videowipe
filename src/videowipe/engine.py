"""WipeEngine and convenience functions."""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from videowipe.tasks.base import (
    BaseTask,
    read_frame_info,
    read_mask,
    validate_mask_shape,
)
from videowipe.tasks.detext import DetextTask
from videowipe.inpainters import InpaintJob, get_registry
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


class WipeEngine:
    """Reusable engine for video inpainting tasks.

    Create one engine, call process() for each video, then cleanup().

    Args:
        task: Task type, currently "detext".
        weight: Path to model weight file (.pth for PyTorch, .onnx for ONNX).
            None to auto-download the default.
        device: "auto", "cuda", or "cpu". Only used with .pth weights.
        gap: Segment length per pass. Higher = better quality, slower.
        dual: Show original video side-by-side in output.
        detector: TextDetector for auto mask generation. None to use default.
        external_command: External inpainting command string. When set,
            bypasses built-in STTN and calls the command with
            ``<command> <video> <mask> <output_dir>``.
        model: Inpainter name resolved via the registry. Default "sttn".
            Use "propainter" for the ProPainter external model (requires a
            ProPainter source checkout; see ``propainter_dir``).
        propainter_dir: Path to a ProPainter source checkout, used only when
            ``model == "propainter"``. Falls back to the wrapper script's own
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
        gap: int = 200,
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
        self._model_loaded = False

    def _ensure_model(self):
        if self._model_loaded:
            return
        weight_path = self._weight
        if weight_path is None:
            # Auto-detect: prefer ONNX if onnxruntime is available, else PyTorch
            try:
                import onnxruntime  # noqa: F401
                base = ensure_onnx_weights(_DEFAULT_WEIGHTS_ONNX[self.task])
                weight_path = base + ".onnx"
            except ImportError:
                try:
                    weight_path = ensure_weight(_DEFAULT_WEIGHTS_PTH[self.task])
                except Exception:
                    raise RuntimeError(
                        "No inference backend found. Install one of:\n"
                        "  pip install videowipe[onnx]   (lightweight, ~200MB)\n"
                        "  pip install videowipe[torch]  (full PyTorch, ~2.5GB)"
                    ) from None
        inpainter = get_registry().create(self._model)
        inpainter.load(weight_path, device=self._device)
        self._task_impl.inpainter = inpainter
        self._task_impl.backend = inpainter.backend
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
                ocr: str | None = None) -> str:
        """Process a single video. Returns the output file path.

        Args:
            video: Path to input video.
            mask: Path to mask image. ``None`` to auto-detect subtitle regions.
            output: Output directory.
            detector: Override the text detector for auto-mask generation.
        """
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
                from videowipe.detect import (
                    detect_clean_candidates,
                    infer_regions_from_text,
                    infer_targets_from_text,
                    mask_from_candidates,
                    normalize_target,
                    resolve_detect_params,
                    select_clean_candidates,
                    write_clean_artifacts,
                )

                target_text = " ".join(targets or [])
                intent_text = " ".join(part for part in [target_text, intent or ""] if part)
                requested_regions = list(regions or [])
                requested_regions.extend(infer_regions_from_text(intent_text))
                requested_regions = list(dict.fromkeys(requested_regions))

                inferred_targets = infer_targets_from_text(target_text)
                intent_targets = infer_targets_from_text(intent or "")
                effective_targets = list(targets or [])
                effective_targets.extend(inferred_targets)
                effective_targets = [
                    target for target in dict.fromkeys(effective_targets)
                    if normalize_target(target) != target or target in inferred_targets
                ]
                normalized_targets = {normalize_target(target) for target in effective_targets}
                if requested_regions:
                    effective_targets.append("region")
                    normalized_targets.add("region")
                mentioned_targets = normalized_targets | set(intent_targets)
                include_logo = "logo" in mentioned_targets
                include_translucent = "watermark" in mentioned_targets
                text_targets = {
                    "subtitle", "timestamp", "watermark",
                    "scene_text", "unknown_text",
                }
                detect_text = (
                    bool(mentioned_targets & text_targets)
                    or (not requested_regions and not mentioned_targets)
                    or bool(intent and not mentioned_targets)
                )

                effective_mode = detect_mode or self._detect_mode
                has_subtitle_target = "subtitle" in normalized_targets
                mode_params = resolve_detect_params(
                    effective_mode, has_subtitle_target=has_subtitle_target,
                )

                effective_ocr = ocr or self._ocr
                recognizer = self._build_recognizer(effective_ocr)

                result = detect_clean_candidates(
                    video,
                    detector=det,
                    regions=requested_regions,
                    detect_text=detect_text,
                    include_logo=include_logo,
                    include_translucent_watermark=include_translucent,
                    sample_count=mode_params["sample_count"],
                    consistency=mode_params["consistency"],
                    subtitle_fallback=mode_params["subtitle_fallback"],
                    recognizer=recognizer,
                )
                selected = select_clean_candidates(
                    result.candidates,
                    targets=effective_targets,
                    intent=intent,
                )
                self._warn_if_timestamp_unresolved(effective_targets, result.candidates)
                if agent and intent:
                    agent_selected = self._select_candidates_with_agent(
                        agent, result.candidates, intent
                    )
                    if agent_selected is not None:
                        selected = agent_selected
                write_clean_artifacts(result, selected, output)
                if confirm:
                    selected = self._confirm_candidates(result.candidates, selected)
                    write_clean_artifacts(result, selected, output)
                mask_arr = mask_from_candidates(selected, result.frame_shape)
                cv2.imwrite(
                    os.path.join(output, "auto_mask.png"),
                    (mask_arr * 255).astype(np.uint8),
                )
                if preview:
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

        if preview:
            print(f"Preview saved to {output}")
            return output

        # Resolve mask file path for external command or normal pipeline
        if mask is not None:
            mask_path_saved = mask
        else:
            mask_path_saved = os.path.join(output, "auto_mask.png")

        file_inpainter = self._resolve_file_inpainter()
        if file_inpainter is not None:
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
            )
            out_path = file_inpainter.inpaint(ext_job).output_path
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
        self._ensure_model()
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
            out_path = self._task_impl.process_video(
                reader, frame_info, mask_arr, output, video_path=video
            )
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
        if not self._external_command:
            self._task_impl.cleanup()
        self._model_loaded = False

    @staticmethod
    def _build_recognizer(ocr_mode: str):
        """Build a text recognizer callable based on the OCR mode setting.

        Returns ``None`` when OCR is disabled or unavailable, or a
        ``callable(image_crop) -> str | None`` when OCR is active.
        """
        if ocr_mode == "off":
            return None
        try:
            from videowipe.ocr import recognize_text, _get_engine
            # Eagerly validate that the OCR backend is usable
            _get_engine()
            return recognize_text
        except Exception:
            if ocr_mode == "rapidocr":
                raise RuntimeError(
                    "OCR mode 'rapidocr' requested but rapidocr-onnxruntime "
                    "is not installed. Install it with: "
                    "pip install videowipe[ocr]"
                ) from None
            # auto mode: silently degrade
            return None

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
        from videowipe.agent import select_with_agent

        selected_ids = select_with_agent(agent, candidates, intent)
        if selected_ids is None:
            print("Agent selection unavailable; using local rules.")
            return None
        id_set = set(selected_ids)
        return [candidate for candidate in candidates if candidate.id in id_set]

    @staticmethod
    def _warn_if_timestamp_unresolved(targets, candidates):
        from videowipe.detect import normalize_target

        requested = {normalize_target(target) for target in (targets or [])}
        if "timestamp" not in requested:
            return
        if any(candidate.type == "timestamp" for candidate in candidates):
            return
        print(
            "No timestamp target was confirmed. Timestamp detection requires "
            "recognized text content; use --region top-left/top-right if the "
            "current detector only finds text boxes."
        )


def remove_text(
    video: str,
    mask: str | None = None,
    output: str = "result/",
    weight: str | None = None,
    device: str = "auto",
    gap: int = 200,
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
