"""Detection regression snapshots and remove/keep fact-baseline evaluation.

The default invocation preserves the historic candidate snapshot workflow.  It
is deliberately named a *regression snapshot*: it is not a quality benchmark.
Pass ``--manifest`` to evaluate fixed, indexed annotations with remove/keep
semantics and to write a structured fact-baseline report plus visual previews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import videowipe
from videowipe.detect import (
    detect_clean_candidates,
    mask_from_candidates,
    resolve_detect_params,
    select_clean_candidates,
)

FACT_BASELINE_REPORT_SCHEMA_VERSION = 1
FACT_BASELINE_MANIFEST_SCHEMA_VERSION = 1


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute region Jaccard for two binary masks.

    The empty/empty case is a perfect match.  Callers that want to exclude
    frames with no remove target do so explicitly rather than changing the
    metric definition.
    """
    a = np.asarray(mask_a).squeeze().astype(bool)
    b = np.asarray(mask_b).squeeze().astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return intersection / union if union else 1.0


def _davis_seg2bmap(mask: np.ndarray) -> np.ndarray:
    """Port DAVIS' ``_seg2bmap`` for same-size binary masks.

    Its half-pixel, origin-directed boundary convention differs from a simple
    morphological erosion.  Keep this implementation aligned with the DAVIS
    reference so reported Boundary F remains comparable to its definition.
    """
    seg = np.asarray(mask).squeeze().astype(bool)
    if seg.ndim != 2:
        raise ValueError(f"DAVIS boundary requires a 2D mask, got {seg.shape}")
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    southeast = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    southeast[:-1, :-1] = seg[1:, 1:]
    boundary = (seg ^ east) | (seg ^ south) | (seg ^ southeast)
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def _davis_disk(radius: int) -> np.ndarray:
    """Return the same flat, Euclidean disk used by ``skimage.disk`` here."""
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)


def compute_boundary_f(
    mask_a: np.ndarray, mask_b: np.ndarray, bound_th: float = 0.008
) -> float:
    """Compute the DAVIS 2017 Boundary F definition for two binary masks."""
    foreground = np.asarray(mask_a).squeeze().astype(bool)
    ground_truth = np.asarray(mask_b).squeeze().astype(bool)
    if foreground.shape != ground_truth.shape:
        raise ValueError(f"Mask shapes differ: {foreground.shape} vs {ground_truth.shape}")
    if foreground.ndim != 2:
        raise ValueError(f"Boundary F requires 2D masks, got {foreground.shape}")
    bound_pix = bound_th if bound_th >= 1 else math.ceil(
        bound_th * np.linalg.norm(foreground.shape)
    )
    radius = int(bound_pix)
    foreground_boundary = _davis_seg2bmap(foreground)
    ground_truth_boundary = _davis_seg2bmap(ground_truth)
    kernel = _davis_disk(radius)
    foreground_dilated = cv2.dilate(foreground_boundary.astype(np.uint8), kernel).astype(bool)
    ground_truth_dilated = cv2.dilate(ground_truth_boundary.astype(np.uint8), kernel).astype(bool)
    ground_truth_match = ground_truth_boundary & foreground_dilated
    foreground_match = foreground_boundary & ground_truth_dilated
    foreground_count = int(foreground_boundary.sum())
    ground_truth_count = int(ground_truth_boundary.sum())
    if foreground_count == 0 and ground_truth_count > 0:
        precision, recall = 1.0, 0.0
    elif foreground_count > 0 and ground_truth_count == 0:
        precision, recall = 0.0, 1.0
    elif foreground_count == 0 and ground_truth_count == 0:
        precision = recall = 1.0
    else:
        precision = float(foreground_match.sum()) / foreground_count
        recall = float(ground_truth_match.sum()) / ground_truth_count
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _sha256_named_files(entries: list[tuple[str, Path]]) -> str:
    """Hash stable logical names and bytes, never caller-specific paths."""
    digest = hashlib.sha256()
    for logical_name, path in sorted(entries, key=lambda entry: entry[0]):
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_command(*args: str) -> list[str]:
    return ["git", "-C", str(_project_root()), *args]


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("GIT_DIR", None)
    environment.pop("GIT_WORK_TREE", None)
    return environment


def _git_provenance() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            _git_command("rev-parse", "HEAD"),
            text=True,
            stderr=subprocess.DEVNULL,
            env=_git_environment(),
        ).strip()
        tracked_status = subprocess.check_output(
            _git_command("status", "--porcelain=v1", "--untracked-files=no"),
            text=True,
            stderr=subprocess.DEVNULL,
            env=_git_environment(),
        )
        full_status = subprocess.check_output(
            _git_command("status", "--porcelain=v1", "--untracked-files=all"),
            text=True,
            stderr=subprocess.DEVNULL,
            env=_git_environment(),
        )
        untracked_paths = [
            line[3:] for line in full_status.splitlines() if line.startswith("?? ")
        ]
        return {
            "git_head": head,
            "tracked_worktree_is_clean": not bool(tracked_status.strip()),
            "untracked_paths": untracked_paths,
            "worktree_status_sha256": hashlib.sha256(
                full_status.encode("utf-8")
            ).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_head": None,
            "tracked_worktree_is_clean": None,
            "untracked_paths": None,
            "worktree_status_sha256": None,
        }


def _require_clean_git_worktree(relevant_paths: list[Path]) -> None:
    provenance = _git_provenance()
    if provenance["tracked_worktree_is_clean"] is not True:
        raise RuntimeError(
            "A formal fact baseline requires a clean tracked git worktree; commit "
            "the evaluator and tracked inputs before rerunning."
        )
    root = _project_root().resolve()
    for path in relevant_paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            raise RuntimeError(
                f"A formal fact-baseline input is outside the project repository: {resolved}"
            ) from None
        try:
            subprocess.check_output(
                _git_command("ls-files", "--error-unmatch", "--", str(relative)),
                text=True,
                stderr=subprocess.DEVNULL,
                env=_git_environment(),
            )
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError(
                f"A formal fact-baseline input is not tracked by git: {relative}"
            ) from None


def _evaluator_source_hash() -> str:
    return _sha256_named_files([
        ("scripts/eval_clean_detection.py", Path(__file__).resolve()),
        ("src/videowipe/detect.py", Path(sys.modules["videowipe.detect"].__file__).resolve()),
        ("src/videowipe/weights.py", Path(videowipe.__file__).resolve().parent / "weights.py"),
    ])


def _build_recognizer(ocr_mode: str):
    """Build an OCR recognizer callable, mirroring engine logic."""
    if ocr_mode == "off":
        return None
    try:
        from videowipe.ocr import _get_engine, recognize_text

        _get_engine()
        return recognize_text
    except Exception:  # noqa: BLE001 - optional OCR auto mode intentionally falls back.
        if ocr_mode == "rapidocr":
            raise RuntimeError(
                "OCR mode 'rapidocr' requested but rapidocr-onnxruntime "
                "is not installed. Install it with: pip install videowipe[ocr]"
            ) from None
        return None


def _validate_supported_opencv() -> None:
    """Keep the evaluator on the same supported path as production detection."""
    try:
        major = int(cv2.__version__.split(".", maxsplit=1)[0])
    except ValueError:
        major = 0
    if major >= 5:
        raise RuntimeError(
            "OpenCV 5 is not currently supported by VideoWipe's production DBNet "
            "path; install opencv-python-headless<5."
        )


def _detect_video(
    video_path: str, detect_mode: str, ocr_mode: str
) -> tuple[Any, list[Any], np.ndarray, str]:
    _validate_supported_opencv()
    mode_params = resolve_detect_params(detect_mode)
    result = detect_clean_candidates(
        video_path,
        sample_count=mode_params["sample_count"],
        consistency=mode_params["consistency"],
        subtitle_fallback=mode_params["subtitle_fallback"],
        recognizer=_build_recognizer(ocr_mode),
    )
    selected = select_clean_candidates(result.candidates)
    return result, selected, np.asarray(
        mask_from_candidates(selected, result.frame_shape)
    ).squeeze() > 0, "dbnet_default"


def _legacy_calibration_metric(
    generated_mask: np.ndarray, video_path: str, mask_dir: str | None
) -> dict[str, Any]:
    """Return the old static-Golden comparison with non-quality wording."""
    metric: dict[str, Any] = {"name": "legacy_calibration_metric", "available": False}
    if not mask_dir:
        return metric
    stem = Path(video_path).stem
    golden_path = Path(mask_dir) / f"{stem}_mask.png"
    if not golden_path.exists():
        metric["missing_golden"] = str(golden_path)
        return metric
    golden = cv2.imread(str(golden_path), cv2.IMREAD_GRAYSCALE)
    if golden is None:
        metric["missing_golden"] = f"cannot read: {golden_path}"
        return metric
    golden_binary = golden > 127
    if golden_binary.shape != generated_mask.shape:
        metric["missing_golden"] = (
            f"shape mismatch: {golden_binary.shape} vs {generated_mask.shape}"
        )
        return metric
    metric.update(
        {
            "available": True,
            "golden_area_ratio": round(float(golden_binary.mean()), 6),
            "iou": round(_compute_mask_iou(generated_mask, golden_binary), 6),
        }
    )
    return metric


def _candidate_report(result: Any, selected: list[Any]) -> list[dict[str, Any]]:
    selected_ids = {candidate.id for candidate in selected}
    height, width = result.frame_shape
    rows = []
    for candidate in result.candidates:
        x1, y1, x2, y2 = candidate.bbox
        rows.append(
            {
                "id": candidate.id,
                "type": candidate.type,
                "bbox": list(candidate.bbox),
                "area_ratio": round(((x2 - x1) * (y2 - y1)) / (width * height), 4),
                "selected": candidate.id in selected_ids,
                "reason": candidate.reason,
                "abnormal_bbox": x2 > width or y2 > height or x1 < 0 or y1 < 0,
                "text_samples": candidate.text_samples[:3],
            }
        )
    return rows


def _eval_video(
    video_path: str,
    mask_dir: str | None = None,
    detect_mode: str = "balanced",
    ocr_mode: str = "off",
) -> dict[str, Any]:
    """Return a candidate regression snapshot for compatibility."""
    result, selected, generated_mask, execution_path = _detect_video(
        video_path, detect_mode, ocr_mode
    )
    legacy = _legacy_calibration_metric(generated_mask, video_path, mask_dir)
    report: dict[str, Any] = {
        "report_kind": "regression_snapshot",
        "video": Path(video_path).name,
        "detect_mode": detect_mode,
        "ocr_mode": ocr_mode,
        "detector_execution_path": execution_path,
        "candidate_count": len(result.candidates),
        "selected_count": len(selected),
        "frame_shape": list(result.frame_shape),
        "empty_detection": not result.candidates,
        "generated_mask_area_ratio": round(float(generated_mask.mean()), 6),
        "candidates": _candidate_report(result, selected),
        "legacy_calibration_metric": legacy,
    }
    if not legacy["available"] and "missing_golden" in legacy:
        report["golden_mask_missing"] = legacy["missing_golden"]
    if report["empty_detection"]:
        report["warning"] = "No candidates detected"
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}\nRegression snapshot: {report['video']}")
    print(f"Shape: {report['frame_shape']}")
    print(f"Mask area ratio: {report['generated_mask_area_ratio']:.4f}")
    legacy = report["legacy_calibration_metric"]
    if legacy["available"]:
        print(f"Legacy calibration IoU: {legacy['iou']:.4f}")
    if report.get("golden_mask_missing"):
        print(f"MISSING LEGACY GOLDEN: {report['golden_mask_missing']}")
    print(f"Candidates: {report['candidate_count']}  Selected: {report['selected_count']}")


def _compare_against_regression_snapshot(
    reports: list[dict[str, Any]], snapshot_path: str
) -> bool:
    if not os.path.exists(snapshot_path):
        print(f"Regression snapshot not found: {snapshot_path}")
        return False
    with open(snapshot_path, encoding="utf-8") as fh:
        snapshot = json.load(fh)
    by_video = {row["video"]: row for row in snapshot}
    current_videos = {report["video"] for report in reports}
    stable = True
    for report in reports:
        previous = by_video.get(report["video"])
        if previous is None:
            print(f"NEW: {report['video']} (not in regression snapshot)")
            stable = False
            continue
        if report["candidate_count"] != previous["candidate_count"]:
            print(f"DRIFT: {report['video']} candidate count changed")
            stable = False
        if report["empty_detection"] != previous["empty_detection"]:
            print(f"DRIFT: {report['video']} empty_detection changed")
            stable = False
        previous_types = {row["type"] for row in previous["candidates"]}
        current_types = {row["type"] for row in report["candidates"]}
        if previous_types != current_types:
            print(f"DRIFT: {report['video']} candidate types changed")
            stable = False
    for video in sorted(set(by_video) - current_videos):
        print(f"REMOVED: {video} (present in regression snapshot but not current run)")
        stable = False
    return stable


def _resolve_within(root: Path, relative_path: str, label: str) -> Path:
    """Resolve a manifest path and reject absolute, traversal, and escaping links."""
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise ValueError(f"{label} must be relative, not absolute: {relative_path}")
    if ".." in supplied.parts:
        raise ValueError(f"{label} must not contain parent traversal: {relative_path}")
    root = root.resolve()
    resolved = (root / supplied).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} escapes its root: {relative_path}") from None
    return resolved


def _load_fact_manifest(manifest_path: str, input_dir: str) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    with path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != FACT_BASELINE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing fact-baseline manifest schema_version")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Manifest must contain a non-empty videos list")
    seen_videos: set[str] = set()
    root = Path(input_dir).resolve()
    for video in videos:
        name = video.get("file")
        if not isinstance(name, str) or not name or name in seen_videos:
            raise ValueError("Every manifest video requires a unique file")
        seen_videos.add(name)
        video_path = _resolve_within(root, name, "Manifest video")
        if not video_path.is_file():
            raise ValueError(f"Manifest video is missing: {name}")
        objects = video.get("objects")
        frames = video.get("frames")
        if not isinstance(objects, list) or not isinstance(frames, list) or not frames:
            raise ValueError(f"Video {name} needs non-empty objects and frames")
        ids: set[int] = set()
        for obj in objects:
            object_id = obj.get("id")
            if not isinstance(object_id, int) or object_id < 1 or object_id in ids:
                raise ValueError(f"Video {name} has duplicate or invalid object ID: {object_id}")
            ids.add(object_id)
            if obj.get("action") not in {"remove", "keep"}:
                raise ValueError(f"Video {name} object {object_id} needs remove or keep action")
            if not isinstance(obj.get("type"), str) or not obj["type"]:
                raise ValueError(f"Video {name} object {object_id} needs a type")
        frame_numbers: set[int] = set()
        for frame in frames:
            index = frame.get("frame")
            mask = frame.get("mask")
            if not isinstance(index, int) or index < 0 or index in frame_numbers:
                raise ValueError(f"Video {name} has duplicate or invalid frame number: {index}")
            frame_numbers.add(index)
            if not isinstance(mask, str):
                raise TypeError(f"Video {name} frame {index} mask must be a string")
            mask_path = _resolve_within(path.parent, mask, "Annotation mask")
            if not mask_path.is_file():
                raise ValueError(f"Video {name} frame {index} has missing mask: {mask}")
    return manifest


def _read_indexed_mask(mask_path: Path, shape: tuple[int, int], allowed_ids: set[int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot read annotation mask: {mask_path}")
    if mask.ndim == 3:
        if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(mask[..., 1], mask[..., 2]):
            raise ValueError(f"Annotation mask must be indexed grayscale: {mask_path}")
        mask = mask[..., 0]
    if mask.shape != shape:
        raise ValueError(f"Annotation mask shape mismatch for {mask_path}: {mask.shape} vs {shape}")
    unknown = set(np.unique(mask).tolist()) - {0, *allowed_ids}
    if unknown:
        raise ValueError(f"Annotation mask has unknown object IDs {sorted(unknown)}: {mask_path}")
    return mask


def _read_fixed_frames(video_path: Path, frame_numbers: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in frame_numbers:
            frames[index] = frame
        index += 1
    capture.release()
    missing = sorted(frame_numbers - frames.keys())
    if missing:
        raise ValueError(f"Video has missing annotated frame(s) {missing}: {video_path}")
    return frames


def _annotation_overlay(frame: np.ndarray, indexed: np.ndarray, objects: dict[int, dict[str, Any]]) -> np.ndarray:
    overlay = frame.copy()
    for object_id, obj in objects.items():
        color = (51, 204, 51) if obj["action"] == "remove" else (255, 204, 0)
        region = indexed == object_id
        overlay[region] = (
            overlay[region].astype(np.float32) * 0.35
            + np.asarray(color, dtype=np.float32) * 0.65
        ).astype(np.uint8)
    return overlay


def _write_previews(
    artifact_dir: Path,
    artifact_id: str,
    frame_index: int,
    frame: np.ndarray,
    indexed: np.ndarray,
    predicted: np.ndarray,
    objects: dict[int, dict[str, Any]],
) -> dict[str, str]:
    root = artifact_dir.resolve()
    target = root / artifact_id
    if target.is_symlink():
        raise ValueError(f"Preview target must not be a symlink: {target}")
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(root)
    except ValueError:
        raise ValueError(f"Preview target escapes artifact directory: {artifact_id}") from None
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Preview target escapes artifact directory: {artifact_id}") from None
    prefix = target / f"frame-{frame_index:06d}"
    annotated = _annotation_overlay(frame, indexed, objects)
    prediction = frame.copy()
    prediction[predicted] = (
        prediction[predicted].astype(np.float32) * 0.3
        + np.asarray((255, 0, 255), dtype=np.float32) * 0.7
    ).astype(np.uint8)
    remove = np.isin(indexed, [key for key, value in objects.items() if value["action"] == "remove"])
    keep = np.isin(indexed, [key for key, value in objects.items() if value["action"] == "keep"])
    errors = frame.copy()
    errors[predicted & remove] = (0, 220, 0)  # true positive
    errors[remove & ~predicted] = (0, 220, 255)  # missed remove target
    errors[predicted & ~remove & ~keep] = (0, 0, 255)  # false removal
    errors[predicted & keep] = (255, 0, 0)  # keep-region injury
    paths = {
        "annotations": f"{prefix.name}-annotations.png",
        "prediction": f"{prefix.name}-prediction.png",
        "errors": f"{prefix.name}-errors.png",
    }
    for key, image in (("annotations", annotated), ("prediction", prediction), ("errors", errors)):
        supplied_output = target / paths[key]
        if supplied_output.is_symlink():
            raise ValueError(f"Preview output must not be a symlink: {supplied_output}")
        output = supplied_output.resolve()
        try:
            output.relative_to(root)
        except ValueError:
            raise ValueError(f"Preview output escapes artifact directory: {output}") from None
        if output.exists() and not output.is_file():
            raise ValueError(f"Preview output must be a regular file: {output}")
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Cannot write preview: {output}")
        paths[key] = str(output.relative_to(root))
    return paths


def _write_json_report(
    destination: Path,
    report: dict[str, Any],
    protected_paths: list[Path],
) -> None:
    """Write a report atomically without following an existing symlink."""
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Fact-baseline report must not be a symlink: {destination}")
    resolved_destination = destination.resolve()
    if resolved_destination in {path.resolve() for path in protected_paths}:
        raise ValueError(
            f"Fact-baseline report must not overwrite an input: {resolved_destination}"
        )
    if resolved_destination.exists() and not resolved_destination.is_file():
        raise ValueError(
            f"Fact-baseline report must be a regular file: {resolved_destination}"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved_destination.parent,
            prefix=f".{resolved_destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, resolved_destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _artifact_id(video_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(video_name).stem).strip("-") or "video"
    fingerprint = hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{fingerprint}"


def _candidate_masks(candidates: list[Any], shape: tuple[int, int]) -> list[tuple[Any, np.ndarray]]:
    return [
        (candidate, np.asarray(mask_from_candidates([candidate], shape)).squeeze() > 0)
        for candidate in candidates
    ]


def _visible_annotation_matches(
    indexed: np.ndarray,
    objects: dict[int, dict[str, Any]],
    candidate_masks: list[tuple[Any, np.ndarray]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    """Separate classifier evidence from the selected removal decision."""
    matches = []
    for object_id, obj in objects.items():
        actual = indexed == object_id
        if not actual.any():
            continue
        hits = [candidate for candidate, mask in candidate_masks if np.any(mask & actual)]
        selected_hits = [candidate for candidate in hits if candidate.id in selected_ids]
        candidate_types = sorted({candidate.type for candidate in hits})
        selected_types = sorted({candidate.type for candidate in selected_hits})
        selected_masks = [
            mask for candidate, mask in candidate_masks
            if candidate.id in selected_ids and np.any(mask & actual)
        ]
        selected_prediction = (
            np.logical_or.reduce(selected_masks) if selected_masks else np.zeros_like(actual)
        )
        selected_coverage = float((selected_prediction & actual).sum()) / float(actual.sum())
        matches.append(
            {
                "object_id": object_id,
                "annotation_type": obj["type"],
                "action": obj["action"],
                "candidate_types": candidate_types,
                "selected_candidate_types": selected_types,
                "classifier_semantic_match": obj["type"] in candidate_types,
                "selected_prediction_coverage": round(selected_coverage, 6),
                "selection_intent_match": bool(selected_hits) if obj["action"] == "remove" else not bool(selected_hits),
            }
        )
    return matches


def _matched_object_metrics(
    indexed: np.ndarray,
    objects: dict[int, dict[str, Any]],
    candidate_masks: list[tuple[Any, np.ndarray]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    """Score each visible object only against selected candidates that hit it.

    Global output quality is reported separately as the remove-union metric.
    This local score avoids treating a different, valid remove object as a
    false positive for the object currently being inspected.
    """
    metrics = []
    for object_id, obj in objects.items():
        actual = indexed == object_id
        if not actual.any():
            continue
        matched_masks = [
            mask for candidate, mask in candidate_masks
            if candidate.id in selected_ids and np.any(mask & actual)
        ]
        prediction = np.logical_or.reduce(matched_masks) if matched_masks else np.zeros_like(actual)
        if obj["action"] == "remove":
            metrics.append(
                {
                    "object_id": object_id,
                    "action": "remove",
                    "region_jaccard": round(_compute_mask_iou(prediction, actual), 6),
                    "boundary_f": round(compute_boundary_f(prediction, actual), 6),
                }
            )
        else:
            metrics.append(
                {
                    "object_id": object_id,
                    "action": "keep",
                    "prediction_coverage": round(float((prediction & actual).sum()) / float(actual.sum()), 6),
                }
            )
    return metrics


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def evaluate_fact_baseline(
    input_dir: str,
    manifest_path: str,
    output_path: str,
    artifact_dir: str,
    mask_dir: str | None = None,
    detect_mode: str = "balanced",
    ocr_mode: str = "off",
    require_clean_git: bool = False,
) -> dict[str, Any]:
    """Evaluate fixed indexed annotations and write the stable report schema."""
    manifest = _load_fact_manifest(manifest_path, input_dir)
    manifest_path_obj = Path(manifest_path).resolve()
    manifest_root = manifest_path_obj.parent
    input_root = Path(input_dir).resolve()
    artifacts = Path(artifact_dir).resolve()
    video_reports: list[dict[str, Any]] = []
    frame_union_jaccards: list[float] = []
    frame_union_boundary_f: list[float] = []
    object_jaccards: list[float] = []
    object_boundary_f: list[float] = []
    keep_injury: list[float] = []
    no_target_false_removal: list[float] = []
    all_semantic_matches: list[bool] = []
    all_selection_matches: list[bool] = []
    input_paths: list[tuple[str, Path]] = [
        (
            video_spec["file"],
            _resolve_within(input_root, video_spec["file"], "Manifest video"),
        )
        for video_spec in manifest["videos"]
    ]
    annotation_paths: list[tuple[str, Path]] = [
        (f"manifest/{manifest_path_obj.name}", manifest_path_obj)
    ]
    for video_spec in manifest["videos"]:
        for frame_spec in video_spec["frames"]:
            annotation_paths.append(
                (
                    frame_spec["mask"],
                    _resolve_within(
                        manifest_root, frame_spec["mask"], "Annotation mask"
                    ),
                )
            )
    legacy_calibration_paths: list[tuple[str, Path]] = []
    if mask_dir:
        for video_spec in manifest["videos"]:
            golden_path = Path(mask_dir) / f"{Path(video_spec['file']).stem}_mask.png"
            if golden_path.is_file():
                legacy_calibration_paths.append(
                    (f"legacy/{golden_path.name}", golden_path.resolve())
                )
    protected_paths = [
        path for _, path in input_paths + annotation_paths + legacy_calibration_paths
    ]
    if require_clean_git:
        _require_clean_git_worktree(protected_paths)

    for video_spec in manifest["videos"]:
        video_name = video_spec["file"]
        video_path = _resolve_within(input_root, video_name, "Manifest video")
        result, selected, generated, execution_path = _detect_video(
            str(video_path), detect_mode, ocr_mode
        )
        shape = tuple(result.frame_shape)
        if generated.shape != shape:
            raise ValueError(f"Generated mask shape mismatch for {video_path}: {generated.shape} vs {shape}")
        objects = {obj["id"]: obj for obj in video_spec["objects"]}
        fixed_frames = _read_fixed_frames(video_path, {entry["frame"] for entry in video_spec["frames"]})
        candidate_masks = _candidate_masks(result.candidates, shape)
        selected_ids = {candidate.id for candidate in selected}
        frame_reports = []
        legacy = _legacy_calibration_metric(generated, str(video_path), mask_dir)
        for frame_spec in video_spec["frames"]:
            frame_index = frame_spec["frame"]
            mask_name = frame_spec["mask"]
            mask_path = _resolve_within(manifest_root, mask_name, "Annotation mask")
            indexed = _read_indexed_mask(mask_path, shape, set(objects))
            remove = np.isin(indexed, [key for key, value in objects.items() if value["action"] == "remove"])
            keep = np.isin(indexed, [key for key, value in objects.items() if value["action"] == "keep"])
            has_remove = bool(remove.any())
            jaccard = _compute_mask_iou(generated, remove) if has_remove else None
            boundary_f = compute_boundary_f(generated, remove) if has_remove else None
            frame_keep_injury = float((generated & keep).sum()) / float(keep.sum()) if keep.any() else None
            false_removal = float(generated.sum()) / float(generated.size) if not has_remove else None
            matches = _visible_annotation_matches(
                indexed, objects, candidate_masks, selected_ids
            )
            object_metrics = _matched_object_metrics(
                indexed, objects, candidate_masks, selected_ids
            )
            for metric in object_metrics:
                if metric["action"] == "remove":
                    object_jaccards.append(metric["region_jaccard"])
                    object_boundary_f.append(metric["boundary_f"])
                else:
                    keep_injury.append(metric["prediction_coverage"])
            all_semantic_matches.extend(
                match["classifier_semantic_match"] for match in matches
            )
            all_selection_matches.extend(
                match["selection_intent_match"] for match in matches
            )
            previews = _write_previews(
                artifacts,
                _artifact_id(video_name),
                frame_index,
                fixed_frames[frame_index],
                indexed,
                generated,
                objects,
            )
            frame_reports.append(
                {
                    "frame": frame_index,
                    "annotation_mask": frame_spec["mask"],
                    "remove_region_jaccard": round(jaccard, 6) if jaccard is not None else None,
                    "remove_boundary_f": round(boundary_f, 6) if boundary_f is not None else None,
                    "keep_prediction_coverage": round(frame_keep_injury, 6) if frame_keep_injury is not None else None,
                    "no_remove_false_removal_area_ratio": round(false_removal, 6) if false_removal is not None else None,
                    "object_metrics": object_metrics,
                    "visible_annotation_matches": matches,
                    "previews": previews,
                }
            )
            if jaccard is not None:
                frame_union_jaccards.append(jaccard)
            if boundary_f is not None:
                frame_union_boundary_f.append(boundary_f)
            if false_removal is not None:
                no_target_false_removal.append(false_removal)
        video_reports.append(
            {
                "video": video_name,
                "artifact_id": _artifact_id(video_name),
                "frame_shape": list(shape),
                "objects": video_spec["objects"],
                "detect_mode": detect_mode,
                "ocr_mode": ocr_mode,
                "detector_execution_path": execution_path,
                "candidate_count": len(result.candidates),
                "selected_count": len(selected),
                "candidates": _candidate_report(result, selected),
                "legacy_calibration_metric": legacy,
                "frames": frame_reports,
            }
        )
    report = {
        "schema_version": FACT_BASELINE_REPORT_SCHEMA_VERSION,
        "report_kind": "detection_fact_baseline",
        "provenance": {
            "git": _git_provenance(),
            "videowipe_version": videowipe.__version__,
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__,
            "detect_mode": detect_mode,
            "ocr_mode": ocr_mode,
            "evaluator_source_sha256": _evaluator_source_hash(),
            "input_sha256": _sha256_named_files(input_paths),
            "annotation_sha256": _sha256_named_files(annotation_paths),
            "legacy_calibration_sha256": (
                _sha256_named_files(legacy_calibration_paths)
                if legacy_calibration_paths else None
            ),
        },
        "aggregation": {
            "frame_remove_union": "Mean across annotated frames with a visible remove target; compares the selected union with the remove union.",
            "visible_object": "Mean across visible annotation instances only; each object uses only selected candidates that overlap that object.",
            "semantic": "Classifier type and selection intent are evaluated separately on visible annotation instances only.",
        },
        "macro_average": {
            "frame_remove_union_region_jaccard": _mean(frame_union_jaccards),
            "frame_remove_union_boundary_f": _mean(frame_union_boundary_f),
            "visible_remove_object_region_jaccard": _mean(object_jaccards),
            "visible_remove_object_boundary_f": _mean(object_boundary_f),
            "visible_keep_prediction_coverage": _mean(keep_injury),
            "no_remove_false_removal_area_ratio": _mean(no_target_false_removal),
            "visible_annotation_semantic_match_rate": _mean([float(value) for value in all_semantic_matches]),
            "visible_annotation_selection_intent_match_rate": _mean([float(value) for value in all_selection_matches]),
        },
        "videos": video_reports,
    }
    _write_json_report(Path(output_path), report, protected_paths)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoWipe detection regression and fact-baseline evaluator")
    parser.add_argument("input_dir", help="Directory containing calibration videos")
    parser.add_argument("--manifest", help="Fact-baseline indexed annotation manifest")
    parser.add_argument("--output", default="result/fact-baseline/report.json", help="Fact-baseline JSON output")
    parser.add_argument("--artifact-dir", default="result/fact-baseline/previews", help="Fact-baseline preview directory")
    parser.add_argument("--require-clean-git", action="store_true", help="Refuse a formal fact baseline from a dirty worktree")
    parser.add_argument("--write-regression-snapshot", dest="write_snapshot", action="store_true", help="Write detection_regression_snapshot.json")
    parser.add_argument("--write-baseline", dest="write_legacy_snapshot", action="store_true", help="Compatibility mode: write detection_baseline.json")
    parser.add_argument("--compare-regression-snapshot", action="store_true", help="Compare detection_regression_snapshot.json")
    parser.add_argument("--compare-baseline", dest="compare_legacy_snapshot", action="store_true", help="Compatibility mode: compare detection_baseline.json")
    parser.add_argument("--mask-dir", help="Legacy static Golden masks ({stem}_mask.png)")
    parser.add_argument("--detect-mode", default="balanced", choices=["fast", "balanced", "sensitive"])
    parser.add_argument("--ocr", default="off", choices=["auto", "off", "rapidocr"])
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        parser.error(f"Not a directory: {args.input_dir}")
    try:
        if args.manifest:
            report = evaluate_fact_baseline(
                args.input_dir, args.manifest, args.output, args.artifact_dir,
                mask_dir=args.mask_dir, detect_mode=args.detect_mode, ocr_mode=args.ocr,
                require_clean_git=args.require_clean_git,
            )
            print(f"Fact baseline report: {args.output}")
            print(json.dumps(report["macro_average"], ensure_ascii=False))
            return
        videos = sorted(
            str(path) for path in Path(args.input_dir).iterdir()
            if path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        )
        if not videos:
            parser.error(f"No video files found in: {args.input_dir}")
        reports = []
        for path in videos:
            try:
                report = _eval_video(path, args.mask_dir, args.detect_mode, args.ocr)
            except Exception as exc:  # noqa: BLE001 - one bad video must not hide the rest.
                print(f"ERROR processing {path}: {exc}", file=sys.stderr)
                reports.append(
                    {
                        "video": Path(path).name,
                        "error": str(exc),
                        "candidate_count": 0,
                        "selected_count": 0,
                        "empty_detection": True,
                        "candidates": [],
                    }
                )
                continue
            reports.append(report)
            _print_report(report)
        new_snapshot_path = Path(args.input_dir) / "detection_regression_snapshot.json"
        legacy_snapshot_path = Path(args.input_dir) / "detection_baseline.json"
        write_paths = []
        if args.write_snapshot:
            write_paths.append(new_snapshot_path)
        if args.write_legacy_snapshot:
            write_paths.append(legacy_snapshot_path)
        for snapshot_path in write_paths:
            snapshot_path.write_text(
                json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Regression snapshot written to {snapshot_path}")
        comparisons_are_stable = True
        if args.compare_regression_snapshot:
            comparisons_are_stable &= _compare_against_regression_snapshot(
                reports, str(new_snapshot_path)
            )
        if args.compare_legacy_snapshot:
            comparisons_are_stable &= _compare_against_regression_snapshot(
                reports, str(legacy_snapshot_path)
            )
        total_candidates = sum(report["candidate_count"] for report in reports)
        empty_count = sum(1 for report in reports if report["empty_detection"])
        errors = sum(1 for report in reports if "error" in report)
        abnormal_count = sum(
            1
            for report in reports
            for candidate in report.get("candidates", [])
            if candidate.get("abnormal_bbox")
        )
        print(
            f"Summary: {len(videos)} videos, {total_candidates} total candidates, "
            f"{empty_count} empty detections, {errors} errors, "
            f"{abnormal_count} abnormal bboxes"
        )
        if errors:
            raise SystemExit(1)
        if not comparisons_are_stable or abnormal_count:
            raise SystemExit(2)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
