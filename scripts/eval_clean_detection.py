#!/usr/bin/env python
"""Eval harness for clean detection quality.

Usage:
    python scripts/eval_clean_detection.py input/detext_examples
    python scripts/eval_clean_detection.py input/detext_examples --write-baseline
    python scripts/eval_clean_detection.py input/detext_examples --compare-baseline
    python scripts/eval_clean_detection.py input/detext_examples --detect-mode sensitive --ocr rapidocr
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from videowipe.detect import (
    detect_clean_candidates,
    mask_from_candidates,
    resolve_detect_params,
    select_clean_candidates,
)


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    a = np.asarray(mask_a).squeeze().astype(bool).flatten()
    b = np.asarray(mask_b).squeeze().astype(bool).flatten()
    if len(a) != len(b):
        return 0.0
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return intersection / union if union > 0 else 0.0


def _build_recognizer(ocr_mode: str):
    """Build an OCR recognizer callable, mirroring engine logic."""
    if ocr_mode == "off":
        return None
    try:
        from videowipe.ocr import recognize_text, _get_engine
        _get_engine()
        return recognize_text
    except Exception:
        if ocr_mode == "rapidocr":
            raise RuntimeError(
                "OCR mode 'rapidocr' requested but rapidocr-onnxruntime "
                "is not installed. Install it with: pip install videowipe[ocr]"
            ) from None
        return None


def _eval_video(
    video_path: str,
    mask_dir: str | None = None,
    detect_mode: str = "balanced",
    ocr_mode: str = "off",
) -> dict:
    """Run clean detection on a single video and return a report dict."""
    mode_params = resolve_detect_params(detect_mode)
    recognizer = _build_recognizer(ocr_mode)
    result = detect_clean_candidates(
        video_path,
        sample_count=mode_params["sample_count"],
        consistency=mode_params["consistency"],
        subtitle_fallback=mode_params["subtitle_fallback"],
        recognizer=recognizer,
    )
    selected = select_clean_candidates(result.candidates)
    selected_ids = {c.id for c in selected}
    h, w = result.frame_shape

    # Generate mask from selected candidates
    generated_mask = mask_from_candidates(selected, result.frame_shape)
    gen_pixels = int(np.sum(generated_mask > 0))
    gen_area_ratio = round(gen_pixels / (h * w), 6) if h and w else 0.0

    candidates_report: list[dict] = []
    for c in result.candidates:
        x1, y1, x2, y2 = c.bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h) if w and h else 0.0
        abnormal = x2 > w or y2 > h or x1 < 0 or y1 < 0
        candidates_report.append({
            "id": c.id,
            "type": c.type,
            "bbox": list(c.bbox),
            "area_ratio": round(area_ratio, 4),
            "selected": c.id in selected_ids,
            "reason": c.reason,
            "abnormal_bbox": abnormal,
            "text_samples": c.text_samples[:3],
        })

    empty = len(result.candidates) == 0
    report: dict = {
        "video": os.path.basename(video_path),
        "detect_mode": detect_mode,
        "ocr_mode": ocr_mode,
        "candidate_count": len(result.candidates),
        "selected_count": len(selected),
        "frame_shape": list(result.frame_shape),
        "empty_detection": empty,
        "generated_mask_area_ratio": gen_area_ratio,
        "candidates": candidates_report,
    }
    if empty:
        report["warning"] = "No candidates detected"

    # Golden mask comparison
    if mask_dir:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        golden_path = os.path.join(mask_dir, f"{video_stem}_mask.png")
        if os.path.exists(golden_path):
            golden_img = cv2.imread(golden_path, cv2.IMREAD_GRAYSCALE)
            if golden_img is not None:
                _, golden_bin = cv2.threshold(golden_img, 127, 1, cv2.THRESH_BINARY)
                golden_area = int(np.sum(golden_bin > 0))
                report["golden_mask_area_ratio"] = round(
                    golden_area / (h * w), 6
                ) if h and w else 0.0
                report["mask_iou"] = round(_compute_mask_iou(generated_mask, golden_bin), 4)
            else:
                report["golden_mask_missing"] = f"cannot read: {golden_path}"
        else:
            report["golden_mask_missing"] = golden_path

    return report


def _print_report(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Video: {report['video']}")
    print(f"Shape: {report['frame_shape']}")
    gen_ratio = report.get("generated_mask_area_ratio", 0)
    print(f"Mask area ratio: {gen_ratio:.4f}")
    if "mask_iou" in report:
        golden_ratio = report.get("golden_mask_area_ratio", 0)
        print(f"Golden IoU: {report['mask_iou']:.4f}  Golden area: {golden_ratio:.4f}")
    if report.get("golden_mask_missing"):
        print(f"MISSING GOLDEN: {report['golden_mask_missing']}")
    print(f"Candidates: {report['candidate_count']}  Selected: {report['selected_count']}")
    if report.get("warning"):
        print(f"WARNING: {report['warning']}")
    print("-" * 60)
    for c in report["candidates"]:
        sel = "SEL" if c["selected"] else "   "
        flag = "!" if c["abnormal_bbox"] else " "
        print(
            f"  {sel} {flag} {c['id']:>4}  {c['type']:<14}  "
            f"bbox={c['bbox']}  area={c['area_ratio']:.3f}  {c['reason']}"
        )
    if not report["candidates"]:
        print("  (no candidates)")


def _compare_against_baseline(reports: list[dict], baseline_path: str) -> bool:
    """Compare reports against a saved baseline. Returns True if stable."""
    if not os.path.exists(baseline_path):
        print(f"Baseline not found: {baseline_path}")
        print("Run with --write-baseline first.")
        return False

    with open(baseline_path, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    baseline_by_video = {b["video"]: b for b in baseline}
    all_stable = True

    for report in reports:
        base = baseline_by_video.get(report["video"])
        if base is None:
            print(f"NEW: {report['video']} (not in baseline)")
            all_stable = False
            continue

        if report["candidate_count"] != base["candidate_count"]:
            print(
                f"DRIFT: {report['video']} candidate count "
                f"{report['candidate_count']} vs baseline {base['candidate_count']}"
            )
            all_stable = False

        if report["empty_detection"] != base["empty_detection"]:
            print(
                f"DRIFT: {report['video']} empty_detection changed: "
                f"{report['empty_detection']} vs {base['empty_detection']}"
            )
            all_stable = False

        base_types = {c["type"] for c in base["candidates"]}
        report_types = {c["type"] for c in report["candidates"]}
        if base_types != report_types:
            print(f"DRIFT: {report['video']} candidate types changed: {report_types} vs {base_types}")
            all_stable = False

    # Check for removed videos
    report_videos = {r["video"] for r in reports}
    for video_name in baseline_by_video:
        if video_name not in report_videos:
            print(f"REMOVED: {video_name} (in baseline but not in current run)")
            all_stable = False

    if all_stable:
        print("All videos stable against baseline.")
    return all_stable


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval harness for clean detection")
    parser.add_argument("input_dir", help="Directory containing test videos")
    parser.add_argument("--write-baseline", action="store_true",
                        help="Write results as baseline JSON")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare results against saved baseline")
    parser.add_argument("--mask-dir",
                        help="Directory with golden masks ({stem}_mask.png)")
    parser.add_argument("--detect-mode", default="balanced",
                        choices=["fast", "balanced", "sensitive"],
                        help="Detection preset (default: balanced)")
    parser.add_argument("--ocr", default="off",
                        choices=["auto", "off", "rapidocr"],
                        help="OCR text recognition (default: off)")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    videos = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in video_exts
    )

    if not videos:
        print(f"No video files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(videos)} video(s) in {input_dir}")

    reports: list[dict] = []
    for video_path in videos:
        try:
            report = _eval_video(
                video_path,
                mask_dir=args.mask_dir,
                detect_mode=args.detect_mode,
                ocr_mode=args.ocr,
            )
        except Exception as exc:
            print(f"ERROR processing {video_path}: {exc}", file=sys.stderr)
            reports.append({
                "video": os.path.basename(video_path),
                "error": str(exc),
                "candidate_count": 0,
                "selected_count": 0,
                "empty_detection": True,
                "candidates": [],
            })
            continue
        reports.append(report)
        _print_report(report)

    # JSON output
    baseline_path = os.path.join(input_dir, "detection_baseline.json")
    if args.write_baseline:
        with open(baseline_path, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2, ensure_ascii=False)
        print(f"\nBaseline written to {baseline_path}")

    if args.compare_baseline:
        stable = _compare_against_baseline(reports, baseline_path)
        sys.exit(0 if stable else 2)

    # Summary
    total_candidates = sum(r["candidate_count"] for r in reports)
    empty_count = sum(1 for r in reports if r["empty_detection"])
    errors = sum(1 for r in reports if "error" in r)
    abnormal_count = sum(
        1 for r in reports
        for c in r.get("candidates", [])
        if c.get("abnormal_bbox")
    )
    missing_goldens = [r["video"] for r in reports if r.get("golden_mask_missing")]
    print(f"\n{'=' * 60}")
    print(f"Summary: {len(videos)} videos, {total_candidates} total candidates, "
          f"{empty_count} empty detections, {errors} errors, "
          f"{abnormal_count} abnormal bboxes")
    if missing_goldens:
        print(f"Missing goldens: {', '.join(missing_goldens)}")

    exit_code = 0
    if errors:
        exit_code = 1
    elif abnormal_count > 0:
        exit_code = 2
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
