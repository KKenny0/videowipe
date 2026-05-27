#!/usr/bin/env python
"""Eval harness for clean detection quality.

Usage:
    python scripts/eval_clean_detection.py input/detext_examples
    python scripts/eval_clean_detection.py input/detext_examples --write-baseline
    python scripts/eval_clean_detection.py input/detext_examples --compare-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from videowipe.detect import (
    detect_clean_candidates,
    select_clean_candidates,
)


def _eval_video(video_path: str) -> dict:
    """Run clean detection on a single video and return a report dict."""
    result = detect_clean_candidates(video_path, subtitle_fallback="light")
    selected = select_clean_candidates(result.candidates)
    selected_ids = {c.id for c in selected}
    h, w = result.frame_shape

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
        "candidate_count": len(result.candidates),
        "selected_count": len(selected),
        "frame_shape": list(result.frame_shape),
        "empty_detection": empty,
        "candidates": candidates_report,
    }
    if empty:
        report["warning"] = "No candidates detected"
    return report


def _print_report(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Video: {report['video']}")
    print(f"Shape: {report['frame_shape']}")
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
            report = _eval_video(video_path)
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
    print(f"\n{'=' * 60}")
    print(f"Summary: {len(videos)} videos, {total_candidates} total candidates, "
          f"{empty_count} empty detections, {errors} errors")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
