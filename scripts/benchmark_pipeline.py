#!/usr/bin/env python
"""Benchmark the full VideoWipe pipeline (detection + inpainting).

Usage:
    # Reproducible Chinese One baseline:
    python scripts/benchmark_pipeline.py input/detext_examples/chinese1.mp4 \
        --repeat 3 --detect-mode balanced --ocr off --gap 25 \
        --output-dir result/benchmark-chinese1

    python scripts/benchmark_pipeline.py input/detext_examples \
        --mask-dir input/detext_examples/mask \
        --output-dir result/benchmark

    # With detect-mode and OCR:
    python scripts/benchmark_pipeline.py input/detext_examples \
        --detect-mode sensitive --ocr rapidocr \
        --output-dir result/benchmark-sensitive-ocr

The script runs the full clean pipeline via WipeEngine (loads model,
processes video). Results are written as JSON to the output directory.
Benchmark output is not intended to be committed to git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time

try:
    import resource
except ImportError:  # Windows
    resource = None

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import cv2

from videowipe.engine import WipeEngine

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


_RSS_NOTE = (
    "Current benchmark process lifetime high-water RSS; repeated runs share "
    "this value, so repeat > 1 is not a per-run memory measurement. Null means "
    "the platform does not expose resource.getrusage."
)


def _process_peak_rss_so_far_mib() -> float | None:
    if resource is None:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(raw / divisor, 3)


def _video_metadata(video_path: str) -> dict:
    digest = hashlib.sha256()
    with open(video_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        return {
            "path": os.path.abspath(video_path),
            "sha256": digest.hexdigest(),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "frame_count": int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) + 0.5),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        }
    finally:
        cap.release()


def _find_videos(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return (
            [input_path]
            if os.path.splitext(input_path)[1].lower() in _VIDEO_EXTS
            else []
        )
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if os.path.splitext(name)[1].lower() in _VIDEO_EXTS
        )
    return []


def _median_timings(runs: list[dict]) -> dict:
    medians = {}
    for key in ("total_s", "detection_s", "model_load_s", "inpainting_s"):
        values = [
            run.get("timing", {}).get(key)
            for run in runs
            if isinstance(run.get("timing", {}).get(key), (int, float))
        ]
        medians[key] = round(float(statistics.median(values)), 3) if values else None
    return medians


def _benchmark_video(
    video_path: str,
    mask_path: str | None,
    output_dir: str,
    mask_source_label: str,
    detect_mode: str = "balanced",
    ocr_mode: str = "auto",
    gap: int = 25,
) -> dict:
    """Run the full clean pipeline on one video and collect benchmark data."""
    os.makedirs(output_dir, exist_ok=True)

    started = time.monotonic()
    engine = WipeEngine(
        task="clean",
        detect_mode=detect_mode,
        ocr=ocr_mode,
        gap=gap,
    )
    result = None
    try:
        engine.process(
            video=video_path,
            mask=mask_path,
            output=output_dir,
        )
    except Exception as exc:
        result = {"video": os.path.basename(video_path), "error": str(exc)}
    finally:
        engine.cleanup()
    if result is None:
        bm_path = os.path.join(output_dir, "benchmark.json")
        if os.path.exists(bm_path):
            with open(bm_path, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            result["mask_source"] = mask_source_label
        else:
            result = {
                "video": os.path.basename(video_path),
                "error": "benchmark.json not found",
            }
    result["wall_time_s"] = round(time.monotonic() - started, 3)
    result["process_peak_rss_so_far_mib"] = _process_peak_rss_so_far_mib()
    result["output_dir"] = os.path.abspath(output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VideoWipe pipeline",
        epilog=f"Memory metric: {_RSS_NOTE}",
    )
    parser.add_argument("input_path", help="Video file or directory containing videos")
    parser.add_argument(
        "--mask-dir", help="Directory with manual masks ({stem}_mask.png)"
    )
    parser.add_argument(
        "--output-dir",
        default="result/benchmark",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--detect-mode",
        default="balanced",
        choices=["fast", "balanced", "sensitive"],
        help="Detection preset (default: balanced)",
    )
    parser.add_argument(
        "--ocr",
        default="auto",
        choices=["auto", "off", "rapidocr"],
        help="OCR text recognition (default: auto)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=25,
        help=(
            "Frames per segment (default: 25, conservative performance/quality "
            "balance); larger values add context but cost grows superlinearly"
        ),
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Runs per video (default: 1)"
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    videos = _find_videos(args.input_path)
    if not videos:
        print(f"No video files found at: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking {len(videos)} video(s)")

    results: list[dict] = []
    for video_path in videos:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        mask_path = None
        mask_source = "auto"

        if args.mask_dir:
            candidate = os.path.join(args.mask_dir, f"{video_stem}_mask.png")
            if os.path.exists(candidate):
                mask_path = candidate
                mask_source = "manual"

        print(f"\n--- {os.path.basename(video_path)} ---")
        runs = []
        for repeat_index in range(1, args.repeat + 1):
            run_dir = os.path.join(
                args.output_dir,
                video_stem,
                f"run-{repeat_index:03d}",
            )
            run = _benchmark_video(
                video_path,
                mask_path,
                run_dir,
                mask_source,
                detect_mode=args.detect_mode,
                ocr_mode=args.ocr,
                gap=args.gap,
            )
            run["run"] = repeat_index
            runs.append(run)
            if run.get("error"):
                print(f"  Run {repeat_index}: ERROR: {run['error']}")
            else:
                print(
                    f"  Run {repeat_index}: total={run.get('timing', {}).get('total_s', '?')}s "
                    f"wall={run['wall_time_s']}s "
                    f"process_peak_rss_so_far={run['process_peak_rss_so_far_mib']} MiB"
                )
        results.append(
            {
                "input": _video_metadata(video_path),
                "runs": runs,
                "median_timing_s": _median_timings(runs),
            }
        )

    # Write aggregate results
    report_path = os.path.join(args.output_dir, "benchmark_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "environment": {
                    "python": platform.python_version(),
                    "opencv": cv2.__version__,
                    "platform": platform.platform(),
                },
                "config": {
                    "detect_mode": args.detect_mode,
                    "ocr": args.ocr,
                    "gap": args.gap,
                    "repeat": args.repeat,
                },
                "metric_notes": {
                    "process_peak_rss_so_far_mib": _RSS_NOTE,
                },
                "results": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nReport written to {report_path}")

    errors = sum(1 for result in results for run in result["runs"] if run.get("error"))
    if errors:
        print(f"{errors} run(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
