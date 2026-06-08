#!/usr/bin/env python
"""Benchmark the full VideoWipe pipeline (detection + inpainting).

Usage:
    python scripts/benchmark_pipeline.py input/detext_examples \
        --mask-dir input/detext_examples/mask \
        --output-dir result/benchmark

The script runs two phases per video:

1. Detection eval via eval_clean_detection (no model loading).
2. Full detext pipeline via WipeEngine (loads model, processes video).

Results are written as JSON to the output directory. Benchmark output is
not intended to be committed to git.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from videowipe.engine import WipeEngine


def _benchmark_video(
    video_path: str,
    mask_path: str | None,
    output_dir: str,
    mask_source_label: str,
) -> dict:
    """Run the full detext pipeline on one video and collect benchmark data."""
    os.makedirs(output_dir, exist_ok=True)

    engine = WipeEngine(task="detext")
    try:
        out_path = engine.process(
            video=video_path,
            mask=mask_path,
            output=output_dir,
        )
    except Exception as exc:
        return {"video": os.path.basename(video_path), "error": str(exc)}
    finally:
        engine.cleanup()

    # Read the benchmark.json that WipeEngine.process() wrote
    bm_path = os.path.join(output_dir, "benchmark.json")
    if os.path.exists(bm_path):
        with open(bm_path, "r", encoding="utf-8") as fh:
            bm = json.load(fh)
        bm["mask_source"] = mask_source_label
        return bm

    return {"video": os.path.basename(video_path), "error": "benchmark.json not found"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VideoWipe pipeline")
    parser.add_argument("input_dir", help="Directory containing test videos")
    parser.add_argument("--mask-dir",
                        help="Directory with manual masks ({stem}_mask.png)")
    parser.add_argument("--output-dir", default="result/benchmark",
                        help="Output directory for benchmark results")
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

        video_out_dir = os.path.join(args.output_dir, video_stem)
        print(f"\n--- {os.path.basename(video_path)} ---")
        result = _benchmark_video(video_path, mask_path, video_out_dir, mask_source)
        results.append(result)

        if "error" in result and result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            timing = result.get("timing", {})
            print(f"  Total: {timing.get('total_s', '?')}s  "
                  f"Detection: {timing.get('detection_s', 'N/A')}s  "
                  f"Model load: {timing.get('model_load_s', 'N/A')}s  "
                  f"Inpainting: {timing.get('inpainting_s', 'N/A')}s")
            print(f"  Backend: {result.get('backend', '?')}  "
                  f"Mask area: {result.get('mask_area_ratio', '?')}")
            print(f"  Output: {result.get('output_path', '?')}")

    # Write aggregate results
    report_path = os.path.join(args.output_dir, "benchmark_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nReport written to {report_path}")

    errors = sum(1 for r in results if r.get("error"))
    if errors:
        print(f"{errors} video(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
