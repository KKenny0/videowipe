"""Process a batch with one long-lived VideoWipe engine."""
from __future__ import annotations

import argparse
from pathlib import Path

from videowipe import ProgressEvent, WipeEngine, WipeRequest


def report_progress(video: Path):
    def report(event: ProgressEvent) -> None:
        if event.fraction is None:
            print(f"{video.name}: {event.phase}")
        else:
            print(f"{video.name}: {event.phase} {event.fraction:.0%}")

    return report


def process_batch(videos, mask, output_dir, *, weight=None, device="auto"):
    """Reuse one model; *mask* may be None for per-video auto-detection."""
    results = []
    with WipeEngine(weight=weight, device=device) as engine:
        for index, video in enumerate(videos, start=1):
            request = WipeRequest(
                video=video,
                mask=mask,
                output_dir=output_dir / f"{index:04d}-{video.stem}",
            )
            results.append(engine.run(request, on_progress=report_progress(video)))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--mask",
        type=Path,
        help="Shared mask for same-layout inputs; omit for per-video detection",
    )
    parser.add_argument("--output", default=Path("result"), type=Path)
    parser.add_argument("--weight")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)

    results = process_batch(
        args.videos,
        args.mask,
        args.output,
        weight=args.weight,
        device=args.device,
    )
    for result in results:
        print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
