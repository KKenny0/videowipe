"""CLI entry point for videowipe."""
import argparse
import sys

from videowipe.engine import WipeEngine


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="videowipe",
        description="STTN-based video inpainting: remove hardcoded subtitles",
    )
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("detext", help="Remove subtitles")
    sub.add_argument("-v", "--video", required=True, help="Input video path")
    sub.add_argument("-m", "--mask", default=None,
                     help="Mask image path (auto-detect if omitted)")
    sub.add_argument("-o", "--output", default="result/", help="Output directory")
    sub.add_argument("-w", "--weight", default=None, help="Model weight path")
    sub.add_argument("-g", "--gap", type=int, default=200,
                     help="Segment length per pass; higher = better quality")
    sub.add_argument("-d", "--dual", action="store_true",
                     help="Show original video side-by-side")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = WipeEngine(
        task=args.command,
        weight=args.weight,
        gap=args.gap,
        dual=args.dual,
    )
    try:
        engine.process(video=args.video, mask=args.mask, output=args.output)
    except Exception as exc:
        print(f"videowipe: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
