"""CLI entry point for videowipe."""
import argparse
import sys

from videowipe.engine import WipeEngine


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="videowipe",
        description="STTN-based video inpainting: remove subtitles, logos, and watermarks",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name, help_text in [("detext", "Remove subtitles"), ("delogo", "Remove logos")]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("-v", "--video", required=True, help="Input video path")
        sub.add_argument("-m", "--mask", required=True, help="Mask image path")
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
    engine.process(video=args.video, mask=args.mask, output=args.output)
    engine.cleanup()


if __name__ == "__main__":
    main()
