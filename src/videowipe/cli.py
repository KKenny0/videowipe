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
    sub.add_argument("--external-command", default=None,
                     help="External inpainting command (bypasses built-in STTN)")
    sub.add_argument("--model", default="sttn",
                     help="Inpainter name from the registry (default: sttn)")
    sub.add_argument("--propainter-dir", default=None,
                     help="Path to ProPainter source (used with --model propainter)")

    clean = subparsers.add_parser("clean", help="Clean subtitles and text overlays")
    clean.add_argument("video", help="Input video path")
    clean.add_argument("-m", "--mask", default=None,
                       help="Mask image path (skip detection if provided)")
    clean.add_argument("-o", "--output", default="result/", help="Output directory")
    clean.add_argument("-w", "--weight", default=None, help="Model weight path")
    clean.add_argument("-g", "--gap", type=int, default=200,
                       help="Segment length per pass; higher = better quality")
    clean.add_argument("-d", "--dual", action="store_true",
                       help="Show original video side-by-side")
    clean.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target type to clean: subtitle, timestamp, or watermark",
    )
    clean.add_argument(
        "--region",
        action="append",
        default=None,
        help="Region to clean: top-left, top-right, bottom-left, bottom-right, top, bottom, center",
    )
    clean.add_argument("--intent", default=None,
                       help="Natural-language cleanup intent")
    clean.add_argument("--agent", default=None,
                       help="Optional local agent CLI for intent selection")
    clean.add_argument("--preview", action="store_true",
                       help="Only write detection preview artifacts")
    clean.add_argument("--confirm", action="store_true",
                       help="Confirm detected targets before processing")
    clean.add_argument(
        "--detect-mode",
        choices=["fast", "balanced", "sensitive"],
        default="balanced",
        help="Detection preset: fast (24 frames), balanced (50), sensitive (80)",
    )
    clean.add_argument(
        "--ocr",
        choices=["auto", "off", "rapidocr"],
        default="auto",
        help="OCR text recognition: auto (use if installed), off, rapidocr (error if missing)",
    )
    clean.add_argument("--external-command", default=None,
                       help="External inpainting command (bypasses built-in STTN)")
    clean.add_argument("--model", default="sttn",
                       help="Inpainter name from the registry (default: sttn)")
    clean.add_argument("--propainter-dir", default=None,
                       help="Path to ProPainter source (used with --model propainter)")

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
        external_command=args.external_command,
        model=getattr(args, "model", "sttn"),
        propainter_dir=getattr(args, "propainter_dir", None),
        detect_mode=getattr(args, "detect_mode", "balanced"),
        ocr=getattr(args, "ocr", "auto"),
    )
    try:
        engine.process(
            video=args.video,
            mask=args.mask,
            output=args.output,
            targets=getattr(args, "target", None),
            intent=getattr(args, "intent", None),
            agent=getattr(args, "agent", None),
            regions=getattr(args, "region", None),
            preview=getattr(args, "preview", False),
            confirm=getattr(args, "confirm", False),
        )
    except Exception as exc:
        print(f"videowipe: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
