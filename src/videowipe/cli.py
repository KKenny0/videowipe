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
    sub.add_argument(
        "-g", "--gap", type=int, default=25,
        help=(
            "Frames per segment (default: 25, conservative performance/quality "
            "balance); larger values add context but cost grows superlinearly"
        ),
    )
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
    # A hand-drawn mask and a reviewed/agent-edited WipePlan are two ways to
    # skip detection; they are mutually exclusive (the engine enforces this
    # too, but argparse gives a cleaner pre-flight error).
    mask_or_plan = clean.add_mutually_exclusive_group()
    mask_or_plan.add_argument("-m", "--mask", default=None,
                              help="Mask image path (skip detection if provided)")
    mask_or_plan.add_argument("--plan", default=None,
                              help="Execute an existing wipe_plan.json instead of detecting")
    clean.add_argument("-o", "--output", default="result/", help="Output directory")
    clean.add_argument("-w", "--weight", default=None, help="Model weight path")
    clean.add_argument(
        "-g", "--gap", type=int, default=25,
        help=(
            "Frames per segment (default: 25, conservative performance/quality "
            "balance); larger values add context but cost grows superlinearly"
        ),
    )
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

    serve = subparsers.add_parser("serve", help="Start local web server")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=8000, help="Bind port")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print(
                "videowipe: install web support with `pip install videowipe[web]`",
                file=sys.stderr,
            )
            sys.exit(1)
        uvicorn.run(
            "videowipe.server.app:app",
            host=args.host,
            port=args.port,
        )
        return

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
            plan=getattr(args, "plan", None),
        )
    except Exception as exc:
        print(f"videowipe: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
