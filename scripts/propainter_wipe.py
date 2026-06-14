"""Wrapper to run ProPainter as a VideoWipe external model adapter.

Usage (called by ``videowipe --model propainter``, or directly)::

    python scripts/propainter_wipe.py <video_path> <mask_path> <output_dir> \\
        [--propainter-dir /path/to/ProPainter]

ProPainter saves to ``<output>/<video_name>/inpaint_out.mp4``. This wrapper
extracts frames (workaround for ``torchvision.io.read_video`` removal), runs
ProPainter, and copies the result to ``<output_dir>/inpaint_out.mp4``.

The ProPainter source directory is resolved in priority order:

1. ``--propainter-dir`` argument
2. ``VIDEOWIPE_PROPINTER_DIR`` environment variable
3. ``../../models/ProPainter`` relative to this script (matches the README
   clone instruction)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import imageio


def _resolve_propainter_dir(arg=None):
    """Resolve the ProPainter source directory.

    Priority: explicit argument > ``VIDEOWIPE_PROPINTER_DIR`` env > default
    ``../../models/ProPainter`` relative to this script.
    """
    if arg:
        return arg
    env = os.environ.get("VIDEOWIPE_PROPINTER_DIR")
    if env:
        return env
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "..", "models", "ProPainter",
        )
    )


def extract_frames(video_path, frames_dir):
    """Extract frames from video using imageio, return fps."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 24)
    for i, frame in enumerate(reader):
        # imageio reads RGB, ProPainter expects image folder read via cv2 (BGR)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.png"), bgr)
    reader.close()
    return fps


def frames_to_video(frames_dir, output_path, fps):
    """Combine frames back to video using imageio."""
    import glob
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = [imageio.imread(p) for p in paths]
    imageio.mimwrite(output_path, frames, fps=fps, quality=7)
    print(f"Encoded {len(frames)} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ProPainter as a VideoWipe external model adapter.",
    )
    parser.add_argument("video_path")
    parser.add_argument("mask_path")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--propainter-dir", default=None,
        help="Path to the ProPainter source checkout "
             "(or set VIDEOWIPE_PROPINTER_DIR)",
    )
    args = parser.parse_args()

    propainter_dir = _resolve_propainter_dir(args.propainter_dir)
    if not os.path.isfile(os.path.join(propainter_dir, "inference_propainter.py")):
        print(
            f"ERROR: ProPainter not found at {propainter_dir}. "
            f"Pass --propainter-dir or set VIDEOWIPE_PROPINTER_DIR.",
            file=sys.stderr,
        )
        sys.exit(1)

    video_path = os.path.abspath(args.video_path)
    mask_path = os.path.abspath(args.mask_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(video_path):
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(mask_path):
        print(f"ERROR: mask not found: {mask_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Extract frames to temp dir (workaround for torchvision.io.read_video removal)
    work = tempfile.mkdtemp(prefix="propainter_")
    try:
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        print(f"Extracting frames from {video_path} ...")
        fps = extract_frames(video_path, frames_dir)
        print(f"Extracted {len(os.listdir(frames_dir))} frames, fps={fps}")

        pp_tmp = os.path.join(work, "pp_output")
        os.makedirs(pp_tmp)

        cmd = [
            sys.executable,
            os.path.join(propainter_dir, "inference_propainter.py"),
            "-i", frames_dir,
            "-m", mask_path,
            "-o", pp_tmp,
            "--fp16",
            "--save_fps", str(int(fps)),
        ]
        print(f"Running ProPainter: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=propainter_dir, capture_output=True, text=True, check=False,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: ProPainter exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(1)

        # ProPainter saves to <pp_tmp>/<video_name>/inpaint_out.mp4 (fixed name).
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pp_output = os.path.join(pp_tmp, video_name, "inpaint_out.mp4")

        if os.path.isfile(pp_output):
            dest = os.path.join(output_dir, "inpaint_out.mp4")
            shutil.copy2(pp_output, dest)
            print(f"Output: {dest}")
        else:
            # ProPainter may have saved frames but not a video (imageio issue);
            # re-encode from the frames directory as a fallback.
            out_frames_dir = os.path.join(pp_tmp, video_name, "frames")
            if os.path.isdir(out_frames_dir):
                dest = os.path.join(output_dir, "inpaint_out.mp4")
                frames_to_video(out_frames_dir, dest, fps)
            else:
                print("ERROR: ProPainter output not found", file=sys.stderr)
                sys.exit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
