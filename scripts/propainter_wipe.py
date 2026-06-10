"""Wrapper to run ProPainter as a VideoWipe external model adapter.

Usage (called by VideoWipe --external-command):
    python scripts/propainter_wipe.py <video_path> <mask_path> <output_dir>

ProPainter saves to <output>/<video_name>/inpaint_out.mp4.
This wrapper extracts frames (workaround for torchvision.io.read_video removal),
runs ProPainter, and moves the result to <output_dir>.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import imageio

PROPINTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "..", "models", "ProPainter",
)
PROPINTER_DIR = os.path.normpath(PROPINTER_DIR)

if not os.path.isfile(os.path.join(PROPINTER_DIR, "inference_propainter.py")):
    print(f"ERROR: ProPainter not found at {PROPINTER_DIR}", file=sys.stderr)
    sys.exit(1)


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
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = [imageio.imread(p) for p in paths]
    imageio.mimwrite(output_path, frames, fps=fps, quality=7)
    print(f"Encoded {len(frames)} frames -> {output_path}")


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <video_path> <mask_path> <output_dir>", file=sys.stderr)
        sys.exit(1)

    video_path = os.path.abspath(sys.argv[1])
    mask_path = os.path.abspath(sys.argv[2])
    output_dir = os.path.abspath(sys.argv[3])

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
            os.path.join(PROPINTER_DIR, "inference_propainter.py"),
            "-i", frames_dir,
            "-m", mask_path,
            "-o", pp_tmp,
            "--fp16",
            "--save_fps", str(int(fps)),
        ]
        print(f"Running ProPainter: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=PROPINTER_DIR, capture_output=True, text=True, check=False,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: ProPainter exited with code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(1)

        # ProPainter saves to <pp_tmp>/<video_name>/inpaint_out.mp4
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pp_output = os.path.join(pp_tmp, video_name, "inpaint_out.mp4")

        if not os.path.isfile(pp_output):
            # Search for any mp4
            for root, dirs, files in os.walk(pp_tmp):
                for f in files:
                    if f.endswith(".mp4"):
                        pp_output = os.path.join(root, f)
                        break

        if os.path.isfile(pp_output):
            dest = os.path.join(output_dir, "inpaint_out.mp4")
            shutil.copy2(pp_output, dest)
            print(f"Output: {dest}")
        else:
            # ProPainter may have saved frames but not video (imageio issue)
            # Try to find output frames and encode ourselves
            out_frames_dir = os.path.join(pp_tmp, video_name, "frames")
            if os.path.isdir(out_frames_dir):
                dest = os.path.join(output_dir, "inpaint_out.mp4")
                frames_to_video(out_frames_dir, dest, fps)
            else:
                print(f"ERROR: ProPainter output not found", file=sys.stderr)
                sys.exit(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
