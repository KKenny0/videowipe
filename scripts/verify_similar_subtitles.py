"""Synthetic similar-scene material: yellow-white mixed glyphs with a black
outline over a moving background, plus adjacent empty frames.

The material is SYNTHETIC and provenance-tracked here: the background is a
losslessly panned band cropped from ``input/detext_examples/english1.mp4``
frames and the glyphs are drawn with OpenCV Hershey strokes. It exercises the
subtitle-body, outline, adjacent-empty, and moving-background cases of the
similar-scene ticket; it is not a real user case and never substitutes for one.

Run: PYTHONPATH=src python scripts/verify_similar_subtitles.py \
        --output result/spec-3-acceptance/synth
Outputs the generated video, the rendered result, per-frame scores, and a
band montage for visual playback sign-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from videowipe import WipeEngine, WipeRequest

WIDTH, HEIGHT, FPS = 1280, 720, 25
EMPTY_HEAD, PRESENT, EMPTY_TAIL = 50, 80, 50  # frames: 0-49 / 50-129 / 130-179
LINE_1 = "I need your love"
LINE_2 = "Yellow White Mixed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _background_frames() -> list[np.ndarray]:
    """Pan a real sample band so the backdrop moves like handheld footage."""
    cap = cv2.VideoCapture("input/detext_examples/english1.mp4")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read input/detext_examples/english1.mp4")
    band = frame[60:180, :]  # a text-free upper band with real texture
    band = cv2.resize(band, (WIDTH * 2, HEIGHT), interpolation=cv2.INTER_CUBIC)
    frames = []
    for index in range(EMPTY_HEAD + PRESENT + EMPTY_TAIL):
        offset = (index * 6) % (WIDTH + 320)
        tile = band[:, offset:offset + WIDTH].copy()
        if tile.shape[1] < WIDTH:  # wrap the pan
            tile = np.hstack([tile, band[:, :WIDTH - tile.shape[1]]])
        frames.append(tile)
    return frames


def _draw_subtitle(frame: np.ndarray, glyph: np.ndarray) -> None:
    """Mixed yellow/white glyphs with a black outline; records the glyph mask."""
    for text, y in ((LINE_1, 590), (LINE_2, 650)):
        advances = [cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0][0]
                    for char in text]
        cursor = (WIDTH - sum(advances)) // 2
        stencil = np.zeros(frame.shape[:2], np.uint8)
        for position, (char, advance) in enumerate(zip(text, advances)):
            color = (255, 255, 255) if position % 2 == 0 else (0, 255, 255)
            for target, value, thickness in ((frame, (0, 0, 0), 9),
                                              (frame, color, 3),
                                              (stencil, 255, 9),
                                              (stencil, 255, 3)):
                cv2.putText(target, char, (cursor, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, value, thickness, cv2.LINE_AA)
            cursor += advance
        glyph |= stencil > 0


def generate(out: Path) -> tuple[Path, Path, dict]:
    """Write source video + per-frame truth; return paths and provenance."""
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "source-frames"
    frames_dir.mkdir(exist_ok=True)
    frames = _background_frames()
    glyph = np.zeros((len(frames), HEIGHT, WIDTH), bool)
    for index, background in enumerate(frames):
        frame = background.copy()
        if EMPTY_HEAD <= index < EMPTY_HEAD + PRESENT:
            _draw_subtitle(frame, glyph[index])
        cv2.imwrite(str(frames_dir / f"{index:06d}.png"), frame)
    video = out / "similar_scene.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", str(frames_dir / "%06d.png"), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    truth_dir = out / "truth-frames"
    truth_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):  # exact background-only truth
        cv2.imwrite(str(truth_dir / f"{index:06d}.png"), frame)
    np.savez_compressed(out / "glyph_masks.npz", glyph=glyph)
    provenance = {
        "kind": "synthetic",
        "background_source": "input/detext_examples/english1.mp4 (band 60:180, panned +6px/frame, 2x upscale)",
        "subtitle_style": "per-character alternating white(BGR 255,255,255)/yellow(BGR 0,255,255), 9px black outline, Hershey SIMPLEX 1.5",
        "layout": f"{EMPTY_HEAD} empty / {PRESENT} subtitle / {EMPTY_TAIL} empty frames @ {FPS}fps {WIDTH}x{HEIGHT}",
        "encoder": "ffmpeg libx264 crf16 yuv420p",
        "source_sha256": _sha256(Path("input/detext_examples/english1.mp4")),
        "video_sha256": _sha256(video),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return video, truth_dir, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result/spec-3-acceptance/synth")
    args = parser.parse_args()
    out = Path(args.output)
    video, truth_dir, provenance = generate(out)

    engine = WipeEngine(task="clean")
    request = WipeRequest(
        video=str(video), output_dir=str(out / "run"), targets=["subtitle"],
    )
    result = engine.run(request, on_progress=lambda event: print(
        f"{event.phase} {event.completed}/{event.total}", flush=True))
    plan_path = Path(result.plan_path) if getattr(result, "plan_path", None) else None

    masks = np.load(out / "glyph_masks.npz")["glyph"]
    result_video = cv2.VideoCapture(result.output_path)
    scores = {"glyph_mae": [], "background_mae": [], "empty_frame_equal": [],
              "empty_frame_maxdiff": []}
    band_rows = []
    for index in range(EMPTY_HEAD + PRESENT + EMPTY_TAIL):
        ok, rendered = result_video.read()
        assert ok, f"result video ended at frame {index}"
        truth = cv2.imread(str(truth_dir / f"{index:06d}.png"))
        glyph = masks[index]
        if glyph.any():
            error = np.abs(rendered.astype(np.float32) - truth.astype(np.float32))
            dilated = cv2.dilate(glyph.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            scores["glyph_mae"].append(round(float(error[dilated].mean()), 3))
            background = ~dilated
            scores["background_mae"].append(round(float(error[background].mean()), 3))
            if index % 10 == 0 or index in (EMPTY_HEAD, EMPTY_HEAD + PRESENT - 1):
                pair = np.hstack([truth[540:720, 240:1040], rendered[540:720, 240:1040]])
                cv2.putText(pair, f"f{index} truth|out", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                band_rows.append(pair)
        else:
            equal = np.array_equal(rendered, truth)
            maxdiff = int(np.abs(rendered.astype(np.int16) - truth.astype(np.int16)).max())
            scores["empty_frame_equal"].append(equal)
            scores["empty_frame_maxdiff"].append(maxdiff)
    report = {
        "provenance": provenance,
        "plan": str(plan_path),
        "warnings": list(result.warnings),
        "glyph_mae_mean": round(float(np.mean(scores["glyph_mae"])), 3),
        "background_mae_mean": round(float(np.mean(scores["background_mae"])), 3),
        "empty_frames_total": len(scores["empty_frame_equal"]),
        "empty_frames_bitwise_equal": int(sum(scores["empty_frame_equal"])),
        "empty_frame_maxdiff_worst": int(max(scores["empty_frame_maxdiff"])),
        "glyph_mae_per_checked_frame": scores["glyph_mae"],
        "note": "SYNTHETIC material; scores cover this generation only.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    if band_rows:
        cv2.imwrite(str(out / "band_montage.png"), np.vstack(band_rows))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("glyph_mae_per_checked_frame",)}, indent=1))


if __name__ == "__main__":
    main()
