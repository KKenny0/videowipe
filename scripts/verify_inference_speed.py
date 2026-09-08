"""Compare fixed three-second trials against the pre-optimization STTN loop.

Requires local Torch weights and an MPS-capable execution environment. Writes
raw frames for numerical comparison and videos for visual review under result/.
This measures trial performance, not full-video speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob
from videowipe.tasks.base import read_mask
from videowipe.plan import (
    MaskAsset, Segment, TemporalResolution, Track, WipePlan, compute_source, execution_masks,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_temporal(baseline, root, output, weight):
    """Same-device old/new comparison, including dense-mask overhead."""
    video = root / "input/detext_examples/english1.mp4"
    source = compute_source(str(video))
    mask = read_mask(str(video.parent / "mask/english1_mask.png"))[:, :, 0]
    ys, xs = np.where(mask)
    plan = WipePlan(
        kind="wipe_plan", schema_version=1, source=source, request={},
        temporal_resolution=TemporalResolution(25, 1.0, 12),
        mask_asset=MaskAsset("wipe_plan_masks.npz", ""),
        tracks=[Track(
            id="subtitle", type="subtitle", label="subtitle", action="remove",
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            confidence=1.0, presence_fraction=1.0, decision_reason="fixed speed fixture",
            segments=[Segment(125, 200)], mask_key="subtitle", mask=mask,
        )],
    )
    evidence = {}
    for label, interval in (("sparse", (151, 174)), ("dense", (125, 200))):
        plan.tracks[0].segments = [Segment(*interval)]
        evidence[label] = {}
        for version, module in (("before", baseline), ("after", sttn)):
            painter = module.STTNInpainter()
            painter.load(str(weight), device="mps")
            timings = []
            original_blend, original_segment = module._blend_frame_regions, module._process_segment
            digest = hashlib.sha256()
            calls = 0

            def capture(*args):
                frame = original_blend(*args)
                digest.update(frame.tobytes())
                return frame

            def count_segment(*args):
                nonlocal calls
                calls += 1
                return original_segment(*args)

            try:
                for repeat in range(4):
                    if repeat == 0:
                        module._blend_frame_regions = capture
                        module._process_segment = count_segment
                    static, temporal = execution_masks(plan, feather_radius=4)
                    reader = cv2.VideoCapture(str(video))
                    folder = output / "temporal" / label / version / str(repeat)
                    folder.mkdir(parents=True, exist_ok=True)
                    start = time.perf_counter()
                    try:
                        painter.inpaint(InpaintJob(
                            video_path=str(video), reader=reader, mask=static,
                            frame_mask=temporal, output_dir=str(folder), width=source.width,
                            height=source.height, fps=source.fps, frame_count=source.frame_count,
                            trial_range=(125, 200), gap=25,
                        ))
                        elapsed = time.perf_counter() - start
                    finally:
                        reader.release()
                    module._blend_frame_regions, module._process_segment = original_blend, original_segment
                    if repeat:
                        timings.append(elapsed)
                evidence[label][version] = {
                    "seconds": timings, "raw_sha256": digest.hexdigest(), "segment_calls": calls,
                }
            finally:
                module._blend_frame_regions, module._process_segment = original_blend, original_segment
                painter.cleanup()
        before, after = evidence[label]["before"], evidence[label]["after"]
        assert before["raw_sha256"] == after["raw_sha256"]
        evidence[label]["speedup"] = statistics.median(before["seconds"]) / statistics.median(after["seconds"])
    assert evidence["sparse"]["after"]["segment_calls"] * 3 == evidence["sparse"]["before"]["segment_calls"]
    assert evidence["dense"]["speedup"] >= 1 / 1.05
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("result/inference-speed"))
    parser.add_argument("--baseline-ref", default="a7a09b1")
    parser.add_argument("--weight", type=Path, default=Path.home() / ".videowipe/weights/detext_trial.pth")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable; run in an MPS-capable macOS environment")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_code = subprocess.check_output([
        "git", "show", f"{args.baseline_ref}:src/videowipe/inpainters/sttn.py",
    ], cwd=root)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "baseline.py"
        path.write_bytes(baseline_code)
        spec = importlib.util.spec_from_file_location("speed_baseline", path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)

    report = {
        "scope": "fixed three-second trials; raw pre-encoding quality; visual review required",
        "baseline_ref": args.baseline_ref,
        "baseline_loop_sha256": hashlib.sha256(baseline_code).hexdigest(),
        "current_loop_sha256": sha256(Path(sttn.__file__)),
        "script_sha256": sha256(__file__),
        "plan_sha256": sha256(root / "src/videowipe/plan.py"),
        "backend_sha256": sha256(root / "src/videowipe/backends.py"),
        "weight_sha256": sha256(args.weight), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "gap": 25, "samples": {},
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    for device, module in (("cpu", baseline), ("mps", sttn)):
        painter = module.STTNInpainter()
        start = time.perf_counter()
        painter.load(str(args.weight), device=device)
        report[f"{device}_load_s"] = time.perf_counter() - start
        try:
            # One 25-frame warmup per device; identical fixed input shape.
            module._process_segment([np.zeros((120, 640, 3), np.float32)] * 25,
                                    painter.backend, 640, 120, 5, 5)
            for name, seconds in (("chinese1", 5.3), ("english1", 5.3), ("others", 2.3)):
                video = root / "input/detext_examples" / f"{name}.mp4"
                mask_path = video.parent / "mask" / f"{name}_mask.png"
                mask = read_mask(str(mask_path))
                cap = cv2.VideoCapture(str(video))
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width, height = int(cap.get(3)), int(cap.get(4))
                cap.release()
                first = int(seconds * fps)
                last = min(count, first + round(3 * fps))
                sample = report["samples"].setdefault(name, {
                    "source_sha256": sha256(video), "mask_sha256": sha256(mask_path),
                    "range": [first, last], "fps": fps,
                })
                runs = []
                for repeat in range(4):
                    folder = output / name / device / str(repeat)
                    folder.mkdir(parents=True, exist_ok=True)
                    raw = None
                    original = module._blend_frame_regions
                    index = 0
                    if repeat == 0:
                        raw = np.lib.format.open_memmap(
                            output / f"{name}-{device}.npy", mode="w+", dtype=np.uint8,
                            shape=(last - first, height, width, 3),
                        )

                        def capture(*values):
                            nonlocal index
                            frame = original(*values)
                            raw[index] = frame
                            index += 1
                            return frame

                        module._blend_frame_regions = capture
                    reader = cv2.VideoCapture(str(video))
                    metrics = {}
                    started = time.perf_counter()
                    try:
                        result = painter.inpaint(InpaintJob(
                            video_path=str(video), reader=reader, mask=mask,
                            output_dir=str(folder), width=width, height=height,
                            fps=fps, frame_count=count, gap=25, trial_range=(first, last),
                            dual=True, metrics=metrics,
                        ))
                        elapsed = time.perf_counter() - started
                    finally:
                        reader.release()
                        module._blend_frame_regions = original
                    if raw is not None:
                        assert index == last - first
                        raw.flush()
                        del raw
                    check = cv2.VideoCapture(result.output_path)
                    assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == last - first
                    assert int(check.get(3)) == width and int(check.get(4)) == height * 2
                    check.release()
                    if repeat == 0:
                        sample[f"{device}_quality_capture_s"] = elapsed
                    else:
                        runs.append({"wall_s": elapsed, **metrics})
                    sample[device] = runs
                    save()
                    print(name, device, repeat, {"wall_s": elapsed, **metrics}, flush=True)
        finally:
            painter.cleanup()

    for name, sample in report["samples"].items():
        cpu = np.load(output / f"{name}-cpu.npy", mmap_mode="r")
        mps = np.load(output / f"{name}-mps.npy", mmap_mode="r")
        mask = read_mask(str(root / "input/detext_examples/mask" / f"{name}_mask.png"))
        active = np.any(mask != 0, axis=2)
        error_sum = temporal_sum = outside = 0
        previous = None
        for index in range(len(cpu)):
            delta = mps[index].astype(np.int16) - cpu[index].astype(np.int16)
            error_sum += int(np.abs(delta[active]).sum())
            outside += int(np.count_nonzero(delta[~active]))
            if previous is not None:
                temporal_sum += int(np.abs(delta[active] - previous).sum())
            previous = delta[active]
        channels = int(active.sum()) * 3
        sample["quality"] = {
            "removed_mae": error_sum / (len(cpu) * channels),
            "outside_changed_channels": outside,
            "temporal_extra_mae": temporal_sum / ((len(cpu) - 1) * channels),
        }
        sample["speedup"] = {
            key: statistics.median(run[key] for run in sample["cpu"]) /
                 statistics.median(run[key] for run in sample["mps"])
            for key in ("wall_s", "inpainting_s")
        }
        # CPU/MPS cleaned crops, one row per timestamp, for actual bitmap review.
        rows = []
        ys, xs = np.where(active)
        top, bottom = max(0, int(ys.min()) - 20), min(cpu.shape[1], int(ys.max()) + 21)
        for index in (0, len(cpu) // 2, len(cpu) - 1):
            pair = np.hstack((cpu[index, top:bottom], mps[index, top:bottom]))
            rows.append(cv2.resize(pair, (1280, max(1, round(pair.shape[0] * 1280 / pair.shape[1])))))
        if not cv2.imwrite(str(output / f"{name}-comparison.png"), np.vstack(rows)):
            raise OSError(f"Could not write comparison image for {name}")
        save()
        assert sample["quality"]["removed_mae"] <= 0.5
        assert sample["quality"]["temporal_extra_mae"] <= 0.5
        assert outside == 0
        assert sample["speedup"]["inpainting_s"] >= 2
        assert sample["speedup"]["wall_s"] >= 1
    assert sum(s["speedup"]["wall_s"] >= 2 for s in report["samples"].values()) >= 2
    report["temporal"] = verify_temporal(baseline, root, output, args.weight)
    save()
    print("Trial numerical/performance gates passed; review comparison images and videos.", flush=True)


if __name__ == "__main__":
    main()
