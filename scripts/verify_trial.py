"""Generate fixed real-model trial evidence; visual quality remains human-reviewed.

Run: PYTHONPATH=src python scripts/verify_trial.py --output result/trial-acceptance
Uses the existing three golden masks, not automatic detection. No models or
videos are downloaded by this script beyond the engine's normal weight loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from videowipe import WipeEngine, WipeRequest
from videowipe.plan import compute_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("result/trial-acceptance"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "STTN trials with existing golden masks; not detector quality",
              "visual_quality": "requires human review", "samples": []}
    with WipeEngine(task="clean") as engine:
        for name, start_seconds in (("chinese1", 5.3), ("english1", 5.3), ("others", 2.3)):
            video = root / "input/detext_examples" / f"{name}.mp4"
            mask = video.parent / "mask" / f"{name}_mask.png"
            source = compute_source(str(video))
            first = int(start_seconds * source.fps)
            last = min(source.frame_count, first + round(3 * source.fps))
            result = engine.run(WipeRequest(
                video=video, mask=mask, output_dir=args.output / name,
                trial_range=(first, last),
            ))
            cap = cv2.VideoCapture(result.output_path)
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                assert count == last - first, (name, count, last - first)
                assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == source.width
                assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 2 * source.height
                tiles = []
                for index in (0, count // 2, count - 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    assert ok, (name, index)
                    tiles.append(cv2.resize(frame, (480, round(960 * source.height / source.width))))
                contact = args.output / name / "contact.jpg"
                assert cv2.imwrite(str(contact), np.hstack(tiles))
            finally:
                cap.release()
            report["samples"].append({
                "name": name, "source": source.to_dict(),
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "trial_range": [first, last], "result": result.to_dict(),
                "contact": str(contact), "output_frame_count": count,
            })
            (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{name}: trial validated, review {contact}", flush=True)


if __name__ == "__main__":
    main()
