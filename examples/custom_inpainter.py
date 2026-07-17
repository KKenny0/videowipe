"""Register a small third-party Inpainter and run it through WipeEngine."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from videowipe import (
    InpaintJob,
    InpaintOutcome,
    WipeEngine,
    WipeRequest,
    get_registry,
    register_inpainter,
)


MODEL_NAME = "copy-example"


class CopyInpainter:
    """Example adapter that copies the input instead of running a model."""

    name = MODEL_NAME

    def load(self, weight_path: str | None, device: str = "auto") -> None:
        self.weight_path = weight_path
        self.device = device

    def inpaint(self, job: InpaintJob) -> InpaintOutcome:
        if not job.mask_path:
            raise ValueError("CopyInpainter requires a mask file path")
        destination = Path(job.output_dir) / f"{Path(job.video_path).stem}_copy.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job.video_path, destination)
        return InpaintOutcome(str(destination), backend=self.name)

    def cleanup(self) -> None:
        pass


def register_example() -> None:
    if MODEL_NAME not in get_registry().names():
        register_inpainter(MODEL_NAME, CopyInpainter)


def build_engine() -> WipeEngine:
    register_example()
    return WipeEngine(model=MODEL_NAME)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--output", default=Path("result"), type=Path)
    args = parser.parse_args(argv)

    with build_engine() as engine:
        result = engine.run(WipeRequest(
            video=args.video,
            mask=args.mask,
            output_dir=args.output,
        ))
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
