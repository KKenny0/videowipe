"""Base class for inpainting tasks."""
import os
import time

import cv2

from videowipe.backends import load_backend


def read_frame_info(video_path: str):
    """Read video metadata. Returns (VideoCapture, frame_info dict)."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        "W_ori": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
        "H_ori": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
        "fps": reader.get(cv2.CAP_PROP_FPS),
        "len": int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5),
    }
    return reader, info


def read_mask(path: str):
    """Read and binarize a mask image. Returns ndarray with shape (H, W, 1), values in {0, 1}."""
    img = cv2.imread(path, 0)
    if img is None:
        raise ValueError(f"Cannot read mask image: {path}")
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return img[:, :, None]


def validate_mask_shape(mask, frame_info: dict) -> None:
    """Validate that a mask has the same spatial size as the video."""
    expected = (frame_info["H_ori"], frame_info["W_ori"])
    actual = mask.shape[:2]
    if actual != expected:
        raise ValueError(
            f"Mask shape {actual[1]}x{actual[0]} does not match video "
            f"shape {expected[1]}x{expected[0]}"
        )


class BaseTask:
    """Base class for video inpainting tasks."""

    def __init__(
        self,
        gap: int = 200,
        ref_length: int = 5,
        neighbor_stride: int = 5,
        dual: bool = False,
    ):
        self.gap = gap
        self.ref_length = ref_length
        self.neighbor_stride = neighbor_stride
        self.dual = dual
        self.backend = None
        self._bm = None

    def load_model(self, weight_path: str, device: str = "auto"):
        """Load model from a weight file (.pth or .onnx)."""
        self.backend = load_backend(weight_path, device=device)
        print(f"Loaded weight: {weight_path} (backend: {type(self.backend).__name__})")

    def process_video(self, reader, frame_info, mask, output_dir: str,
                      video_path: str = "") -> str:
        """Process video. Subclasses must implement this.

        Returns the output file path.
        """
        raise NotImplementedError

    def run(self, video_path: str, mask_path: str, weight_path: str,
            output_dir: str) -> str:
        """Full pipeline: load model, read inputs, process, return output path."""
        start = time.time()
        self.load_model(weight_path)

        reader = None
        try:
            reader, frame_info = read_frame_info(video_path)
            print(f"Video: {video_path} ({frame_info['len']} frames, {frame_info['fps']:.1f} fps)")

            mask = read_mask(mask_path)
            validate_mask_shape(mask, frame_info)
            print(f"Mask: {mask_path}")

            os.makedirs(output_dir, exist_ok=True)

            out_path = self.process_video(reader, frame_info, mask, output_dir,
                                          video_path=video_path)
        finally:
            if reader is not None:
                reader.release()
            self.cleanup()
        print(f"Done in {time.time() - start:.1f}s -> {out_path}")
        return out_path

    def cleanup(self):
        """Release model and GPU memory."""
        if self.backend is not None:
            self.backend.cleanup()
            self.backend = None
