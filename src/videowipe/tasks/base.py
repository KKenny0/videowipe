"""Base class for inpainting tasks."""
import os
import sys
import time

import cv2
import torch

from videowipe.core.utils import Stack, ToTorchFormatTensor


def read_frame_info(video_path: str):
    """Read video metadata. Returns (VideoCapture, frame_info dict)."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        print(f"Failed to open video: {video_path}")
        sys.exit(1)
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
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return img[:, :, None]


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
        self.model = None
        self.device = None

    def load_model(self, weight_path: str, device: str = "auto"):
        """Load STTN model from a weight file."""
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from videowipe.models.sttn import InpaintGenerator

        self.model = InpaintGenerator().to(self.device)
        data = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(data["netG"])
        self.model.eval()
        print(f"Loaded weight: {weight_path}")

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

        reader, frame_info = read_frame_info(video_path)
        print(f"Video: {video_path} ({frame_info['len']} frames, {frame_info['fps']:.1f} fps)")

        mask = read_mask(mask_path)
        print(f"Mask: {mask_path}")

        os.makedirs(output_dir, exist_ok=True)

        out_path = self.process_video(reader, frame_info, mask, output_dir,
                                      video_path=video_path)
        print(f"Done in {time.time() - start:.1f}s -> {out_path}")
        return out_path

    def cleanup(self):
        """Release model and GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
