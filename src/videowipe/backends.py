"""Inference backends: PyTorch or ONNX Runtime."""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np


def _detect_backend(weight_path: str) -> str:
    """Detect backend from weight file extension."""
    if weight_path.endswith(".onnx"):
        return "onnx"
    if weight_path.endswith(".pth") or weight_path.endswith(".pt"):
        return "torch"
    raise ValueError(
        f"Unsupported weight file: {weight_path}. Expected .onnx, .pth, or .pt."
    )


def load_backend(
    weight_path: str,
    device: str = "auto",
) -> "InpaintBackend":
    """Load the appropriate backend based on weight file type.

    If weight_path ends with .onnx, uses ONNX Runtime.
    Otherwise, uses PyTorch (requires torch installed).
    """
    backend = _detect_backend(weight_path)
    if backend == "onnx":
        return ONNXBackend(weight_path)
    return TorchBackend(weight_path, device=device)


class InpaintBackend:
    """Unified interface for model inference.

    All methods take and return numpy arrays so that the
    calling code (task layer) stays backend-agnostic.
    """

    def preprocess(self, frames: List[np.ndarray]) -> np.ndarray:
        """Convert list of (H, W, 3) BGR uint8 frames to (T, 3, H, W) float32 in [-1, 1]."""
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        arr = np.stack(rgb, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        arr = arr / 127.5 - 1.0
        return arr

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        """(T, 3, H, W) → (T, C, H', W')"""
        raise NotImplementedError

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """(T, C, H', W') → (T, C, H', W')"""
        raise NotImplementedError

    def decode(self, feats: np.ndarray) -> np.ndarray:
        """(T, C, H', W') → (T, H, W, 3) uint8 in [0, 255]."""
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class TorchBackend(InpaintBackend):
    """PyTorch model backend. Requires torch and torchvision."""

    def __init__(self, weight_path: str, device: str = "auto"):
        import torch
        from videowipe.models.sttn import InpaintGenerator

        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = InpaintGenerator().to(self.device)
        data = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data["netG"])
        self.model.eval()
        self._torch = torch

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(tensor).to(self.device)
            if t.is_cuda:
                t = t.to(memory_format=torch.channels_last)
            feats = self.model.encoder(t)
        return feats.cpu().numpy()

    def transform(self, feats: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(feats).to(self.device)
            if t.is_cuda:
                t = t.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=t.is_cuda):
                result = self.model.infer(t)
        return result.cpu().numpy()

    def decode(self, feats: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(feats).to(self.device)
            decoded = self.model.decoder(t)
            pred_img = torch.tanh(decoded)
            pred_img = ((pred_img + 1.0) * 127.5).clamp(0, 255)
            pred_img = pred_img.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return pred_img.astype(np.uint8)

    def cleanup(self) -> None:
        torch = self._torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()


class ONNXBackend(InpaintBackend):
    """ONNX Runtime backend. Requires onnxruntime.

    Expects three ONNX model files:
      - <base>_encoder.onnx
      - <base>_transformer.onnx
      - <base>_decoder.onnx
    where <base> is weight_path with .onnx stripped.
    """

    def __init__(self, weight_path: str):
        import onnxruntime as ort

        base = weight_path[: -len(".onnx")]
        required = [
            f"{base}_encoder.onnx",
            f"{base}_transformer.onnx",
            f"{base}_decoder.onnx",
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise ValueError(
                "ONNX backend expects a prefix path ending in .onnx. Missing files: "
                + ", ".join(missing)
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.encoder_session = ort.InferenceSession(
            f"{base}_encoder.onnx", providers=providers
        )
        self.transformer_session = ort.InferenceSession(
            f"{base}_transformer.onnx", providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            f"{base}_decoder.onnx", providers=providers
        )

        self._enc_name = self.encoder_session.get_inputs()[0].name
        self._trans_name = self.transformer_session.get_inputs()[0].name
        self._dec_name = self.decoder_session.get_inputs()[0].name

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        return self.encoder_session.run(None, {self._enc_name: tensor})[0]

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return self.transformer_session.run(None, {self._trans_name: feats})[0]

    def decode(self, feats: np.ndarray) -> np.ndarray:
        raw = self.decoder_session.run(None, {self._dec_name: feats})[0]
        raw = np.tanh(raw)
        raw = ((raw + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return raw.transpose(0, 2, 3, 1)
