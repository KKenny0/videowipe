#!/usr/bin/env python3
"""Export STTN model from PyTorch .pth to three ONNX files.

Usage:
    python scripts/export_onnx.py --weight pretrained_weight/detext_trial.pth --output-dir weights_onnx/

Produces:
    - sttn_encoder.onnx
    - sttn_transformer.onnx
    - sttn_decoder.onnx

Requires: pip install torch onnx
"""
import argparse
import os

import numpy as np
import torch

# Allow running from project root without installing the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from videowipe.models.sttn import InpaintGenerator


def load_model(weight_path: str, device: str = "cpu") -> InpaintGenerator:
    """Load InpaintGenerator from a .pth checkpoint."""
    model = InpaintGenerator()
    data = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(data["netG"])
    model.eval()
    return model


def export_encoder(model, output_path: str, opset: int = 14):
    """Export encoder: (T, 3, H, W) → (T, C, H', W')."""
    dummy = torch.randn(1, 3, 120, 640)
    torch.onnx.export(
        model.encoder,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["frames"],
        output_names=["features"],
        dynamic_axes={
            "frames": {0: "T"},
            "features": {0: "T"},
        },
    )
    print(f"Exported encoder → {output_path}")


def _transformer_forward(self, feats):
    """Standalone forward for transformer export (b=1, c from input)."""
    c = feats.size(1)
    x = feats
    for block in self.transformer:
        x = block(x, b=1, c=c)
    return x


class _TransformerWrapper(torch.nn.Module):
    """Wrapper to export the transformer stack with a clean signature."""

    def __init__(self, model: InpaintGenerator):
        super().__init__()
        self.transformer = model.transformer

    def forward(self, feats):
        c = feats.size(1)
        x = feats
        for block in self.transformer:
            x = block(x, b=1, c=c)
        return x


def export_transformer(model, output_path: str, opset: int = 14):
    """Export transformer: (T, C, H', W') → (T, C, H', W')."""
    wrapper = _TransformerWrapper(model)
    dummy = torch.randn(10, 256, 30, 160)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["features"],
        output_names=["transformed"],
        dynamic_axes={
            "features": {0: "T"},
            "transformed": {0: "T"},
        },
    )
    print(f"Exported transformer → {output_path}")


def export_decoder(model, output_path: str, opset: int = 14):
    """Export decoder: (T, C, H', W') → (T, 3, H, W)."""
    dummy = torch.randn(1, 256, 30, 160)
    torch.onnx.export(
        model.decoder,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["features"],
        output_names=["frames"],
        dynamic_axes={
            "features": {0: "T"},
            "frames": {0: "T"},
        },
    )
    print(f"Exported decoder → {output_path}")


def validate(model, output_dir: str, device: str = "cpu"):
    """Quick numerical check: compare PyTorch vs ONNX outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping validation")
        return

    encoder_path = os.path.join(output_dir, "sttn_encoder.onnx")
    transformer_path = os.path.join(output_dir, "sttn_transformer.onnx")
    decoder_path = os.path.join(output_dir, "sttn_decoder.onnx")

    enc_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    trans_sess = ort.InferenceSession(transformer_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(3, 3, 120, 640).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        t = torch.from_numpy(dummy).to(device)
        pt_enc = model.encoder(t).cpu().numpy()
        pt_trans = model.infer(torch.from_numpy(pt_enc).to(device)).cpu().numpy()
        pt_dec = torch.tanh(model.decoder(torch.from_numpy(pt_trans).to(device))).cpu().numpy()

    # ONNX
    onnx_enc = enc_sess.run(None, {"frames": dummy})[0]
    onnx_trans = trans_sess.run(None, {"features": onnx_enc})[0]
    onnx_dec = np.tanh(dec_sess.run(None, {"features": onnx_trans})[0])

    def _mae(a, b):
        return np.abs(a - b).mean()

    print(f"\nValidation (MAE):")
    print(f"  encoder:     {_mae(pt_enc, onnx_enc):.6f}")
    print(f"  transformer: {_mae(pt_trans, onnx_trans):.6f}")
    print(f"  decoder:     {_mae(pt_dec, onnx_dec):.6f}")
    print(f"  end-to-end:  {_mae(pt_dec, onnx_dec):.6f}")

    if _mae(pt_dec, onnx_dec) < 0.01:
        print("  ✓ ONNX export is numerically consistent")
    else:
        print("  ⚠ MAE > 0.01 — verify the exported models")


def main():
    parser = argparse.ArgumentParser(description="Export STTN model to ONNX")
    parser.add_argument("--weight", required=True, help="Path to .pth weight file")
    parser.add_argument("--output-dir", default="weights_onnx", help="Output directory")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.weight}...")
    model = load_model(args.weight)

    export_encoder(model, os.path.join(args.output_dir, "sttn_encoder.onnx"), args.opset)
    export_transformer(model, os.path.join(args.output_dir, "sttn_transformer.onnx"), args.opset)
    export_decoder(model, os.path.join(args.output_dir, "sttn_decoder.onnx"), args.opset)

    if not args.skip_validate:
        validate(model, args.output_dir)

    print("\nDone. Upload the three .onnx files to GitHub Releases.")


if __name__ == "__main__":
    main()
