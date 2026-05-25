<h1 align="center">videowipe</h1>

<p align="center">
  Video inpainting library powered by STTN.<br>
  Remove hardcoded subtitles, watermarks, and text overlays. <code>pip install videowipe</code> and go.
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

---

## What it does

videowipe uses a Spatial-Temporal Transformer Network to erase hardcoded subtitles from video. You provide a video and a mask image marking the region to erase, or let the built-in detector generate one. The model fills in the background using temporal information from surrounding frames.

## Install

Requires Python 3.8+ and either ONNX Runtime or PyTorch.

```bash
# If you already have PyTorch:
pip install videowipe

# Lightweight ONNX Runtime backend:
pip install videowipe[onnx]

# Or the PyTorch backend:
pip install videowipe[torch]
```

Model weights download automatically on first run to `~/.videowipe/weights/`. No manual setup needed.

## Usage

### Python API

```python
from videowipe import remove_text

# Mask is optional — subtitle regions are auto-detected if omitted
remove_text(
    video="input.mp4",
    output="result/",
)

# Or provide your own mask for full control
remove_text(
    video="input.mp4",
    mask="mask.png",
    output="result/",
)
```

For batch processing, reuse the engine to avoid reloading the model:

```python
from videowipe import WipeEngine

engine = WipeEngine(task="detext")
engine.process(video="clip1.mp4", output="result/")
engine.process(video="clip2.mp4", mask="mask.png", output="result/")
engine.cleanup()
```

### CLI

```bash
# Auto-detect and remove all text overlays (recommended)
videowipe clean input.mp4 -o result/

# Legacy command: auto-detect subtitles only
videowipe detext -v input.mp4 -o result/

# With manual mask
videowipe detext -v input.mp4 -m mask.png -o result/
```

#### `clean` command options

```bash
# Only remove specific target types
videowipe clean input.mp4 --target subtitle
videowipe clean input.mp4 --target watermark

# Target a specific screen region
videowipe clean input.mp4 --region bottom
videowipe clean input.mp4 --region top-right

# Natural language intent
videowipe clean input.mp4 --intent "remove bottom Chinese subtitles"

# Preview detection results without processing
videowipe clean input.mp4 --preview -o result/

# Interactively confirm detected targets
videowipe clean input.mp4 --confirm
```

| Flag | Description | Default |
|------|-------------|---------|
| `--target` | Target type to clean (can repeat): `subtitle`, `timestamp`, `watermark`, `logo` | auto-detect all |
| `--region` | Screen region (can repeat): `top`, `bottom`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, `center` | all regions |
| `--intent` | Natural-language cleanup intent | — |
| `--preview` | Write detection artifacts only (no inpainting) | off |
| `--confirm` | Show detected targets and confirm before processing | off |
| `--agent` | Local LLM CLI for intent-based selection (e.g., `claude`, `codex`) | — |
| `-g, --gap` | Segment length per pass; higher = better quality, slower | `200` |
| `-d, --dual` | Show original video side-by-side in output | off |

#### `detext` command arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --video` | Input video path | required |
| `-m, --mask` | Mask image path (auto-detect if omitted) | auto |
| `-o, --output` | Output directory | `result/` |
| `-w, --weight` | Model weight path. PyTorch accepts `.pth`/`.pt`; ONNX expects a prefix path ending in `.onnx` with matching `_encoder`, `_transformer`, and `_decoder` files. | auto |
| `-g, --gap` | Segment length per pass; higher = better quality, slower | `200` |
| `-d, --dual` | Show original video side-by-side in output | off |

## Preview

### Subtitle removal

| Before | After |
|--------|-------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">Watch video</a></p>

## How it works

The model is an STTN (Spatial-Temporal Transformer Network) with 8 stacked transformer blocks operating on multi-scale patches. It encodes video frames with a CNN backbone, runs temporal attention across neighboring and reference frames, then decodes the inpainted result.

Key optimizations in this fork: AMP mixed-precision inference and `channels_last` memory layout. A 23-second test clip processes in 125s (down from 200s in the original).

## Docker

No Python? No problem. Run videowipe directly with Docker.

**CPU:**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe detext -v /data/input.mp4 -o /data/result/
```

**GPU (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu detext -v /data/input.mp4 -o /data/result/
```

Or use the included wrapper script (auto-detects GPU):

```bash
./scripts/docker-videowipe.sh detext -v input.mp4 -o result/
```

| Image | Size | GPU | Notes |
|-------|------|-----|-------|
| `videowipe:latest` | ~480 MB | No | CPU only, smallest image |
| `videowipe:gpu` | ~1.4 GB | Yes | ONNX Runtime with CUDA |

## Credits

This project builds on [STTN](https://github.com/researchmm/STTN) and the original [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) implementation. The built-in text detection model is from [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR).

## License

MIT
