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

# Optional: OCR text recognition for better detection accuracy
pip install videowipe[ocr]
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

### Clean command

Use `task="clean"` for the full detection pipeline with target selection, intent parsing, and OCR:

```python
from videowipe import WipeEngine

engine = WipeEngine(task="clean", detect_mode="balanced", ocr="auto")
engine.process(
    video="input.mp4",
    targets=["subtitle", "watermark"],
    regions=["bottom"],
    intent="remove Chinese subtitles and logo watermark",
    output="result/",
)
engine.cleanup()
```

### Batch processing

Reuse the engine to avoid reloading the model:

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
| `--detect-mode` | Detection preset: `fast` (24 frames), `balanced` (50), `sensitive` (80) | `balanced` |
| `--ocr` | OCR text recognition: `auto`, `off`, `rapidocr` | `auto` |
| `--agent` | Local LLM CLI for intent-based selection (e.g., `claude`, `codex`) | — |
| `--external-command` | External inpainting command (bypasses built-in STTN) | — |
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
| `--external-command` | External inpainting command (bypasses built-in STTN) | — |

## External models

Pass `--external-command` to use any third-party inpainting model instead of the built-in STTN. The command receives `<video> <mask> <output_dir>` and must produce an output video in the output directory.

[ProPainter](https://github.com/sczhou/ProPainter) has been validated as a higher-quality alternative. A ready-to-use wrapper is included:

```bash
# Clone ProPainter outside this repo first
git clone https://github.com/sczhou/ProPainter.git ../models/ProPainter

# Use via the wrapper (requires CUDA PyTorch + fp16)
videowipe clean input.mp4 --external-command "python scripts/propainter_wipe.py"
```

> **Note:** ProPainter requires a GPU with ~16GB VRAM for 480p video and is licensed under NTU S-Lab License 1.0 (non-commercial).

<details>
<summary><strong>Quality comparison: ProPainter vs STTN</strong></summary>

Tested on a multilingual music video (Korean + Burmese subtitles, 852x480, 10s clip). Both models used the same mask.

| Original | ProPainter (GPU fp16) | STTN (CPU ONNX) |
|----------|----------------------|-----------------|
| <img src="pics/comparison/others_original.png" width="260"> | <img src="pics/comparison/others_propainter.png" width="260"> | <img src="pics/comparison/others_sttn.png" width="260"> |

Full evaluation details in `plans/candidate-eval-propainter.md`.

</details>

## Preview

### Subtitle removal

| Before | After |
|--------|-------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">Watch video</a></p>

### Auto-detection accuracy

Built-in detector locates text regions across multilingual content without manual masks:

<p float="left">
  <img src="pics/detection/chinese1_detected.jpg" width="32%">
  <img src="pics/detection/english1_detected.jpg" width="32%">
  <img src="pics/detection/others_detected.jpg" width="32%">
</p>

| Video | Candidates | Selected | Types |
|-------|-----------|----------|-------|
| Chinese drama | 4 | 2 | top subtitle, bottom subtitle |
| English clip | 2 | 2 | bottom subtitle |
| Music video (Korean + Burmese) | 7 | 5 | top watermark, bottom multilingual subtitles |

Tested with `--detect-mode balanced` (50 sampled frames). Green boxes show selected regions for inpainting.

## How it works

The model is an STTN (Spatial-Temporal Transformer Network) with 8 stacked transformer blocks operating on multi-scale patches. It encodes video frames with a CNN backbone, runs temporal attention across neighboring and reference frames, then decodes the inpainted result.

Key optimizations in this fork: AMP mixed-precision inference and `channels_last` memory layout. A 23-second test clip processes in 125s (down from 200s in the original).

## Docker

No Python? No problem. Run videowipe directly with Docker.

**CPU:**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe clean /data/input.mp4 -o /data/result/

# Legacy detext command
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe detext -v /data/input.mp4 -o /data/result/
```

**GPU (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu clean /data/input.mp4 -o /data/result/
```

Or use the included wrapper script (auto-detects GPU):

```bash
./scripts/docker-videowipe.sh detext -v input.mp4 -o result/
```

| Image | Size | GPU | Notes |
|-------|------|-----|-------|
| `videowipe:latest` | ~480 MB | No | CPU only, smallest image |
| `videowipe:gpu` | ~1.4 GB | Yes | ONNX Runtime with CUDA |

### Build from source

Use `--target` to select the image variant:

```bash
# CPU
docker build --target runtime-cpu -t videowipe:latest .

# GPU (requires NVIDIA Container Toolkit at build time for base image)
docker build --target runtime-gpu --build-arg VARIANT=gpu -t videowipe:gpu .
```

> **Note:** The GPU image requires a machine with NVIDIA runtime to verify CUDA execution. Without it, ONNX Runtime silently falls back to CPU.

Run after building:

```bash
# CPU
docker run --rm -v "$(pwd)":/data videowipe:latest detext -v /data/input.mp4 -o /data/result/

# GPU
docker run --rm --gpus all -v "$(pwd)":/data videowipe:gpu detext -v /data/input.mp4 -o /data/result/
```

## Credits

This project builds on [STTN](https://github.com/researchmm/STTN) and the original [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) implementation. The built-in text detection model is from [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR).

## License

MIT
