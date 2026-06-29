<p align="center">
  <img src="assets/logo.svg" width="80" height="80" alt="videowipe">
</p>

<h1 align="center">videowipe</h1>

<p align="center">
  Remove hardcoded subtitles, watermarks, and text overlays from video.<br>
  Auto-detect targets, generate masks, and inpaint locally.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

---

## What it does

videowipe detects and removes hardcoded text, watermarks, logos, and timestamps from video. A full pipeline runs in one command: sample frames → detect text regions → select targets (with optional OCR and natural-language intent parsing) → generate masks → inpaint the background.

No manual mask required. The built-in detector handles multilingual content out of the box.

STTN is the default inpainting backend. Any external model can be plugged in via `--external-command` — [ProPainter](https://github.com/sczhou/ProPainter) has been validated as a higher-quality alternative.

## Install

Requires Python 3.8+ and either ONNX Runtime or PyTorch.
VideoWipe is not published to PyPI yet; install it from source:

```bash
git clone https://github.com/KKenny0/videowipe.git
cd videowipe

# Local web UI with the lightweight ONNX Runtime backend:
pip install -e ".[web,onnx]"

# Optional extras:
pip install -e ".[torch]"  # PyTorch backend
pip install -e ".[ocr]"    # OCR text recognition
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

### Full pipeline with target selection

Use `task="clean"` for the complete detection pipeline with target selection, intent parsing, and OCR:

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

# With manual mask
videowipe clean input.mp4 -m mask.png -o result/
```

### Local web UI

```bash
pip install -e ".[web,onnx]"
videowipe serve
# Open http://127.0.0.1:8000
```

The browser flow runs locally: upload a video, review detected targets, confirm, and download the cleaned MP4.
Files stay on your machine, the preview step shows the detected regions before inpainting, and the downloaded MP4 keeps the original audio track.

| Upload | Preview targets | Download |
|--------|-----------------|----------|
| <img src="pics/web-ui/01-upload.png" width="260" alt="VideoWipe web upload screen"> | <img src="pics/web-ui/02-preview.png" width="260" alt="VideoWipe web target preview screen"> | <img src="pics/web-ui/03-download.png" width="260" alt="VideoWipe web download screen"> |

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
| `-m, --mask` | Mask image path (auto-detect if omitted) | auto |

<details>
<summary><strong>Legacy: detext command</strong></summary>

The `detext` command auto-detects subtitles only. Prefer `clean` for new usage.

```bash
# Auto-detect subtitles
videowipe detext -v input.mp4 -o result/

# With manual mask
videowipe detext -v input.mp4 -m mask.png -o result/
```

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --video` | Input video path | required |
| `-m, --mask` | Mask image path (auto-detect if omitted) | auto |
| `-o, --output` | Output directory | `result/` |
| `-w, --weight` | Model weight path. PyTorch accepts `.pth`/`.pt`; ONNX expects a prefix path ending in `.onnx` with matching `_encoder`, `_transformer`, and `_decoder` files. | auto |
| `-g, --gap` | Segment length per pass; higher = better quality, slower | `200` |
| `-d, --dual` | Show original video side-by-side in output | off |
| `--external-command` | External inpainting command (bypasses built-in STTN) | — |

</details>

## External models

Pass `--external-command` to use any third-party inpainting model instead of the built-in STTN. The command receives `<video> <mask> <output_dir>` and must produce an output video in the output directory.

[ProPainter](https://github.com/sczhou/ProPainter) has been validated as a higher-quality alternative. A ready-to-use wrapper is included:

```bash
# Clone ProPainter outside this repo first
git clone https://github.com/sczhou/ProPainter.git ../models/ProPainter

# Use via the named model (recommended)
videowipe clean input.mp4 --model propainter --propainter-dir ../models/ProPainter

# Or via the generic external command (equivalent, now argv-form)
videowipe clean input.mp4 --external-command "python scripts/propainter_wipe.py"
```

> **Note:** ProPainter requires a GPU with ~16GB VRAM for 480p video and is licensed under NTU S-Lab License 1.0 (non-commercial).

<details>
<summary><strong>Quality comparison: ProPainter vs STTN</strong></summary>

Tested on a multilingual music video (Korean + Burmese subtitles, 852x480, 10s clip). Both models used the same mask.

| Original | ProPainter (GPU fp16) | STTN (CPU ONNX) |
|----------|----------------------|-----------------|
| <img src="pics/comparison/others_original.png" width="260"> | <img src="pics/comparison/others_propainter.png" width="260"> | <img src="pics/comparison/others_sttn.png" width="260"> |

Comparison images are in `pics/comparison/`.

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

The pipeline has three stages:

1. **Detection** — A DBNet-based text detector samples frames across the video, finds text regions in each frame, clusters them by position, and selects the best preview frame. Supports multilingual content out of the box.

2. **Target selection** — Detected regions are classified by type (subtitle, watermark, logo, timestamp). Optional OCR reads the text content. An intent parser (rule-based or LLM-backed via `--agent`) lets you specify what to remove in natural language.

3. **Inpainting** — Masked regions are filled in using temporal information from neighboring frames. The default backend is STTN (8-layer spatial-temporal transformer with CNN encoder). Any external model can be substituted via `--external-command`.

## Docker

No Python? No problem. Run videowipe directly with Docker.

**CPU:**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe clean /data/input.mp4 -o /data/result/
```

**GPU:**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu clean /data/input.mp4 -o /data/result/
```

The `gpu` tag tracks the current main build. A versioned `v0.4.0-gpu` image was not produced by the v0.4.0 tag run; use `gpu` or build locally until the next tagged release.

Use the included wrapper script to auto-select the CPU or GPU image:

```bash
./scripts/docker-videowipe.sh clean input.mp4 -o result/
```

| Image | Size | GPU | Notes |
|-------|------|-----|-------|
| `ghcr.io/kkenny0/videowipe:latest` | ~480 MB | No | CPU only, smallest image |
| `ghcr.io/kkenny0/videowipe:gpu` | ~1.4 GB | Yes | Latest prebuilt GPU image |
| `videowipe:gpu` | ~1.4 GB | Yes | Local build tag |

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
docker run --rm -v "$(pwd)":/data videowipe:latest clean /data/input.mp4 -o /data/result/

# GPU
docker run --rm --gpus all -v "$(pwd)":/data videowipe:gpu clean /data/input.mp4 -o /data/result/
```

## Support

If videowipe saves you time on subtitle, watermark, or text-overlay cleanup, you can support continued maintenance here:

<https://kkenny0.github.io/support/>

Support helps keep model packaging, Docker images, detection tuning, and documentation maintained.

## Credits

This project builds on [STTN](https://github.com/researchmm/STTN) and the original [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) implementation. The built-in text detection model is from [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR).

## License

MIT
