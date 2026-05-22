<h1 align="center">videowipe</h1>

<p align="center">
  Video inpainting library powered by STTN.<br>
  Remove subtitles, logos, and watermarks. <code>pip install videowipe</code> and go.
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

---

## What it does

videowipe uses a Spatial-Temporal Transformer Network to detect and erase fixed-pattern content in video: hardcoded subtitles, channel logos, animated watermarks. You provide a video and a mask image marking the region to erase. The model fills in the background using temporal information from surrounding frames.

## Install

Requires Python 3.8+ and PyTorch.

```bash
# If you already have PyTorch:
pip install videowipe

# If you need PyTorch (CPU):
pip install videowipe[cpu]
```

Model weights download automatically on first run to `~/.videowipe/weights/`. No manual setup needed.

## Usage

### Python API

```python
from videowipe import remove_text

# One-shot: process a single video
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
engine.process(video="clip1.mp4", mask="mask.png", output="result/")
engine.process(video="clip2.mp4", mask="mask.png", output="result/")
engine.cleanup()
```

### CLI

```bash
videowipe detext -v input.mp4 -m mask.png -o result/
videowipe detext -v input.mp4 -m mask.png -o result/ -g 400
videowipe delogo -v input.mp4 -m mask.png -o result/
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-v, --video` | Input video path | required |
| `-m, --mask` | Mask image path | required |
| `-o, --output` | Output directory | `result/` |
| `-w, --weight` | Model weight path (skips auto-download if set) | auto |
| `-g, --gap` | Segment length per pass; higher = better quality, slower | `200` |
| `-d, --dual` | Show original video side-by-side in output | off |

## Preview

### Subtitle removal

| Before | After |
|--------|-------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">Watch video</a></p>

### Logo removal

| Before | After |
|--------|-------|
| <img src="pics/de-logo/delogo_4_before.JPG" width="400"> | <img src="pics/de-logo/delogo_4_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/delogo_04.mp4">Watch video</a></p>

### Dynamic watermark removal

| Before | After |
|--------|-------|
| <img src="pics/de-dynamic-logo/de-dynamic-logo_1_before.JPG" width="400"> | <img src="pics/de-dynamic-logo/de-dynamic-logo_1_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/de_dynamic_logo.mp4">Watch video</a></p>

## How it works

The model is an STTN (Spatial-Temporal Transformer Network) with 8 stacked transformer blocks operating on multi-scale patches. It encodes video frames with a CNN backbone, runs temporal attention across neighboring and reference frames, then decodes the inpainted result.

Key optimizations in this fork: Numba-accelerated frame blending, AMP mixed-precision inference, and `channels_last` memory layout. A 23-second test clip processes in 125s (down from 200s in the original).

## Credits

This project builds on [STTN](https://github.com/researchmm/STTN) and the original [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) implementation.

## License

MIT
