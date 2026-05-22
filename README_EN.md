<p align="center">
  <img src="pics/cover.png" alt="Video-Auto-Wipe" width="480">
</p>

<h1 align="center">Video-Auto-Wipe</h1>

<p align="center">STTN-based video inpainting tool for removing subtitles, logos, and dynamic watermarks.</p>

<p align="center">
  <a href="README.md">中文</a>
</p>

---

Forked from [a312863063/Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) with performance optimizations and code refactoring.

## What changed

- Refactored `demo.py` into an `InpaintingTask` base class with task-specific subclasses
- Added Numba-accelerated frame blending, AMP mixed-precision inference, and `channels_last` memory format
- Reduced processing time on a 23s test clip from 200s to 125s

## Preview

### Subtitle Removal

![detext](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-text/detext_9_ko.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">Watch video</a></p>

The model detects subtitle regions (uniformly styled text) and removes them.

### Logo Removal

![delogo](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-logo/delogo_4.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/delogo_04.mp4">Watch video</a></p>

The model detects stationary pixel blocks in the time domain and removes them as logos.

### Dynamic Watermark Removal

![dynamic logo](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-dynamic-logo/de-dynamic-logo_1.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/de_dynamic_logo.mp4">Watch video</a></p>

The model detects flickering or moving pixel blocks in the time domain and removes them as dynamic watermarks.

## Installation

Requires Python 3.8+. Install PyTorch first, then:

```bash
pip install opencv-python==4.12.0.88 matplotlib==3.10.3 numba==0.61.2 pysrt==1.1.2 tqdm==4.67.1 PyYAML==6.0.2 moviepy==2.1.2
```

## Usage

1. Download pre-trained weights and place them in `pretrained_weight/`: [Baidu Drive](https://pan.baidu.com/s/1JN9-8Glw_ozOrSMgBIyHOw) (code `px0s`)
2. More input samples: [Baidu Drive](https://pan.baidu.com/s/1_tzmvIoEQi3h_24-ieZJ_Q) (code `cnqf`)
3. Run:

```bash
python demo.py
```

Or with arguments:

```bash
python demo.py --task detext --video input/detext_examples/chinese1.mp4 --mask input/detext_examples/mask/chinese1_mask.png --result result/ --weight pretrained_weight/detext_trial.pth
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --task` | Task type: `detext` or `delogo` | `detext` |
| `-v, --video` | Input video path | `input/detext_examples/chinese1.mp4` |
| `-m, --mask` | Mask image path | `input/detext_examples/mask/chinese1_mask.png` |
| `-r, --result` | Output directory | `result/` |
| `-w, --weight` | Model weight path | `pretrained_weight/detext_trial.pth` |
| `-d, --dual` | Show original video side-by-side in output | `False` |
| `-g, --gap` | Segment length; higher values yield better results | `200` |

## Training

### Background Data

- 2,709 movie clips from 300+ HD movies: [Baidu Drive](https://pan.baidu.com/s/1CIgJmFmx5iR2JfgAyjVaeg) (code `xb7o`)
- 864 TV show clips from 40+ HD TV shows: [Baidu Drive](https://pan.baidu.com/s/1lJk6IIWlwxknAie0LlGYOg) (code `9rd4`)

### Foreground Data

- Subtitle removal: generate random text with ImageDraw and simulate variations
- Logo removal: generate random pixel blocks with ImageDraw and simulate temporal consistency
- Dynamic watermark removal: create flickering/jumping effects with Premiere Pro

### Training Process

1. Temporal perception training for specific tasks, teaching the model to detect foreground regions
2. End-to-end fine-tuning with the inpainting model

## Credits

- Upstream: [a312863063/Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)
- STTN paper: [researchmm/STTN](https://github.com/researchmm/STTN)
- Original author's blog: [seeprettyface.com](https://www.seeprettyface.com/)

## License

MIT
