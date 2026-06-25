<p align="center">
  <img src="assets/logo.svg" width="80" height="80" alt="videowipe">
</p>

<h1 align="center">videowipe</h1>

<p align="center">
  擦除视频中的硬字幕、水印和文字叠加。<br>
  自动检测目标、生成 mask、修复画面 — <code>pip install videowipe</code> 即可使用。
</p>

<p align="center">
  <a href="https://pypi.org/project/videowipe/"><img src="https://img.shields.io/pypi/v/videowipe.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/videowipe/"><img src="https://img.shields.io/pypi/pyversions/videowipe.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 功能

videowipe 可以检测并擦除视频中的硬字幕、水印、Logo 和时间戳。一条命令完成完整流水线：采样帧 → 检测文字区域 → 选择目标（支持 OCR 和自然语言意图解析）→ 生成 mask → 修复背景。

无需手动提供 mask。内置检测器开箱即用，支持多语种内容。

默认使用 STTN 作为修复后端。通过 `--external-command` 可接入任何外部模型 — [ProPainter](https://github.com/sczhou/ProPainter) 已验证为更高质量的替代方案。

## 安装

需要 Python 3.8+，以及 ONNX Runtime 或 PyTorch。

```bash
# 已有 PyTorch：
pip install videowipe

# 轻量 ONNX Runtime 后端：
pip install videowipe[onnx]

# 或 PyTorch 后端：
pip install videowipe[torch]

# 可选：OCR 文字识别，提升检测准确率
pip install videowipe[ocr]
```

模型权重在首次运行时自动下载到 `~/.videowipe/weights/`，无需手动配置。

## 使用

### Python API

```python
from videowipe import remove_text

# Mask 可选 — 省略时自动检测字幕区域
remove_text(
    video="input.mp4",
    output="result/",
)

# 也可以手动指定 mask
remove_text(
    video="input.mp4",
    mask="mask.png",
    output="result/",
)
```

### 完整流水线（目标选择）

使用 `task="clean"` 启用完整检测流水线，支持目标选择、意图解析和 OCR：

```python
from videowipe import WipeEngine

engine = WipeEngine(task="clean", detect_mode="balanced", ocr="auto")
engine.process(
    video="input.mp4",
    targets=["subtitle", "watermark"],
    regions=["bottom"],
    intent="去掉底部中文字幕和 logo 水印",
    output="result/",
)
engine.cleanup()
```

### 批量处理

复用引擎避免重复加载模型：

```python
from videowipe import WipeEngine

engine = WipeEngine(task="detext")
engine.process(video="clip1.mp4", output="result/")
engine.process(video="clip2.mp4", mask="mask.png", output="result/")
engine.cleanup()
```

### CLI

```bash
# 自动检测并清除所有文字叠加（推荐）
videowipe clean input.mp4 -o result/

# 手动指定 mask
videowipe clean input.mp4 -m mask.png -o result/
```

### 本地 Web UI

```bash
pip install "videowipe[web,onnx]"
videowipe serve
# 打开 http://127.0.0.1:8000
```

浏览器流程完全在本地运行：上传视频、预览并确认检测到的目标、执行清理，然后下载保留原始音轨的 MP4。

| 上传 | 预览目标 | 下载 |
|------|----------|------|
| <img src="pics/web-ui/01-upload.png" width="260" alt="VideoWipe 网页上传界面"> | <img src="pics/web-ui/02-preview.png" width="260" alt="VideoWipe 网页目标预览界面"> | <img src="pics/web-ui/03-download.png" width="260" alt="VideoWipe 网页下载界面"> |

#### `clean` 命令选项

```bash
# 只清除特定类型的目标
videowipe clean input.mp4 --target subtitle
videowipe clean input.mp4 --target watermark

# 指定屏幕区域
videowipe clean input.mp4 --region bottom
videowipe clean input.mp4 --region top-right

# 自然语言意图
videowipe clean input.mp4 --intent "去掉底部中文字幕"

# 预览检测结果（不执行修复）
videowipe clean input.mp4 --preview -o result/

# 交互确认检测结果后再处理
videowipe clean input.mp4 --confirm
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--target` | 清除目标类型（可重复）：`subtitle`、`timestamp`、`watermark`、`logo` | 自动检测所有 |
| `--region` | 屏幕区域（可重复）：`top`、`bottom`、`top-left`、`top-right`、`bottom-left`、`bottom-right`、`center` | 所有区域 |
| `--intent` | 自然语言清除意图 | — |
| `--preview` | 仅输出检测产物（不执行修复） | 关闭 |
| `--confirm` | 显示检测目标并确认后再处理 | 关闭 |
| `--detect-mode` | 检测预设：`fast`（24帧）、`balanced`（50帧）、`sensitive`（80帧） | `balanced` |
| `--ocr` | OCR 文字识别：`auto`、`off`、`rapidocr` | `auto` |
| `--agent` | 本地 LLM CLI 做意图选择（如 `claude`、`codex`） | — |
| `--external-command` | 外部修复命令（绕过内置 STTN） | — |
| `-g, --gap` | 每轮处理的分段长度，值越大效果越好、速度越慢 | `200` |
| `-d, --dual` | 输出中同时显示原视频 | 关闭 |
| `-m, --mask` | Mask 图片路径（省略时自动检测） | 自动检测 |

<details>
<summary><strong>旧命令：detext</strong></summary>

`detext` 仅自动检测字幕。新项目建议使用 `clean`。

```bash
# 自动检测字幕
videowipe detext -v input.mp4 -o result/

# 手动指定 mask
videowipe detext -v input.mp4 -m mask.png -o result/
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-v, --video` | 输入视频路径 | 必填 |
| `-m, --mask` | Mask 图片路径（省略时自动检测） | 自动检测 |
| `-o, --output` | 输出目录 | `result/` |
| `-w, --weight` | 模型权重路径。PyTorch 接受 `.pth`/`.pt`；ONNX 需要以 `.onnx` 结尾的前缀路径，并存在对应的 `_encoder`、`_transformer`、`_decoder` 文件。 | 自动下载 |
| `-g, --gap` | 每轮处理的分段长度，值越大效果越好、速度越慢 | `200` |
| `-d, --dual` | 输出中同时显示原视频 | 关闭 |
| `--external-command` | 外部修复命令（绕过内置 STTN） | — |

</details>

## 外部模型

通过 `--external-command` 使用第三方修复模型代替内置 STTN。命令接收 `<video> <mask> <output_dir>` 三个参数，需要在输出目录中生成结果视频。

[ProPainter](https://github.com/sczhou/ProPainter) 已通过验证，是更高质量的替代方案。附带开箱即用的包装脚本：

```bash
# 先在仓库外克隆 ProPainter
git clone https://github.com/sczhou/ProPainter.git ../models/ProPainter

# 通过命名模型调用（推荐）
videowipe clean input.mp4 --model propainter --propainter-dir ../models/ProPainter

# 或通过通用外部命令（等价，现为 argv 形式）
videowipe clean input.mp4 --external-command "python scripts/propainter_wipe.py"
```

> **注意：** ProPainter 需要 ~16GB 显存的 GPU 处理 480p 视频，许可证为 NTU S-Lab License 1.0（非商业用途）。

<details>
<summary><strong>效果对比：ProPainter vs STTN</strong></summary>

测试视频为多语言 MV（韩语 + 缅甸语字幕，852x480，10秒片段），两个模型使用相同 mask。

| 原始画面 | ProPainter（GPU fp16） | STTN（CPU ONNX） |
|----------|----------------------|-----------------|
| <img src="pics/comparison/others_original.png" width="260"> | <img src="pics/comparison/others_propainter.png" width="260"> | <img src="pics/comparison/others_sttn.png" width="260"> |

对比截图位于 `pics/comparison/` 目录。

</details>

## 效果预览

### 字幕擦除

| Before | After |
|--------|-------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">查看视频</a></p>

### 自动检测效果

内置检测器可自动定位多语种文字区域，无需手动提供 mask：

<p float="left">
  <img src="pics/detection/chinese1_detected.jpg" width="32%">
  <img src="pics/detection/english1_detected.jpg" width="32%">
  <img src="pics/detection/others_detected.jpg" width="32%">
</p>

| 视频 | 检测候选 | 选中 | 类型 |
|------|---------|------|------|
| 中文剧集 | 4 | 2 | 顶部字幕、底部字幕 |
| 英文片段 | 2 | 2 | 底部字幕 |
| 音乐视频（韩语 + 缅甸语） | 7 | 5 | 顶部水印、底部多语种字幕 |

使用 `--detect-mode balanced`（采样 50 帧）测试。绿框为选中待修复区域。

## 工作原理

流水线分三个阶段：

1. **检测** — 基于 DBNet 的文字检测器对视频进行多帧采样，逐帧定位文字区域，按位置聚类，选出最佳预览帧。开箱即用，支持多语种。

2. **目标选择** — 检测到的区域按类型分类（字幕、水印、Logo、时间戳）。可选 OCR 读取文字内容。意图解析器（基于规则或通过 `--agent` 接入 LLM）支持用自然语言指定要清除什么。

3. **修复** — 利用前后帧的时域信息填充被 mask 的区域。默认后端为 STTN（8 层时空 Transformer + CNN 编码器）。通过 `--external-command` 可替换为任何外部模型。

## Docker

没有 Python 环境？直接用 Docker 运行。

**CPU：**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe clean /data/input.mp4 -o /data/result/
```

**GPU（需要 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)）：**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu clean /data/input.mp4 -o /data/result/
```

或者使用自带的 wrapper 脚本（自动检测 GPU）：

```bash
./scripts/docker-videowipe.sh clean input.mp4 -o result/
```

| 镜像 | 大小 | GPU | 说明 |
|------|------|-----|------|
| `videowipe:latest` | ~480 MB | 否 | 仅 CPU，体积最小 |
| `videowipe:gpu` | ~1.4 GB | 是 | ONNX Runtime + CUDA |

### 从源码构建

使用 `--target` 选择镜像类型：

```bash
# CPU
docker build --target runtime-cpu -t videowipe:latest .

# GPU（构建时需要 NVIDIA Container Toolkit 以拉取基础镜像）
docker build --target runtime-gpu --build-arg VARIANT=gpu -t videowipe:gpu .
```

> **注意：** GPU 镜像需要在具备 NVIDIA 运行时的机器上验证 CUDA 执行。否则 ONNX Runtime 会静默回退到 CPU。

构建完成后运行：

```bash
# CPU
docker run --rm -v "$(pwd)":/data videowipe:latest clean /data/input.mp4 -o /data/result/

# GPU
docker run --rm --gpus all -v "$(pwd)":/data videowipe:gpu clean /data/input.mp4 -o /data/result/
```

## 支持项目

如果 videowipe 帮你节省了字幕、水印或文字叠加清理时间，可以在这里支持后续维护：

<https://kkenny0.github.io/support/>

你的支持会帮助我继续维护模型打包、Docker 镜像、检测调优和文档。

## 致谢

基于 [STTN](https://github.com/researchmm/STTN) 和 [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)。内置文字检测模型来自 [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)。

## License

MIT
