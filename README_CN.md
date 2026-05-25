<h1 align="center">videowipe</h1>

<p align="center">
  基于 STTN 的视频修复库。<br>
  擦除硬字幕、水印和文字叠加，<code>pip install videowipe</code> 即可使用。
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 功能

videowipe 使用时空 Transformer 网络擦除视频中的硬字幕。你可以提供标记擦除区域的 mask 图片，也可以让内置检测器自动生成，模型利用前后帧的时域信息填充背景。

## 安装

需要 Python 3.8+，以及 ONNX Runtime 或 PyTorch。

```bash
# 已有 PyTorch：
pip install videowipe

# 轻量 ONNX Runtime 后端：
pip install videowipe[onnx]

# 或 PyTorch 后端：
pip install videowipe[torch]
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

批量处理时复用引擎，避免重复加载模型：

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

# 旧命令：仅自动检测字幕
videowipe detext -v input.mp4 -o result/

# 手动指定 mask
videowipe detext -v input.mp4 -m mask.png -o result/
```

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
| `--agent` | 本地 LLM CLI 做意图选择（如 `claude`、`codex`） | — |
| `-g, --gap` | 每轮处理的分段长度，值越大效果越好、速度越慢 | `200` |
| `-d, --dual` | 输出中同时显示原视频 | 关闭 |

#### `detext` 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-v, --video` | 输入视频路径 | 必填 |
| `-m, --mask` | Mask 图片路径（省略时自动检测） | 自动检测 |
| `-o, --output` | 输出目录 | `result/` |
| `-w, --weight` | 模型权重路径。PyTorch 接受 `.pth`/`.pt`；ONNX 需要以 `.onnx` 结尾的前缀路径，并存在对应的 `_encoder`、`_transformer`、`_decoder` 文件。 | 自动下载 |
| `-g, --gap` | 每轮处理的分段长度，值越大效果越好、速度越慢 | `200` |
| `-d, --dual` | 输出中同时显示原视频 | 关闭 |

## 效果预览

### 字幕擦除

| Before | After |
|--------|-------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">查看视频</a></p>

## 原理

模型基于 STTN（时空 Transformer 网络），8 层 transformer block 对多尺度 patch 做时域注意力。CNN 编码器提取帧特征，跨帧注意力机制利用邻近帧和参考帧信息，解码器生成修复结果。

性能优化：AMP 混合精度推理、`channels_last` 内存布局。23 秒测试视频处理时间 125s。

## Docker

没有 Python 环境？直接用 Docker 运行。

**CPU：**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe detext -v /data/input.mp4 -o /data/result/
```

**GPU（需要 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)）：**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu detext -v /data/input.mp4 -o /data/result/
```

或者使用自带的 wrapper 脚本（自动检测 GPU）：

```bash
./scripts/docker-videowipe.sh detext -v input.mp4 -o result/
```

| 镜像 | 大小 | GPU | 说明 |
|------|------|-----|------|
| `videowipe:latest` | ~480 MB | 否 | 仅 CPU，体积最小 |
| `videowipe:gpu` | ~1.4 GB | 是 | ONNX Runtime + CUDA |

## 致谢

基于 [STTN](https://github.com/researchmm/STTN) 和 [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)。内置文字检测模型来自 [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)。

## License

MIT
