<p align="center">
  <img src="assets/logo.svg" width="80" height="80" alt="videowipe">
</p>

<h1 align="center">videowipe</h1>

<p align="center">
  面向生产项目嵌入的视频清理引擎，用于擦除硬字幕、水印和文字叠加。<br>
  通过 Python、CLI 或自有 Worker 完成目标检测、时序轨道审阅和本地画面修复。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="License: GPL-3.0"></a>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## SDK 优先

videowipe 是可复用的 Python 视频清理引擎。一个 `WipeEngine` 可以在批处理中连续处理多个视频，避免重复加载模型；第三方修复模型也可以通过统一的整段视频协议接入。CLI、本地 Web UI 和 Docker 镜像只是 SDK 的适配入口，不是彼此独立的产品运行时。

当前产品边界刻意保持收敛：检测不需要的画面叠加、生成 mask、修复被覆盖的视频区域。翻译工作台、时间线、发布流程和 Agent 聊天界面不属于核心 SDK。

## 功能

videowipe 可以检测并擦除视频中的硬字幕、水印、Logo 和时间戳。一条命令完成完整流水线：采样帧 → 检测文字区域 → 生成可审阅的 WipePlan → 选择 remove/keep 轨道 → 修复背景。

无需手动提供 mask。内置检测器开箱即用，支持多语种内容。
每条轨道独立保存生效时间段和精确 mask，字幕消失后，不会继续擦除后续帧里的同一片区域。

默认使用 STTN 作为修复后端。通过 `--external-command` 可接入任何外部模型 — [ProPainter](https://github.com/sczhou/ProPainter) 已验证为更高质量的替代方案。

## 安装

需要 Python 3.10+，以及 ONNX Runtime 或 PyTorch。基础 SDK 默认使用 `opencv-python-headless`，因此可以在没有桌面显示服务的 Worker 和容器中运行。VideoWipe 目前还没有发布到 PyPI，请从源码安装：

```bash
git clone https://github.com/KKenny0/videowipe.git
cd videowipe

# 无界面 SDK + 轻量 ONNX Runtime 后端：
pip install -e ".[onnx]"

# 需要时再添加本地 Web 适配入口：
pip install -e ".[web,onnx]"

# 可选 extras：
pip install -e ".[torch]"  # PyTorch 后端
pip install -e ".[ocr]"    # OCR 文字识别
pip install -e ".[propainter]"  # 仅适配器依赖，不附带模型代码或权重
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

使用结构化 SDK 契约并复用引擎，避免重复加载模型。取消令牌只属于单次任务；下一项任务应创建新的令牌。

```python
from videowipe import CancellationToken, WipeEngine, WipeRequest

def report(event):
    print(event.phase, event.completed, event.total)

with WipeEngine(task="detext") as engine:
    result = engine.run(
        WipeRequest(
            video="clip1.mp4",
            mask="mask.png",
            output_dir="result/clip1",
        ),
        on_progress=report,
        cancellation=CancellationToken(),
    )
    print(result.output_path, result.backend, result.timings)
```

`WipeEngine.run()` 返回 `WipeResult`；无效输入、缺少后端、取消和处理失败会抛出稳定的 `WipeError` 子类。兼容入口 `process()` 与 `remove_text()` 继续保留。长期 Worker 和第三方模型接入可参考可运行的[批处理示例](examples/batch_worker.py)与[自定义 Inpainter 示例](examples/custom_inpainter.py)。

### 审阅与编辑 WipePlan

clean 流水线会生成一份 `WipePlan`：可读、可审阅的 JSON 计划，每个检测目标是一条 *track*，带 `remove`|`keep` 动作、生效时间段（segments）和精确 mask。可以不加载修复模型先生成计划，再执行审阅或修改过的计划。

画面顶部的常驻叠加默认保留，只有明确选择时才会移除。采样较稀造成的边界误差会写入 warnings，不会伪装成逐帧精确判断。

```python
from videowipe import WipeEngine, WipeRequest

engine = WipeEngine(task="clean")
plan = engine.plan(WipeRequest(video="input.mp4", output_dir="plan/"))
# → plan/wipe_plan.json + plan/wipe_plan_masks.npz
engine.cleanup()
```

`wipe_plan.json` 对 agent 可读。只编辑每条 track 的 `action` 和 `segments`——空间 mask 存在同目录的 `.npz`，不要改动。随后对同一视频校验并执行修改后的计划：

```python
with WipeEngine(task="clean") as engine:
    result = engine.run(WipeRequest(
        video="input.mp4",
        output_dir="result/",
        plan="plan/wipe_plan.json",
    ))
    print(result.warnings, result.timings)
```

或使用 CLI：

```bash
videowipe clean input.mp4 --preview -o plan/                       # 生成计划
videowipe clean input.mp4 --plan plan/wipe_plan.json -o result/    # 执行计划
```

VideoWipe 不绑定任何 LLM 或云服务——计划就是普通 JSON，任何编辑器或本地 agent 都能修改。计划绑定其源视频；对不同的视频执行同一份计划会被拒绝。

### CLI

```bash
# 自动检测并清除所有文字叠加（推荐）
videowipe clean input.mp4 -o result/

# 手动指定 mask
videowipe clean input.mp4 -m mask.png -o result/
```

### 本地 Web UI

```bash
pip install -e ".[web,onnx]"
videowipe serve
# 打开 http://127.0.0.1:8000
```

浏览器流程完全在本地运行：上传视频，查看每条轨道的动作和生效时间段，在 remove 与 keep 之间切换整条轨道，然后下载清理后的 MP4。确认后会直接执行审阅过的 WipePlan 及其精确 mask，不会根据预览框重新拼出近似 mask。文件始终留在本机，下载的视频保留原始音轨。

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
| `--plan` | 执行已有的 `wipe_plan.json` 而非重新检测（与 `-m, --mask` 互斥） | — |
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

1. **检测**：基于 DBNet 的文字检测器对视频进行多帧采样，逐帧定位文字区域，记录目标在时间上的出现情况，再把检测结果聚合成稳定轨道。开箱即用，支持多语种。

2. **生成计划**：检测轨道按字幕、水印、Logo、时间戳分类，并写入经过校验的 WipePlan。计划包含 remove/keep 动作、左闭右开的时间段、源视频身份和精确 mask。可选 OCR 与意图解析帮助决定要移除什么。

3. **修复**：执行审阅后的计划，只在每一帧启用当时生效的 remove 轨道 mask。STTN 利用相邻帧填充这些区域，最终合成继续遵循逐帧 mask。没有时序区间的计划仍可使用静态外部修复后端。

## Docker

没有 Python 环境？直接用 Docker 运行。

**CPU：**

```bash
docker pull ghcr.io/kkenny0/videowipe:latest
docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe clean /data/input.mp4 -o /data/result/
```

**GPU：**

```bash
docker pull ghcr.io/kkenny0/videowipe:gpu
docker run --rm --gpus all -v "$(pwd)":/data ghcr.io/kkenny0/videowipe:gpu clean /data/input.mp4 -o /data/result/
```

浮动标签 `latest` 和 `gpu` 跟随当前 main 构建。只有对应版本的发布工作流成功后，才会提供版本化的 CPU 与 GPU 标签；固定版本前请先检查 GHCR package。

也可以使用自带的 wrapper 脚本自动选择 CPU 或 GPU 镜像：

```bash
./scripts/docker-videowipe.sh clean input.mp4 -o result/
```

| 镜像 | 大小 | GPU | 说明 |
|------|------|-----|------|
| `ghcr.io/kkenny0/videowipe:latest` | ~480 MB | 否 | 仅 CPU，体积最小 |
| `ghcr.io/kkenny0/videowipe:gpu` | ~1.4 GB | 是 | 最新预构建 GPU 镜像 |
| `videowipe:gpu` | ~1.4 GB | 是 | 本地构建标签 |

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

GNU General Public License v3.0，详见 [LICENSE](LICENSE)。

本仓库基于 GPL-3.0 授权的 Video-Auto-Wipe 代码演进。如果分发 videowipe 或与其组合的作品，需要根据实际分发方式确认并履行 GPL-3.0 的相关义务。
