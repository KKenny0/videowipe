<p align="center">
  <img src="assets/logo.svg" width="80" height="80" alt="VideoWipe 标志">
</p>

<h1 align="center">VideoWipe</h1>

<p align="center">
  <strong>本地硬字幕去除、视频去水印、去 Logo。<br>先预览检测结果，再擦烧录字幕和时间戳。命令行、Docker、或本地网页。</strong>
</p>

<p align="center">
  自动检测烧录文字 · 按轨道审阅 · 修复背景 · 保留原音轨。<br>
  视频留在本机，不上传云端。
</p>

<p align="center">
  <a href="https://github.com/KKenny0/videowipe/stargazers"><img src="https://img.shields.io/github/stars/KKenny0/videowipe?style=flat" alt="GitHub stars"></a>
  <a href="https://github.com/KKenny0/videowipe/releases"><img src="https://img.shields.io/github/v/release/KKenny0/videowipe" alt="最新版本"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="许可证: GPL-3.0"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/本地运行-success.svg" alt="本地运行">
  <img src="https://img.shields.io/badge/Docker-CPU%20%7C%20GPU-2496ED.svg" alt="Docker CPU 与 GPU">
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="https://kkenny0.github.io/videowipe/">网站</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#常见问题">常见问题</a>
  ·
  <a href="#docker">Docker</a>
</p>

---

## 本地硬字幕去除、视频去水印

**VideoWipe** 是一个可自托管的 **硬字幕去除** / **烧录字幕擦除** / **视频去水印** 工具。它找出画面里的 **硬字幕（烧录字幕）**、**水印**、**Logo** 和 **时间戳**，让你 **先审阅每条轨道**，再只擦你勾过的区域。

它对应这些常见需求：

- 怎么去掉视频里的硬字幕 / 烧录字幕 / 硬字幕去除
- 本地去水印、去 Logo、视频 delogo
- 不想上传到在线「一键去字幕」网站
- 能用命令行、Docker、自己的 Worker 跑的开源方案

清理后的 MP4 **保留原始音轨**。

如果你要的是 Windows 下一键安装的桌面软件，更成熟的选择是 [video-subtitle-remover (VSR)](https://github.com/YaoFANGUK/video-subtitle-remover)。VideoWipe 适合另一种用法：擦之前先看检测框，能在无界面环境跑，或者把检测 → 审阅 → 修复嵌进自己的流程。

> 注意：VideoWipe **不处理** 软字幕（`.srt` / `.ass` 轨道）。它只擦 **烧进画面** 的文字。

## 快速开始

**环境：** Python 3.10+，以及 ONNX Runtime 或 PyTorch。模型权重首次运行时自动下载到 `~/.videowipe/weights/`。

尚未发布到 PyPI，请从源码安装：

```bash
git clone https://github.com/KKenny0/videowipe.git
cd videowipe
pip install -e ".[onnx]"

# 自动检测并擦除画面上的文字叠加
videowipe clean input.mp4 -o result/
```

更想用浏览器（依然是本机）：

```bash
pip install -e ".[web,onnx]"
videowipe serve
# 打开 http://127.0.0.1:8000 — 上传、预览轨道、下载清理后的 MP4
```

不想装 Python？直接看 [Docker](#docker)。

可选依赖：`.[torch]`（PyTorch）、`.[ocr]`（OCR 识别）、`.[propainter]`（仅适配依赖，不含模型本体）。

## 功能一览

| | |
|--|--|
| **硬字幕擦除** | 烧录字幕，支持多语种画面 |
| **水印 / Logo / 时间戳** | 角落常驻标记、屏显时钟等 |
| **自动检测** | 不必手动画 mask（需要时仍可自备） |
| **先预览再擦** | 按轨道选择 remove / keep，并带时间段 |
| **本地优先** | 命令行、网页、Docker、Python SDK，无需云账号 |
| **保留原音** | 输出 MP4 带上源视频音轨 |
| **多语种检测** | 中文、英文、韩文等开箱可用 |
| **可换修复模型** | 默认 STTN；可选 [ProPainter](https://github.com/sczhou/ProPainter) 或任意外部修复命令 |
| **可批量 / 可嵌入** | 同一引擎连续处理多条视频 |

## 效果示例

### 字幕擦除

| 处理前 | 处理后 |
|--------|--------|
| <img src="pics/de-text/detext_9_ko_before.JPG" width="400" alt="处理前：带有硬字幕的韩文字幕画面"> | <img src="pics/de-text/detext_9_ko_after.JPG" width="400" alt="处理后：VideoWipe 擦除硬字幕后的画面"> |

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">观看示例视频</a></p>

### 自动检测（无需手动画框）

内置检测器能在多语种内容上定位文字区域：

<p float="left">
  <img src="pics/detection/chinese1_detected.jpg" width="32%" alt="VideoWipe 自动检测中文硬字幕">
  <img src="pics/detection/english1_detected.jpg" width="32%" alt="VideoWipe 自动检测英文硬字幕">
  <img src="pics/detection/others_detected.jpg" width="32%" alt="VideoWipe 自动检测多语种字幕与水印">
</p>

| 视频 | 候选 | 选中 | 类型 |
|------|------|------|------|
| 中文剧集 | 4 | 2 | 顶部字幕、底部字幕 |
| 英文片段 | 2 | 2 | 底部字幕 |
| 音乐视频（韩文 + 缅甸文） | 7 | 5 | 顶部水印、底部多语字幕 |

`--detect-mode balanced`（采样约 50 帧）下的结果。绿色框为将要清理的区域。

### 本地网页界面

| 上传 | 预览目标 | 下载 |
|------|----------|------|
| <img src="pics/web-ui/01-upload.png" width="260" alt="VideoWipe 网页：上传待清理视频"> | <img src="pics/web-ui/02-preview.png" width="260" alt="VideoWipe 网页：预览检测到的字幕与水印"> | <img src="pics/web-ui/03-download.png" width="260" alt="VideoWipe 网页：下载保留原音轨的清理结果"> |

## 适合谁用？

VideoWipe 面向已经确定要擦烧录字、但不想把决定权交给黑盒的人。

- **剪辑和资料整理**：二次剪辑前清理私有或归档素材（请自行遵守版权与平台条款）
- **自己托管的人**：不愿意把原片传到在线去字幕站
- **开发者**：需要「检测 → 审阅 → 修复」作为本机命令行、Docker 任务或 Python Worker

如果你要的是免安装的 Windows `.exe`，那是 VSR 的主场。

### 常见场景

- 擦掉剧集、课程、语言学习片里的 **底部硬字幕**
- 清理 **角落 Logo / 水印** 再复用
- 去掉 **时间戳** 或平台角标
- 多语种视频 **先看检测框**，只擦需要的那几条
- 在无桌面环境的 **Worker 上批量跑**

## 工作原理

三步：

1. **检测** — 采样视频帧，找出文字区域，并按时间聚成稳定轨道（多语种，可不手动画 mask）。  
2. **规划** — 生成可审阅的 **WipePlan**：每条轨道有类型（字幕 / 水印 / Logo / 时间戳）、remove|keep、时间段和精确 mask。  
3. **修复** — 仅对标记为 remove 的轨道按帧生效；默认用 **STTN** 从邻帧补全；需要更高画质时可接 ProPainter 等外部模型。  

可先 `--preview` 只做检测，改计划 JSON，再执行清理——不会一上来盲擦。

## 和其他做法对比

| | 在线去字幕站 | [VSR](https://github.com/YaoFANGUK/video-subtitle-remover) | 手动画 mask（AE / 达芬奇） | **VideoWipe** |
|--|--------------|------------------------------------------------------------|---------------------------|---------------|
| 隐私 | 需上传 | 本地 | 本地 | **本地** |
| 自动找烧录字 | 有时有 | 有 | 全靠手 | **有** |
| 擦之前能审阅 | 少见 | 有限 | 靠人工 | **有（WipePlan / 网页）** |
| 无界面 / Docker / Worker | 否 | 桌面软件 | 桌面软件 | **命令行、Docker、Python** |
| Windows 一键 `.exe` | 有时有 | **有** | 不适用 | 无（源码或 Docker） |
| 接入自己的流程 | 难 | 难 | 难 | **Python SDK + CLI** |

### VideoWipe 不做什么

- 不是 **软字幕** 工具（`.srt` / `.ass`）  
- 不是 **翻译** 或重新打轴产品  
- 不是每段素材都完美无痕——快运动、细纹理、半透明标记更难，请 **先预览**  
- 不是云 SaaS——由你在本机或自己的机器上运行  

## 命令行用法

```bash
# 推荐：自动检测并擦除
videowipe clean input.mp4 -o result/

# 只要字幕，只要画面底部
videowipe clean input.mp4 --target subtitle --region bottom -o result/

# 自然语言意图
videowipe clean input.mp4 --intent "去掉底部中文字幕" -o result/

# 只预览检测结果（不做修复）
videowipe clean input.mp4 --preview -o plan/

# 执行已审阅的计划
videowipe clean input.mp4 --plan plan/wipe_plan.json -o result/

# 自备 mask 时
videowipe clean input.mp4 -m mask.png -o result/
```

#### `clean` 选项

| 参数 | 说明 | 默认 |
|------|------|------|
| `--target` | 清理类型（可重复）：`subtitle`、`timestamp`、`watermark`、`logo` | 自动检测全部 |
| `--region` | 屏幕区域（可重复）：`top`、`bottom`、`top-left`、`top-right`、`bottom-left`、`bottom-right`、`center` | 全部区域 |
| `--intent` | 自然语言清理意图 | — |
| `--preview` | 只写检测产物，不修复 | 关 |
| `--plan` | 执行已有 `wipe_plan.json`（与 `-m, --mask` 互斥） | — |
| `--confirm` | 显示检测目标并确认后再处理 | 关 |
| `--detect-mode` | `fast`（24 帧）；`balanced`（50）/ `sensitive`（80）会对检测出的 remove 段做更密复核 | `balanced` |
| `--ocr` | OCR：`auto`、`off`、`rapidocr` | `auto` |
| `--agent` | 本地 LLM CLI，用于意图选择（如 `claude`、`codex`） | — |
| `--external-command` | 外部修复命令（绕过内置 STTN） | — |
| `-g, --gap` | 每个修复分段的帧数；`25` 平衡性能和质量，值越大上下文越多，但计算和内存成本超线性增长 | `25` |
| `-d, --dual` | 输出中并排显示原片 | 关 |
| `-m, --mask` | mask 图片路径 | 自动 |

<details>
<summary><strong>旧命令：detext</strong></summary>

`detext` 主要只处理字幕。新用法请优先用 `clean`。

```bash
videowipe detext -v input.mp4 -o result/
videowipe detext -v input.mp4 -m mask.png -o result/
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `-v, --video` | 输入视频 | 必填 |
| `-m, --mask` | mask 路径 | 自动 |
| `-o, --output` | 输出目录 | `result/` |
| `-w, --weight` | 模型权重（PyTorch `.pth`/`.pt`，或 ONNX 前缀） | 自动 |
| `-g, --gap` | 每个修复分段的帧数；值越大计算和内存成本越高 | `25` |
| `-d, --dual` | 并排原片 | 关 |
| `--external-command` | 外部修复命令 | — |

</details>

## Docker

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

或用脚本自动选 CPU / GPU 镜像：

```bash
./scripts/docker-videowipe.sh clean input.mp4 -o result/
```

| 镜像 | 体积 | GPU | 说明 |
|------|------|-----|------|
| `ghcr.io/kkenny0/videowipe:latest` | ~480 MB | 否 | 仅 CPU，体积最小 |
| `ghcr.io/kkenny0/videowipe:gpu` | ~1.4 GB | 是 | 预构建 GPU 镜像 |
| `videowipe:gpu` | ~1.4 GB | 是 | 本地构建标签 |

浮动标签 `latest` / `gpu` 跟随当前 main 构建；正式版本请到 GHCR 查带版本号的标签。

<details>
<summary><strong>从源码构建镜像</strong></summary>

```bash
docker build --target runtime-cpu -t videowipe:latest .
docker build --target runtime-gpu --build-arg VARIANT=gpu -t videowipe:gpu .
```

GPU 镜像需要 NVIDIA 运行时才能走 CUDA；否则 ONNX Runtime 会回退到 CPU。

```bash
docker run --rm -v "$(pwd)":/data videowipe:latest clean /data/input.mp4 -o /data/result/
docker run --rm --gpus all -v "$(pwd)":/data videowipe:gpu clean /data/input.mp4 -o /data/result/
```

</details>

## 更高画质修复（可选）

默认后端是 **STTN**（可用 CPU + ONNX）。难修的镜头可接外部模型。

[ProPainter](https://github.com/sczhou/ProPainter) 已验证为更高画质选项：

```bash
git clone https://github.com/sczhou/ProPainter.git ../models/ProPainter

videowipe clean input.mp4 --model propainter --propainter-dir ../models/ProPainter
# 等价写法：
videowipe clean input.mp4 --external-command "python scripts/propainter_wipe.py"
```

> **说明：** ProPainter 在 480p 大约需要 16GB 显存，模型许可证为 NTU S-Lab License 1.0（非商业）。

<details>
<summary><strong>画质对比：ProPainter vs STTN</strong></summary>

多语种音乐视频（韩文 + 缅甸文字幕，852×480，约 10 秒），同一 mask。

| 原片 | ProPainter（GPU fp16） | STTN（CPU ONNX） |
|------|------------------------|------------------|
| <img src="pics/comparison/others_original.png" width="260" alt="原片硬字幕画面"> | <img src="pics/comparison/others_propainter.png" width="260" alt="ProPainter 修复结果"> | <img src="pics/comparison/others_sttn.png" width="260" alt="STTN 修复结果"> |

</details>

## Python API（流水线 / 批量）

命令行、网页、Docker 用的是同一套引擎。要做批处理或接入自己的 Worker 时用这里。

```python
from videowipe import remove_text

# mask 可选；省略时自动检测
remove_text(video="input.mp4", output="result/")
```

带目标选择的完整 clean 流水线：

```python
from videowipe import WipeEngine

engine = WipeEngine(task="clean", detect_mode="balanced", ocr="auto")
engine.process(
    video="input.mp4",
    targets=["subtitle", "watermark"],
    regions=["bottom"],
    intent="去掉中文字幕和 Logo 水印",
    output="result/",
)
engine.cleanup()
```

长驻引擎批量处理（避免重复加载模型）：

```python
from videowipe import CancellationToken, WipeEngine, WipeRequest

with WipeEngine(task="detext") as engine:
    result = engine.run(
        WipeRequest(video="clip1.mp4", mask="mask.png", output_dir="result/clip1"),
        cancellation=CancellationToken(),
    )
    print(result.output_path, result.backend, result.timings)
```

示例见 [examples/batch_worker.py](examples/batch_worker.py)、[examples/custom_inpainter.py](examples/custom_inpainter.py)。

<details>
<summary><strong>审阅与修改 WipePlan</strong></summary>

可不加载修复模型先生成计划；只改 JSON 里每条轨道的 `action` / `segments`（不要改 sidecar `.npz` mask），再执行：

```python
from videowipe import CancellationToken, WipeEngine, WipeRequest

engine = WipeEngine(task="clean")
plan = engine.plan(
    WipeRequest(video="input.mp4", output_dir="plan/"),
    cancellation=CancellationToken(),
)
engine.cleanup()

with WipeEngine(task="clean") as engine:
    result = engine.run(WipeRequest(
        video="input.mp4",
        output_dir="result/",
        plan="plan/wipe_plan.json",
    ))
```

```bash
videowipe clean input.mp4 --preview -o plan/
videowipe clean input.mp4 --plan plan/wipe_plan.json -o result/
```

计划就是普通 JSON，不绑定任何 LLM 或云服务。计划与源视频绑定，换片执行会被拒绝。

</details>

## 常见问题

**能去掉软字幕（`.srt` / `.ass`）吗？**  
不能。软字幕是独立文件或轨道。VideoWipe 只擦 **烧进画面** 的硬字幕。

**视频会离开我的电脑吗？**  
默认路径全在本机（命令行、本机网页 `127.0.0.1`、带挂载的 Docker）。不会上传到 VideoWipe 云端。

**音轨还在吗？**  
在。输出 / 下载的 MP4 保留源视频音轨。

**必须有显卡吗？**  
不必。CPU + ONNX Runtime 即可。有 GPU 时会更快，ProPainter 等路径也更吃显存。

**上 PyPI 了吗？**  
还没有。请从本仓库安装，或拉 Docker 镜像。

**硬字幕和水印可以分开选吗？**  
可以。用 `--target subtitle` / `--target watermark` / `--region bottom`、自然语言 `--intent`，或网页里按轨道开关。

**STTN 和 ProPainter 怎么选？**  
STTN 是默认（更轻、CPU 友好）。ProPainter 在难修区域往往更好，但更吃显存，且该模型为非商业许可。

**一定能修得看不出痕迹吗？**  
不能保证。快速运动、细纹理、半透明标记更难。长任务前请用 **预览 / 确认**。

**能嵌进自己的产品或 Worker 吗？**  
可以。把 VideoWipe 当可嵌入引擎：一个 `WipeEngine`、稳定的请求/结果类型、可换修复后端。产品边界是检测 → 计划 → 修复，不是完整剪辑台或云工作室。

**和 video-subtitle-remover（VSR）有什么区别？**  
VSR 是带 Windows 安装包的桌面软件。VideoWipe 强调先预览再擦，并且能用命令行、Docker、本地网页或自己的 Worker 跑。要解决的是同一件事（硬字幕去除 / 视频去水印），用法不一样。

## 支持项目

如果 VideoWipe 帮你省下了去字幕、去水印的时间：

<https://kkenny0.github.io/support/>

支持会用于模型打包、Docker 镜像、检测调参和文档维护。

## 相关项目

| 项目 | 关系 |
|------|------|
| [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe) | 本仓库的上游来源 |
| [video-subtitle-remover (VSR)](https://github.com/YaoFANGUK/video-subtitle-remover) | 同一需求上最常见的桌面软件 |
| [STTN](https://github.com/researchmm/STTN) | 默认画面修复模型 |
| [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) | 内置文字检测 |
| [ProPainter](https://github.com/sczhou/ProPainter) | 可选的更高画质修复（不随包装） |
| [InpaintDelogo](https://github.com/Purfview/InpaintDelogo) | AviSynth+ 去 Logo 插件，另一套栈 |

## 致谢

基于 [STTN](https://github.com/researchmm/STTN) 与原始 [Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)。内置文字检测来自 [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR)。

## 许可证

GNU General Public License v3.0，详见 [LICENSE](LICENSE)。

本仓库衍生自 GPL-3.0 许可的 Video-Auto-Wipe。若分发 VideoWipe 或衍生合并作品，请自行核对 GPL-3.0 对你分发方式的要求。

## Star 趋势

<a href="https://www.star-history.com/?repos=KKenny0%2Fvideowipe&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=KKenny0/videowipe&type=date&theme=dark&legend=top-left&sealed_token=Ocoofe793tAMlarurXhegI2A9NIEC5YIpP2Cz-YgNw1B89AGk08cXRlJw2HNRGjszo-EFjTFczzN4B7nCZmpl313BQKr4pMFCZTxOUMjKUEhUC9M78uQn2Uc04M2uH07nmaDAtJlldSHdnz4TXHJOW5ZjXWzGvZZUfdeCwp-3mrtL92xORJTNgaRZvFQ" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=KKenny0/videowipe&type=date&legend=top-left&sealed_token=Ocoofe793tAMlarurXhegI2A9NIEC5YIpP2Cz-YgNw1B89AGk08cXRlJw2HNRGjszo-EFjTFczzN4B7nCZmpl313BQKr4pMFCZTxOUMjKUEhUC9M78uQn2Uc04M2uH07nmaDAtJlldSHdnz4TXHJOW5ZjXWzGvZZUfdeCwp-3mrtL92xORJTNgaRZvFQ" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=KKenny0/videowipe&type=date&legend=top-left&sealed_token=Ocoofe793tAMlarurXhegI2A9NIEC5YIpP2Cz-YgNw1B89AGk08cXRlJw2HNRGjszo-EFjTFczzN4B7nCZmpl313BQKr4pMFCZTxOUMjKUEhUC9M78uQn2Uc04M2uH07nmaDAtJlldSHdnz4TXHJOW5ZjXWzGvZZUfdeCwp-3mrtL92xORJTNgaRZvFQ" />
 </picture>
</a>
