<p align="center">
  <img src="pics/cover.png" alt="Video-Auto-Wipe" width="480">
</p>

<h1 align="center">Video-Auto-Wipe</h1>

<p align="center">基于 STTN 的视频修复工具，用于擦除字幕、台标、动态水印等固定模式内容。</p>

<p align="center">
  <a href="README_EN.md">English</a>
</p>

---

Forked from [a312863063/Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)，主要做了性能优化和代码重构。

## 与上游的区别

- 重构了 `demo.py`，将处理逻辑拆分为 `InpaintingTask` 基类 + 具体任务子类
- 加入 Numba 加速帧混合、AMP 混合精度推理、`channels_last` 内存格式等优化
- 23 秒测试视频处理时间从 200s 降到 125s

## 效果预览

### 字幕擦除

![detext](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-text/detext_9_ko.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/detext_06.mp4">查看视频</a></p>

模型自动感知字幕位置并擦除。感知方式：具有统一样式的文字区域被视作字幕。

### 台标擦除

![delogo](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-logo/delogo_4.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/delogo_04.mp4">查看视频</a></p>

模型自动感知台标位置并擦除。感知方式：在时域上静止不动的像素块被视作台标。

### 动态水印擦除

![dynamic logo](https://github.com/a312863063/Video-Auto-Wipe/blob/main/pics/de-dynamic-logo/de-dynamic-logo_1.JPG)

<p align="center"><a href="http://www.seeprettyface.com/mp4/video-inpainting/de_dynamic_logo.mp4">查看视频</a></p>

模型自动感知动态水印位置并擦除。感知方式：在时域上闪烁出现或动态移动的固定像素块被视作动态水印。

## 安装

需要 Python 3.8+，先安装 PyTorch，然后：

```bash
pip install opencv-python==4.12.0.88 matplotlib==3.10.3 numba==0.61.2 pysrt==1.1.2 tqdm==4.67.1 PyYAML==6.0.2 moviepy==2.1.2
```

## 使用

1. 下载预训练权重放到 `pretrained_weight/` 目录：[百度网盘](https://pan.baidu.com/s/1JN9-8Glw_ozOrSMgBIyHOw)（提取码 `px0s`）
2. 更多输入样例：[百度网盘](https://pan.baidu.com/s/1_tzmvIoEQi3h_24-ieZJ_Q)（提取码 `cnqf`）
3. 运行：

```bash
python demo.py
```

也可以指定参数：

```bash
python demo.py --task detext --video input/detext_examples/chinese1.mp4 --mask input/detext_examples/mask/chinese1_mask.png --result result/ --weight pretrained_weight/detext_trial.pth
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --task` | 任务类型：`detext` 或 `delogo` | `detext` |
| `-v, --video` | 输入视频路径 | `input/detext_examples/chinese1.mp4` |
| `-m, --mask` | 遮罩图片路径 | `input/detext_examples/mask/chinese1_mask.png` |
| `-r, --result` | 输出目录 | `result/` |
| `-w, --weight` | 模型权重路径 | `pretrained_weight/detext_trial.pth` |
| `-d, --dual` | 输出中同时显示原视频 | `False` |
| `-g, --gap` | 分段长度，值越大效果越好 | `200` |

## 训练

### 背景数据

- 基于 300+ 部高清电影制作的 2,709 个片段数据集：[百度网盘](https://pan.baidu.com/s/1CIgJmFmx5iR2JfgAyjVaeg)（提取码 `xb7o`）
- 基于 40+ 部综艺节目制作的 864 个片段数据集：[百度网盘](https://pan.baidu.com/s/1lJk6IIWlwxknAie0LlGYOg)（提取码 `9rd4`）

### 前景数据

- 字幕擦除：用 ImageDraw 生成随机样式、字体的文字并模拟其变化
- 台标擦除：用 ImageDraw 生成随机像素区块，模拟时域一致性
- 动态水印擦除：用 PR 制作闪烁、跳跃等动态特效

### 训练流程

1. 针对特定任务的时域感知训练，让模型能感知到需要擦除的前景区域
2. 融合进擦除模型，端到端微调

## 致谢

- 上游项目：[a312863063/Video-Auto-Wipe](https://github.com/a312863063/Video-Auto-Wipe)
- STTN 论文：[researchmm/STTN](https://github.com/researchmm/STTN)
- 原作者博客：[seeprettyface.com](https://www.seeprettyface.com/)

## License

MIT
