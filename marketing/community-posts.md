# VideoWipe 社区投放稿

口径：本地擦硬字幕和水印。先预览再擦。不上传。不是 Windows 一键软件。

不要改成「AI 视频清理平台」或「剪辑软件」。搜索的人要的是去硬字幕、去水印。

## 先发哪里

按投入产出排。一次只发一个地方，看有没有人真去 clone。

1. **r/selfhosted**（英文）  
   这里的人讨厌把原片传到陌生网站。Docker + 本机网页是他们的语言。
2. **V2EX「分享创造」**（中文）  
   技术向，能讲清「和 VSR 的差别」。不要写成软文。
3. **Show HN**  
   标题要具体。有人问安装和授权时，当场答。
4. **知乎**  
   挂在已有问题下：硬字幕怎么去、视频去水印不想上传。一篇说清楚边界。
5. **B 站**  
   中文增长几乎全靠 15 秒前后对比。VSR 就是这么长起来的。没有成片之前，先别把精力耗在小红书。

晚一点再做：

- **awesome-selfhosted**：第一条正式 release 是 2026-05-22，列表要求满 4 个月。大约 **2026-09-22** 再提 PR。草稿在文末。
- **r/VideoEditing / r/editors**：用「不想在 AE 里一帧帧画 mask」开场，少讲 Docker。
- **JuneYaooo/awesome-ai-media**：已经收了 VSR。可以提一条并列项。

不要做：

- 去 VSR 的 issue / 讨论区贴广告
- 同一周在 Reddit 多个板复制同一段
- 写「完美去水印」「一键秒出」

## r/selfhosted

Title:

```text
VideoWipe – self-hosted hardcoded subtitle and watermark remover (preview first, then wipe)
```

Body:

```text
I keep hitting the same job: a clip has burned-in subtitles or a corner watermark, I want those pixels gone, and I do not want the source file on someone else's server.

VideoWipe runs locally. It samples frames, groups text into tracks (subtitle / watermark / logo / timestamp), shows you the boxes, then inpaints only the tracks you keep marked for removal. The output MP4 keeps the original audio.

Quickest path if you already use Docker:

    docker pull ghcr.io/kkenny0/videowipe:latest
    docker run --rm -v "$(pwd)":/data ghcr.io/kkenny0/videowipe clean /data/input.mp4 -o /data/result/

Or:

    pip install -e ".[web,onnx]"   # from the repo, not on PyPI yet
    videowipe serve                # http://127.0.0.1:8000

It is not a Windows one-click .exe. If that is what you want, video-subtitle-remover is the established desktop app. This one is for preview-before-erase, CLI / Docker / a local web UI, and embedding in a worker.

GPL-3.0. Derived from Video-Auto-Wipe.

https://github.com/KKenny0/videowipe
```

## Show HN

Title:

```text
Show HN: VideoWipe – preview hardcoded subtitles, then wipe them locally
```

Body:

```text
Hardcoded subtitles and watermarks are painted into the frames. Soft .srt tracks are a different problem.

Most tools in this space are either an upload site or a Windows desktop app. I wanted something I could review before it erased anything, and run from a terminal or Docker.

VideoWipe detects burn-in text, builds a JSON plan per track, then inpaints only what you marked. Default backend is STTN on CPU via ONNX. GPU image is optional. Audio is copied through.

    git clone https://github.com/KKenny0/videowipe.git
    cd videowipe
    pip install -e ".[onnx]"
    videowipe clean input.mp4 -o result/

Web UI: pip install -e ".[web,onnx]" && videowipe serve

Not on PyPI yet. GPL-3.0.

https://github.com/KKenny0/videowipe
```

## V2EX

节点：分享创造

标题：

```text
做了个本地擦硬字幕 / 去水印的工具，先看检测框再擦
```

正文：

```text
有些素材画面能用，但底下一行烧录字幕，或者角上挂着水印。在线网站要上传原片。桌面一键软件我也不放心它直接开擦。

VideoWipe 在本机跑。先检测硬字幕、水印、Logo、时间戳，给你看框，按轨道勾选，再修画面。输出 MP4 保留原音轨。

命令行：

    git clone https://github.com/KKenny0/videowipe.git
    cd videowipe
    pip install -e ".[onnx]"
    videowipe clean input.mp4 -o result/

本地网页：pip install -e ".[web,onnx]" && videowipe serve
浏览器开 http://127.0.0.1:8000

想要 Windows 一键安装包的话，VSR（YaoFANGUK/video-subtitle-remover）更合适。这个项目面向：先预览、能 Docker / Worker 跑、能嵌进自己的流程。

还没上 PyPI。GPL-3.0。

https://github.com/KKenny0/videowipe
```

## 知乎

可挂的问题方向：

- 如何去除视频中的硬字幕
- 有没有本地去视频水印的方法
- video-subtitle-remover 之外还有什么选择

标题：

```text
不想上传原片时，我怎么在本地擦硬字幕和水印
```

正文：

```text
硬字幕是烧进画面的字，不是 .srt。关掉字幕轨道去不掉。

常见三条路：传到在线去字幕站；用手动画 mask（AE / 达芬奇）；装一个桌面一键软件。前一条我不想用，后两条要么太累，要么擦之前看不见它准备动哪里。

我做了 VideoWipe。它在本机检测烧录字幕、水印、Logo、时间戳，先出预览框，你勾过再修。音轨原样留下。

适合：归档或二次剪辑前的清理，以及要在服务器上批量跑的人。
不适合：要 Windows 免安装 exe 的人（用 VSR）；要去软字幕、做翻译、重新打轴的人。

复杂运动和半透明水印修不干净，长视频在 CPU 上会慢。先预览，再决定要不要跑完整段。

项目：https://github.com/KKenny0/videowipe
```

## X / Twitter

```text
VideoWipe: local hardcoded-subtitle and watermark removal.

Detect burn-in text, preview the boxes, then wipe. Original audio stays. CLI, Docker, or a local web UI. No upload.

https://github.com/KKenny0/videowipe
```

中文短帖：

```text
本地擦硬字幕和水印。先看检测框，再决定擦哪条。不上传，音轨还在。

命令行 / Docker / 本机网页。
https://github.com/KKenny0/videowipe
```

## r/VideoEditing（后发）

Title:

```text
Open-source local tool to remove burned-in subtitles without drawing masks in AE
```

Body:

```text
Burned-in subs and corner bugs are a pain when the source is all you have.

VideoWipe auto-detects those regions, lets you accept or reject each track, then fills the pixels. It is local (CLI or a browser UI on 127.0.0.1). Not a cloud site, not an NLE.

Good for archive cleanup before a recut. Not a replacement for Resolve on hard motion shots. Preview first.

https://github.com/KKenny0/videowipe
```

## awesome-selfhosted 草稿（2026-09-22 后再提）

文件：`software/videowipe.yml`

```yaml
name: VideoWipe
website_url: https://kkenny0.github.io/videowipe/
description: "Preview-first remover for hardcoded subtitles, watermarks, logos, and timestamps (alternative to online subtitle removers)."
licenses:
  - GPL-3.0
platforms:
  - Docker
  - Python
tags:
  - Media Management
source_code_url: https://github.com/KKenny0/videowipe
```

提 PR 时说明：有 tagged release，第一条是 v0.1.0（2026-05-22）；提供本机 Web UI 和 Docker，不是纯库。

## 发完怎么看有没有用

看这三件事，不要看点赞：

1. README 的 clone / visitor 有没有在发帖后 48 小时动一下
2. 有没有人开 issue 问安装，而不是只说「不错」
3. 有没有人带着自己的视频问「这个框检错了」

如果三天里只有赞没有 clone，换渠道，不要把同一段话再贴一遍。
