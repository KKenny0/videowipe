# Post-C1 Follow-ups Handoff

> 给 Codex 接力执行。C1（Web 前端）已完成并发布 v0.4.0（commit `e327cc0`）。
> 本文档记录 C1 后的真实发布状态和剩余事项。
>
> **基准状态**（2026-06-29 复核）：
> - `pyproject.toml` version = `0.4.0`
> - `src/videowipe/__init__.py` version = `0.4.0`
> - A0 ✅ / A0.5 ✅ / C1 ✅ 全部落地
> - PyPI 尚未发布，README 应使用源码安装路径
> - GHCR CPU / GPU 浮动镜像存在；`v0.4.0-gpu` 版本镜像缺失
> - P3 fresh-clone Web UI 验收已通过

---

## 已完成的第一批修复

### 版本同步

- `src/videowipe/__init__.py` 已同步为 `0.4.0`。
- 新增版本一致性测试，防止 `pyproject.toml` 和包内版本再次分叉。

验证：

```bash
python -c "import videowipe; print(videowipe.__version__)"
python -m pytest tests/test_boundaries.py -k version_fields -v --basetemp=.pytest_tmp
```

### README 安装入口

- `README.md` / `README_CN.md` 已移除 PyPI badge。
- 安装说明已从 `pip install videowipe` 改为 `git clone` + `pip install -e ".[web,onnx]"`。
- Web UI 小节也已改为源码安装命令。

背景：`https://pypi.org/pypi/videowipe/json` 当前返回 404。发布 PyPI 不在本轮范围内。

### Docker 发布状态

- GHCR CPU 标签存在：`latest`、`main`、`v0.4.0`。
- GHCR GPU 标签存在：`gpu`、`main-gpu`。
- `v0.4.0-gpu` 不存在，因为 v0.4.0 tag workflow 的 `build-gpu` 在 GitHub Actions 6 小时限制处取消，后续成功复跑是 `workflow_dispatch` on `main`。
- README 已恢复 `gpu` 预构建镜像用法，但未声称 `v0.4.0-gpu` 可用。

相关 run：

```text
https://github.com/KKenny0/videowipe/actions/runs/28168947697
https://github.com/KKenny0/videowipe/actions/runs/28346354905
```

### ROADMAP 状态同步

- `plans/ROADMAP.md` 中 C1 已从“未开始”改为“完成（commit 35bfe01, v0.4.0）”。
- C1 的实际 API、交付文件和自动化验收结果已按当前实现更新。
- NEXT_WORK 冲突表已收敛：Web UI 红线已关闭，registry 红线已过时，模型默认依赖红线仍保留。

---

## P1：验证并提交第一批修复（已完成）

这一步已完成并提交，不应再扩大范围。

提交：

```text
cbbb52c docs: fix v0.4.0 post-release entrypoints
```

已验证：

```bash
python -c "import videowipe; print(videowipe.__version__)"
python -m pytest tests/test_server.py tests/test_boundaries.py -v --basetemp=.pytest_tmp_post_c1
python -m ruff check src/videowipe/__init__.py tests/test_boundaries.py
git diff --check
```

---

## P2：GPU Docker workflow 修复（已完成）

**现状**：CPU 镜像可用，GPU 浮动镜像可用。不要声称 `v0.4.0-gpu` 可用，直到下一次 tag workflow 真实产出版本化 GPU 标签。

### 已定位

- `build-gpu` 不是卡在 CUDA 镜像、权重下载、ONNX Runtime GPU 依赖或 buildx cache。
- 日志显示 GPU runtime 的 `apt-get install` 阶段触发了 `tzdata` 交互式时区选择，停在 `Geographic area:` 等输入，直到 GitHub Actions 6 小时后取消。
- 修复点：`Dockerfile` 的 GPU runtime stage 已设置 `TZ=Etc/UTC`，并在 apt 安装时使用 `DEBIAN_FRONTEND=noninteractive`，同时显式安装并非交互配置 `tzdata`。
- 回归保护：`tests/test_boundaries.py` 已增加 Dockerfile 检查，防止 GPU stage 重新丢失非交互时区配置。
- 远端复跑后，`build-gpu` 已越过 `tzdata` 阶段，但在下载权重前报错：`ModuleNotFoundError: No module named 'videowipe'`。
- 新修复点：GPU runtime 使用 deadsnakes Python 3.11，需要显式设置 `PYTHONPATH=/usr/local/lib/python3.11/site-packages`，才能读取 builder stage 复制到 `/usr/local` 的包。

### 远端确认

- 已推送修复 commit：`6ddfbd7`、`9543cda`。
- `Build & Push Docker Images` workflow 已手动触发并成功：`28346354905`。
- `build-cpu` / `build-gpu` 均为 success。
- GHCR 标签已确认：`latest`、`main`、`v0.2.0`、`v0.3.0`、`v0.4.0`、`gpu`、`main-gpu`。

验证命令：

```bash
gh run view 28168947697 --job 83427907164 --log
gh run view 28346354905 --json status,conclusion,jobs,url
python -m pytest tests/test_boundaries.py -k docker_stage -v --basetemp=.pytest_tmp
```

GHCR 标签验证可用 registry API 或 Docker：

```bash
docker manifest inspect ghcr.io/kkenny0/videowipe:gpu
docker manifest inspect ghcr.io/kkenny0/videowipe:main-gpu
```

---

## P3：C1 非技术用户人工验收（已完成）

这是最高价值的产品验收。目标不是再写功能，而是验证 local-first 安装和使用路径是否真的能被非技术用户走通。

### 验收流程

1. `git clone https://github.com/KKenny0/videowipe.git`
2. `cd videowipe`
3. `pip install -e ".[web,onnx]"`
4. `videowipe serve`
5. 浏览器打开 `http://127.0.0.1:8000`
6. 上传真实视频、输入清理意图、预览、确认、等待进度、下载 MP4

### 结果

- fresh clone HEAD：`3a3506c`
- 独立 venv 安装：`pip install -e ".[web,onnx]"` 通过。
- `videowipe serve --host 127.0.0.1 --port 8876` 启动成功，首页返回 200。
- Web UI 自动化完整跑通：
  - 上传 2 秒 480p 真实样例片段；
  - intent：`Remove bottom subtitles`；
  - preview 渲染 5 个候选，默认选中 1 个底部字幕候选；
  - confirm 后清理完成，下载 MP4 成功；
  - 下载文件经 ffmpeg 检查包含视频流和 AAC 音频流。
- 页面首屏、预览态、下载态截图检查通过，无明显布局遮挡。
- 浏览器控制台无 error/warning。
- 服务完成后回到 idle，没有留下 busy job。

### 发现

- 当前 CPU 环境会出现 ONNX Runtime 的 CUDA provider warning：
  `Specified provider 'CUDAExecutionProvider' is not in available provider names`。
  这是 CPU-only 环境的预期降级，不影响本次 ONNX CPU 验收。
- 未发现安装、模型下载、端口、浏览器、视频格式、检测结果、进度、下载或音频保留的阻塞问题。

### 后续判断

- 本次没有暴露安装/GPU 环境类阻塞，因此暂不需要新增安装脚本或调整 Docker CPU 路径。
- 本次短样例没有暴露 Web UI 流程阻塞；更长视频的速度和输出质量仍属于后续真实用户反馈项。
- 暂不启动 A2，除非后续真实样例明确反馈 STTN 输出质量不够。

---

## 剩余 P4：暂不启动的功能方向

### B'1：改选候选的精确 mask

当前 C1 默认路径会直接使用 `auto_mask.png`，质量最好。只有用户手动增删候选时才用 bbox 近似重建 mask。

除非 P3 发现用户频繁改选候选，否则不建议现在做。

### A2：E2FGVI 评估

现在不推荐启动。只有以下任一条件满足才进入 A2：

- P3 人工验收明确反馈 STTN 输出质量不够；
- 在 GPU 机器上补做帧级对比，确认 STTN 填充质量明显不达标。

若启动 A2，第一步必须是核实 E2FGVI 的实际 LICENSE 原文，不凭记忆判断授权。
