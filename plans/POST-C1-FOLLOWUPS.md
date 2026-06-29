# Post-C1 Follow-ups Handoff

> 给 Codex 接力执行。C1（Web 前端）已完成并发布 v0.4.0（commit `e327cc0`）。
> 本文档记录 C1 后的真实发布状态和剩余事项。
>
> **基准状态**（2026-06-29 复核）：
> - `pyproject.toml` version = `0.4.0`
> - `src/videowipe/__init__.py` version = `0.4.0`
> - A0 ✅ / A0.5 ✅ / C1 ✅ 全部落地
> - PyPI 尚未发布，README 应使用源码安装路径
> - GHCR CPU 镜像存在；GPU 预构建镜像缺失

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

- GHCR CPU 标签存在：`latest`、`v0.4.0`。
- GHCR GPU 标签缺失：`gpu`、`v0.4.0-gpu`。
- v0.4.0 Docker workflow 中 `build-cpu` 成功，`build-gpu` 在 GitHub Actions 6 小时限制处取消。
- README 已保留 CPU Docker 用法，并把 GPU 预构建镜像改成“暂不可用，本地构建”。

相关 run：

```text
https://github.com/KKenny0/videowipe/actions/runs/28168947697
```

### ROADMAP 状态同步

- `plans/ROADMAP.md` 中 C1 已从“未开始”改为“完成（commit 35bfe01, v0.4.0）”。
- C1 的实际 API、交付文件和自动化验收结果已按当前实现更新。
- NEXT_WORK 冲突表已收敛：Web UI 红线已关闭，registry 红线已过时，模型默认依赖红线仍保留。

---

## 剩余 P1：验证并提交第一批修复

这一步是当前变更的收口，不应扩大范围。

建议验证：

```bash
python -c "import videowipe; print(videowipe.__version__)"
python -m pytest tests/test_server.py tests/test_boundaries.py -v --basetemp=.pytest_tmp_post_c1
python -m ruff check src/videowipe/__init__.py tests/test_boundaries.py
git diff --check
```

若全部通过，提交建议：

```text
docs: fix v0.4.0 post-release entrypoints
```

---

## P2：GPU Docker workflow 调查

**现状**：CPU 镜像可用，GPU 镜像不可用。不要在 README 里恢复 `docker pull ghcr.io/kkenny0/videowipe:gpu`，直到 GPU 标签真实存在。

### 已定位

- `build-gpu` 不是卡在 CUDA 镜像、权重下载、ONNX Runtime GPU 依赖或 buildx cache。
- 日志显示 GPU runtime 的 `apt-get install` 阶段触发了 `tzdata` 交互式时区选择，停在 `Geographic area:` 等输入，直到 GitHub Actions 6 小时后取消。
- 修复点：`Dockerfile` 的 GPU runtime stage 已设置 `TZ=Etc/UTC`，并在 apt 安装时使用 `DEBIAN_FRONTEND=noninteractive`，同时显式安装并非交互配置 `tzdata`。
- 回归保护：`tests/test_boundaries.py` 已增加 Dockerfile 检查，防止 GPU stage 重新丢失非交互时区配置。
- 远端复跑后，`build-gpu` 已越过 `tzdata` 阶段，但在下载权重前报错：`ModuleNotFoundError: No module named 'videowipe'`。
- 新修复点：GPU runtime 使用 deadsnakes Python 3.11，需要显式设置 `PYTHONPATH=/usr/local/lib/python3.11/site-packages`，才能读取 builder stage 复制到 `/usr/local` 的包。

### 仍需远端确认

1. 推送包含 Dockerfile 修复的 commit。
2. 触发 `Build & Push Docker Images` workflow。
3. 确认 `build-gpu` 成功，并且 GHCR 出现 `gpu` / `main-gpu` 或下一版本对应 GPU tag。
4. 只有远端 GPU 标签真实存在后，README 才能恢复预构建 GPU 镜像命令。

验证命令：

```bash
gh run view 28168947697 --job 83427907164 --log
python -m pytest tests/test_boundaries.py -k docker_stage -v --basetemp=.pytest_tmp
```

GHCR 标签验证可用 registry API 或 Docker：

```bash
docker manifest inspect ghcr.io/kkenny0/videowipe:gpu
docker manifest inspect ghcr.io/kkenny0/videowipe:v0.4.0-gpu
```

---

## 剩余 P3：C1 非技术用户人工验收

这是最高价值的产品验收。目标不是再写功能，而是验证 local-first 安装和使用路径是否真的能被非技术用户走通。

流程：

1. `git clone https://github.com/KKenny0/videowipe.git`
2. `cd videowipe`
3. `pip install -e ".[web,onnx]"`
4. `videowipe serve`
5. 浏览器打开 `http://127.0.0.1:8000`
6. 上传真实视频、输入清理意图、预览、确认、等待进度、下载 MP4

记录：

- 卡在安装、模型下载、端口、浏览器、视频格式、检测结果、速度，还是输出质量。
- 如果主要卡在安装/GPU 环境，优先考虑安装脚本或 Docker CPU 路径。
- 如果主要卡在输出质量，再讨论 A2。

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
