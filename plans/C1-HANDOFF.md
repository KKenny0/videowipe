# C1 Handoff — Local-first Web UI 已完成

> 这份文档是 C1 的最新接力状态，不再是实施前计划。
>
> **当前实现 commit**: `35bfe015cef91a48a488b3099bfdcb237bb63881` (`feat: add local web UI`)
> **发布版本**: `v0.4.0`（已发布）
> **上一版 release**: `v0.3.0`
> **当前边界**: GitHub tag/release 已完成；PyPI 仍不处理。
> **最新状态**: Docker CPU/GPU 浮动镜像和 fresh-clone Web UI 验收见 `plans/POST-C1-FOLLOWUPS.md`。

---

## 当前结论

C1 已完成并提交。VideoWipe 现在有一个 local-first Web UI：用户可以在浏览器里上传视频、输入清理意图、预览检测目标、确认清理、查看进度，并下载保留原始音轨的 MP4。

这次实现已经从“CLI-only 工具”推进到“本地视频清理工作台”。Web UI 仍然是本地单用户模型，不提供托管 SaaS、鉴权、多租户或云端 LLM 服务。

---

## 已交付能力

- `videowipe serve` 启动本地 Web 服务，默认监听 `127.0.0.1:8000`。
- 新增 `videowipe[web]` extra，包含 FastAPI、Uvicorn 和 multipart 上传依赖。
- 浏览器端单页 Web UI：
  - VideoWipe logo 和产品化首屏
  - Upload / Preview / Download 三步状态
  - 视频上传和 intent 输入
  - 检测预览图展示
  - 候选目标列表和默认勾选
  - 清理进度显示
  - 完成后下载 MP4
- 服务端全局串行 job 模型：
  - 同一时间只处理一个任务
  - busy 时返回 409
  - preview 阶段保留任务槽等待用户确认
  - Reset/取消会释放 stale preview job，避免旧页面或多 tab 卡住服务
- confirm 阶段复用 preview 产物：
  - 默认选中未变化时直接使用 `auto_mask.png`
  - 用户改选候选时用 bbox 重建 mask
- STTN 输出沿用 A0 的音频保留能力。
- README/README_CN 已加入 Web UI 小节和三张流程截图。

---

## 实际文件变化

### 新增

- `src/videowipe/server/__init__.py`
- `src/videowipe/server/jobs.py`
- `src/videowipe/server/app.py`
- `src/videowipe/web/__init__.py`
- `src/videowipe/web/index.html`
- `tests/test_server.py`
- `pics/web-ui/01-upload.png`
- `pics/web-ui/02-preview.png`
- `pics/web-ui/03-download.png`

### 修改

- `src/videowipe/cli.py`
- `src/videowipe/engine.py`
- `src/videowipe/external.py`
- `src/videowipe/tasks/base.py`
- `src/videowipe/tasks/detext.py`
- `pyproject.toml`
- `README.md`
- `README_CN.md`

---

## 和原计划的偏差

1. **engine 并非零改动。**
   原计划希望 confirm 透传完全停留在 server 层；实际为了 Web 进度显示，把 `progress` 回调从 server 传到 `WipeEngine.process()`，再透传到 task/inpainter。

2. **改选候选使用 bbox 近似 mask。**
   `clean_candidates.json` 不保存每个候选的精确 mask。默认路径仍直接使用 `auto_mask.png`，质量最好；用户增删候选时 server 用 bbox 重建 mask，这是 C1 可接受折中。

3. **补了 Windows external command 修复。**
   原 handoff 记录过 Windows `shlex.split` 反斜杠问题。实施期间修复了平台感知 argv split，避免 Windows 路径被错误拆解。

4. **Reset 从前端清屏升级为真实后端释放。**
   实测中发现 preview_ready job 会占住唯一任务槽；现在新增 `/jobs/current` 和 `DELETE /jobs/current`，Reset 会释放 stale preview job。运行中的任务不会被强行取消。

---

## 验证记录

已在 Windows 本地验证：

- `python -m pytest tests/test_server.py tests/test_boundaries.py -v --basetemp=.pytest_tmp_commit`
  - 70 passed
- scoped ruff:
  - `python -m ruff check src/videowipe/server tests/test_server.py src/videowipe/cli.py src/videowipe/engine.py src/videowipe/external.py src/videowipe/tasks/base.py src/videowipe/tasks/detext.py`
  - passed
- `git diff --check`
  - passed
- wheel 包内容检查：
  - `videowipe/server/app.py` included
  - `videowipe/server/jobs.py` included
  - `videowipe/web/index.html` included
- 本地服务检查：
  - `http://127.0.0.1:8765/` 返回 200
  - `DELETE /jobs/current` 空闲时返回 `{"state":"idle"}`
- 浏览器流程检查：
  - 上传页正常
  - preview 图和候选列表正常
  - download 状态正常
  - 输出 MP4 保留音轨

---

## Release 结果

这批变化已作为 minor 版本发布：`v0.4.0`。

原因：

- 新增用户可见 Web UI
- 新增 `videowipe serve` 命令
- 新增 `web` optional dependency
- README 和截图已同步
- 修复了 busy stale job 这一类真实可见的使用问题

本轮 release 范围：

- GitHub tag: `v0.4.0`
- GitHub release: `v0.4.0`
- Release URL: `https://github.com/KKenny0/videowipe/releases/tag/v0.4.0`
- Docker CPU 版本标签存在：`v0.4.0`
- Docker GPU 浮动标签已恢复：`gpu`、`main-gpu`
- Docker GPU 版本标签缺失：`v0.4.0-gpu`
- 不发布 PyPI

实际 release notes 可继续沿用：

```markdown
## What's new

- Added a local-first Web UI via `videowipe serve`.
- Added `videowipe[web]` extra for FastAPI/Uvicorn-based local usage.
- Added a review-first browser flow: upload, preview detected targets, confirm cleanup, and download.
- Preserved original audio in downloaded MP4 outputs.
- Added Reset handling that releases stale preview jobs instead of leaving the local server busy.
- Updated English and Chinese READMEs with Web UI screenshots.

## Verification

- 70 tests passed on Windows.
- Scoped ruff checks passed.
- Wheel package includes the Web server and HTML UI files.

## Notes

- This release is local-first and single-user. It does not add hosted SaaS, authentication, multi-tenant queues, or cloud LLM calls.
- PyPI publishing is intentionally out of scope for this release.
```

---

## 仍需注意

- 全仓库 `ruff check src tests` 仍会报旧核心文件中的 unused import/变量；C1 相关文件 scoped ruff 已通过。
- CPU STTN 对长视频仍慢，Web UI 只是把进度显示出来，没有改变模型速度。
- 当前全局串行 job 模型适合 local-first 单用户，不适合 hosted 多用户服务。
- Docker 已确认 CPU/GPU 浮动镜像可用，但当前没有 `v0.4.0-gpu` 版本镜像。
- fresh-clone Web UI 验收已通过；A2 继续保持条件性推迟，除非后续真实样例明确反馈 STTN 输出质量不足。

---

## 当前接力入口

1. `plans/POST-C1-FOLLOWUPS.md` 是 post-C1 最新状态文档。
2. 目前没有必须立即启动的新功能批次。
3. A2/E2FGVI 不应默认启动；只有真实样例显示 STTN 输出质量不足时，才先核实 E2FGVI 授权再进入评估。
4. PyPI 仍不在当前范围内，README 保持源码安装路径。
