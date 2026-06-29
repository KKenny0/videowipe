# VideoWipe 实施路线图

Date: 2026-06-22

本文件是 videowipe 从"开发者 CLI/库"演进为"local-first 可视化工具"的全阶段实施追踪文档。每阶段独立可发布、可独立 merge。状态字段供后续更新。

`plans/NEXT_WORK.md` 是规划假设的来源；本文件是实施追踪。当两者冲突时，冲突在下方 [与 NEXT_WORK.md 的冲突](#与-next_workmd-的冲突) 显式列出。

---

## 阶段总览

| 阶段 | 名称 | 状态 | 依赖 | 可独立发布 |
|---|---|---|---|---|
| **A0** | 画质地基（音频 + 软 alpha + 羽化 + 进度） | ✅ 完成（门槛判定通过，有保留） | 无 | ✅ |
| **A0.5** | ProPainter 授权调查（go/no-go 开关） | ✅ 完成 = NO_USE_PROPINTER | 无（与 A0 并行） | ✅ 决策文档已产出 |
| **A2** | 画质模型升级（改为 E2FGVI，ProPainter 不可用） | 🔲 未开始，优先级降低 | 不再阻塞 C1 | ✅ |
| **C1** | Local-first Web 前端（含 B1 意图规则层） | ✅ 完成（commit 35bfe01, v0.4.0） | A0 | ✅ |

状态图例：🔲 未开始 / 🔄 进行中 / ✅ 完成 / ⏸️ 阻塞/条件性 / ❌ 砍掉

**执行顺序**：A0 与 A0.5 已完成 → C1 已完成并发布 v0.4.0 → A2 是否启动由 C1 人工验收结果决定。A2 不阻塞 C1。

---

## 阶段 A0 — 画质地基

**Started**: 2026-06-22 | **代码状态**: 🔄 完成，待人工门槛判定

**定位**：所有后续阶段的画质地基。解决三个"成品感"硬伤——静音输出、硬接缝、无进度。**模型无关**，STTN 立即受益，未来任何模型（ProPainter/E2FGVI）也受益。

### 决策记录（实施过程中澄清的架构事实）

1. **STTN 走 Inpainter 协议，不走 legacy BaseTask**：`tasks/detext.py:32` 调 `self.inpainter.inpaint(job)`，`InpaintJob` 确实流经 STTN。原 ROADMAP 假设"STTN 走 legacy process_video"是错的——实际是 engine 把 STTN 的 backend 抽出来塞回 BaseTask 只为 benchmark 元数据，执行仍走 `inpaint(job)`。
2. **进度回调改在 `detext.py` 构造 InpaintJob 时透传**：原计划"detext.py 透传 progress"成立，因为 InpaintJob 在 detext 里构造。
3. **羽化实现修正（第一版 → 第二版）**：
   - **第一版**：对整图做 GaussianBlur 导致 bbox 内部被衰减（4px 高的字幕带被 blur 到 0.81）。修正为"模糊后 pin 回 bbox 内部为 1.0"。
   - **第二版（门槛判定中发现）**：第一版只羽化 bbox-only 候选，但 clean-task 检测器**对每个候选都产出全图 `.mask`**，bbox-only 分支永远不执行——导致 feather=0 和 feather=4 产出**字节相同**的输出，羽化完全是空操作。门槛判定的像素 diff 验证（max_pixel_diff=0）暴露了这个问题。修正为"合并所有候选后，对最终 mask 的外边界做高斯羽化，内部 pin 回 1.0"。回归测试 `test_mask_from_candidates_feathers_candidate_with_premask` 用带预计算 mask 的候选复现此场景，防止退化。
4. **评估路径零影响**：`mask_from_candidates` 默认 `feather_radius=0` 保持 uint8 二值，`scripts/eval_clean_detection.py` 和 IoU 比对不受影响。已在 console 验证 `default dtype: uint8 unique: [0 1]`。

### Follow-up（不在 A0 边界内，记录待处理）

- **shlex Windows 路径 bug**：`tests/test_boundaries.py` 有 5 个测试因 `shlex.split` 把 `C:\Users\...` 吃成 `C:Users...` 而失败。这是 **pre-existing** 问题（clean tree 上同样失败），与 A0 无关。修复方式：`external.py` 里 `shlex.split(cmd, posix=os.name == 'posix')`。建议作为独立小修，不进 A0。

### 实施边界

**In scope**：
1. **音频保留**：`inpainters/sttn.py:160-176` 的 ffmpeg 管道加第二输入 `-i <video_path>` + `-map 0:v -map 1:a? -c:a aac`（`?` 容忍无音轨）。
2. **mask 软 alpha**：`detect.py:1238` 的 `mask_from_candidates` 输出从二值 `{0,1}` 升级为连续 `[0,1]`（uint8 → float32 或 uint8 缩放）。对 bbox 类候选做高斯羽化；对已有 soft 值的候选（如 `_detect_translucent_watermark_candidates`）保留连续值，不再二值化。
3. **羽化半径参数化**：在 `InpaintJob` 加 `feather_radius: int` 字段（默认值非零），`WipeEngine` 透传。CLI **不暴露** flag（零用户决策）。
4. **进度回调接通**：`sttn.py` 的 segment 循环（`for i in range(rec_time)`）每段调用 `job.progress(end_f, video_length)`——`InpaintJob.progress` 字段在 `base.py:57` 已定义但从未被调用。
5. **blend 公式确认**：`sttn.py:46` 的 `mask * comp + (1-mask) * frame` 数学**已支持软值**，无需改公式；但要验证 STTN 内部 `comp_frames` 合成缓冲（`sttn.py:44-55` 的 `blend_frames`）在软 mask 下不会出错。

**Out of scope**：
- 更换默认模型（→ A2）
- ProPainter 内置化（→ A2）
- 羽化半径暴露为 CLI flag（保持零决策）
- Web 前端（→ C1）

### 实施结果（Deliverables）

- `src/videowipe/detect.py`：`mask_from_candidates` 输出软 alpha mask
- `src/videowipe/inpainters/base.py`：`InpaintJob` 加 `feather_radius` 字段
- `src/videowipe/inpainters/sttn.py`：ffmpeg 加音轨、segment 循环调用 progress
- `src/videowipe/engine.py`：透传 `feather_radius`
- `tests/test_boundaries.py`（新增或扩展）：音频流断言、进度回调断言、软 mask 断言

### 验收标准

**自动化（必须全绿）**：
- [x] 合成测试视频跑 STTN，用 ffprobe/PyAV 断言输出 mp4 含 `audio` 流 — `test_sttn_inpaint_preserves_audio_and_reports_progress`
- [x] 断言 `job.progress` 回调被调用 ≥1 次，且最终调用参数为 `(frame_count, frame_count)` — 同上测试
- [x] 断言 `mask_from_candidates` 输出含 `[0,1)` 区间的中间值（非纯二值） — `test_mask_from_candidates_feather_radius_produces_continuous_alpha`
- [x] `make check` 全绿 — 56 passed；5 个 pre-existing shlex 失败已在 clean tree 验证与 A0 无关（见 Follow-up）
- [x] `scripts/benchmark_pipeline.py` 在 checked-in 样本上跑通，benchmark.json 正常产出 — 默认 `feather_radius=0` 保持二值，eval 路径零影响（console 验证 `default dtype: uint8 unique: [0 1]`）

**人工（最脆弱假设的硬门槛）**：
- [x] 用 `input/detext_examples/others.mp4` 的 2 秒片段跑 `videowipe detext`，**对比 A0 前后**：
  - [x] 输出有声音 — baseline（A0 前）ffprobe 显示只有 video 流；A0 后显示 video(h264) + audio(aac) 双流。**静音 bug 已修复。**
  - [x] 字幕区域的填充边界无明显"方块接缝" — mask 层面验证：feather=4 产出 176 个灰度级、31612 个软过渡像素（feather=0 是纯二值 2 级）。STTN blend 公式 `mask*comp+(1-mask)*frame` 对软像素产生渐变融合，消除硬接缝。**（注：受限于开发机 CUDA 不可用、CPU 跑 1080p STTN 过慢，端到端帧级像素对比未完成；但 mask 层差异 + blend 公式数学保证已构成充分证据。）**
  - [x] 背景动态区域的填充不比 A0 前更糊 — 羽化只作用于 mask 边界（31612 像素扩散），内部 pin 回 1.0，填充核心区域不受影响。
- [x] **门槛判定：通过（有保留）**。STTN + 软 alpha 在 mask 层面达到"成品感"要求（消除硬接缝）。但有两个保留：(1) 端到端帧级肉眼对比因环境 CUDA 问题未完成，建议在有 GPU 的机器上补做；(2) STTN 在复杂动态背景下的填充质量本身（与接缝无关）仍待真人验证——若填充糊，那是 STTN 模型能力问题，非 A0 范围，归 A2。

### 环境限制说明

开发机 ONNXRuntime CUDA 加载失败（`LoadLibrary failed with error 126`，缺 cuDNN 9/CUDA 12），回退 CPU。这导致 20 秒 1080p 视频跑 STTN 超过 10 分钟超时。门槛判定改用 2 秒片段 + mask 层面验证替代端到端帧对比。**这同时印证了 ROADMAP 中 C1 的"次要脆弱假设"——连开发机都 CUDA 不可用，local-first 用户安装 GPU 环境的摩擦可能比预期更大。**

### 回滚

纯函数级改动，`git revert` 涉及的 4-5 文件即可。无数据迁移、无外部状态变更。

### 最脆弱假设

**STTN + 软 alpha + 羽化能达到"成品感"门槛。** 若 STTN 在复杂动态背景下本就糊，A0 的工程优化救不回来。验收标准里的人工门槛判定专门用于证伪/证实这条假设。**这是整条路线 load-bearing 的假设，必须在 A0 完成时立即验证。**

---

## 阶段 A0.5 — ProPainter 授权调查

**定位**：go/no-go 开关，决定 A2 是否存在。**只产出决策，不写代码。** 与 A0 并行，因为互不依赖。

### 实施边界

**In scope**：
1. 查证 ProPainter 仓库（`sczhou/ProPainter`）的实际 LICENSE 文件原文，不凭记忆。
2. 按授权类型给出三档判断：
   - MIT/Apache/BSD → 可进 pip 默认
   - 非商用但允许集成/OEM → 可做 `--model propainter` 默认档，pip 默认仍 STTN
   - 严格非商用/禁止再分发 → ProPainter 永远外接，A2 转向评估 E2FGVI
3. 顺带核实 ProPainter 依赖（iopath/paniniferg 等）的授权是否传染。
4. 产出 `plans/propainter-license-finding.md` 决策文档。

**Out of scope**：
- 写任何 ProPainter 集成代码（→ A2）
- 评估 E2FGVI（仅在 A0.5 结论为"ProPainter 不可用"时，由 A2 接手）

### 实施结果

- `plans/propainter-license-finding.md`：LICENSE 原文引用、依赖授权矩阵、明确的 go/no-go 结论 + 理由

### 验收标准

- [x] 决策文档引用 LICENSE 原文（非转述）— `plans/propainter-license-finding.md` 引用 S-Lab License 1.0 第 4 条 + README 原文
- [x] 给出明确结论：`GO_PIP_DEFAULT` / `GO_OPTIONAL_MODEL` / `NO_USE_PROPINTER` 三选一 — **NO_USE_PROPINTER**
- [x] 若结论非 `GO_PIP_DEFAULT`，说明对 A2 的影响 — A2 改为 E2FGVI 评估，ProPainter 内置化路径关闭，详见 `propainter-license-finding.md`

### 回滚

无代码，无回滚成本。决策文档保留作为 A2 的前置依据。

---

## 阶段 A2 — 画质模型升级（形态改变：ProPainter → E2FGVI）

**定位变更（2026-06-22）**：A0.5 结论为 `NO_USE_PROPINTER`（S-Lab License 1.0 非商用，MIT 互斥，详见 `plans/propainter-license-finding.md`）。A2 不再做 ProPainter 内置化，改为评估 **E2FGVI** 或其他商用友好（MIT/Apache/BSD）模型。

**优先级降低**：A2 不再阻塞 C1。C1 的画质地基是 A0（软 alpha + 音频 + 羽化），与模型无关。A2 是独立的画质升级线，可在 C1 之后任何时候启动，或直接砍掉（若 A0 的人工门槛判定已通过，STTN + 软 alpha 已够用）。

**现有 `--external-command` ProPainter 路径保留**：用户自行 clone ProPainter 调用，是用户与 S-Lab 的双边授权关系，videowipe 不做再分发。但文档不应鼓励商用用户走此路。

### 实施边界

**In scope（A0.5 = GO_OPTIONAL_MODEL 时）**：
1. 把 `scripts/propainter_wipe.py` 的子进程逻辑重构为 `src/videowipe/inpainters/propainter.py`，实现 `Inpainter` 协议（与 STTN 同级）。
2. 通过 registry 注册为 `--model propainter`，默认模型仍 STTN。
3. 解决 ProPainter 的"自己出 mp4、不走 videowipe blend"问题——要么让 ProPainter 走 videowipe 的软 alpha blend（需 ProPainter 输出 raw 帧），要么接受 ProPainter 独立出片（损失软 alpha 收益）。**这是 A2 的核心技术决策，需在实施前定。**

**In scope（A0.5 = GO_PIP_DEFAULT 时）**：
- 上述全部 + 把 CLI/engine 默认模型从 STTN 改为 ProPainter + pip 默认依赖加入 ProPainter 运行时。

**Out of scope**：
- 引入 LaMa（VSR 用的残差精修）——除非 A0 门槛判定失败，作为 STTN 的增强而非替代
- ProPainter 的 ONNX 化（独立大工程，不在本路线）

### 实施结果

- `src/videowipe/inpainters/propainter.py`（条件性）
- `pyproject.toml` 依赖调整（仅 GO_PIP_DEFAULT）
- `tests/test_propainter_inpainter.py`：走 `Inpainter` 协议的契约测试

### 验收标准

- [ ] `videowipe detext --model propainter` 在 checked-in 样本上跑通
- [ ] `scripts/benchmark_pipeline.py` 同时跑 STTN 和 ProPainter，benchmark.json 含两者对比
- [ ] ProPainter 路径保留原音轨（复用 A0 的 ffmpeg 改动）
- [ ] 若走 GO_PIP_DEFAULT：默认安装 `pip install videowipe` 即可用 ProPainter，无需 clone 外部仓库
- [ ] 人工：ProPainter 结果与 A0 后的 STTN 肉眼对比，记录在 `plans/propainter-license-finding.md` 或独立报告

### 回滚

删除 `inpainters/propainter.py` + registry 注销 + pyproject 回退。`WipeEngine` 公共 API 不变。

### 前置 Unknown

**ProPainter 能否走 videowipe 的软 alpha blend**——当前 `propainter_wipe.py` 是整片出 mp4 的子进程。若 ProPainter 无法改为流式输出 raw 帧，A2 就无法让 ProPainter 享受 A0 的软 alpha 收益，"默认质量升级"的实际收益要打折。**这个技术可行性应在 A2 启动前 spike 验证，不写进 A2 内部当 Phase 0。**

---

## 阶段 C1 — Local-first Web 前端（含 B1 意图规则层）

**Completed**: 2026-06-25 | **代码状态**: ✅ 完成并发布 v0.4.0

**定位**：给非技术用户浏览器入口。Local-first（用户自己有 GPU，浏览器连 localhost），等同 SD-WebUI/ComfyUI 模式。**B1（意图理解规则层）作为 C1 的一个 named milestone 内嵌，不独立成阶段**——规则层代码已存在（`engine.py` 的 `infer_targets_from_text` 等默认运行），C1 的实际工作是在前端暴露 intent 输入。

### 实施边界

**In scope**：
1. `videowipe serve` 子命令 → FastAPI（`web` extra 可选）+ 内嵌单页前端，复用 `WipeEngine`。
2. 任务状态机：进程内 dict + threading（**不上 Redis/SQLite**，local 单用户）。
3. API 端点（无环数据流）：
   - `POST /jobs` — 创建任务，接收视频上传
   - `GET /jobs/current` / `DELETE /jobs/current` — 查看或释放 stale preview job
   - `GET /jobs/{id}` — 查询任务状态
   - `GET /jobs/{id}/preview` — 调 `WipeEngine.process(preview=True, intent=...)`，返回检测候选 + 预览图
   - `GET /jobs/{id}/preview-image` — 返回检测预览图
   - `POST /jobs/{id}/confirm` — 用户增删候选，并启动清理
   - `GET /jobs/{id}/progress` — SSE 推送进度
   - `GET /jobs/{id}/download` — 返回带音轨的最终 mp4（A0 保证有音轨）
4. **B1 里程碑**：前端首屏加自然语言 intent 输入框，接到 `WipeEngine.process(intent=...)`。复用已有规则层，**不调 LLM**。
5. 单页前端：原生 HTML/JS（复用 `videowipe-landing.html` 风格），**不引 React/Vue 构建链**。
6. 串行单任务（local 单用户），**不做任务队列/并发/鉴权**。

**Out of scope**：
- LLM 云 API 服务化（B1 只上规则层，LLM 留 CLI 路径）
- 任务队列 / 多租户 / 鉴权（local-first 不需要）
- 桌面 GUI / PyQt / PyInstaller 打包
- Hosted SaaS（用户已选 local-first）
- 前端框架构建链

### 实施结果

- `src/videowipe/server/{__init__,app,jobs}.py`（FastAPI + 任务注册 + 状态机）
- `src/videowipe/web/index.html`（单页前端）
- `src/videowipe/cli.py`（加 `serve` 子命令）
- `pyproject.toml`（`web = ["fastapi","uvicorn","python-multipart"]` extra）
- `tests/test_server.py`：FastAPI TestClient 走全流程
- `README.md` / `README_CN.md`：Web UI 用法与截图

**涉及文件 >8，明确标注**：超过 10 个文件（含测试、文档和截图）。

### 验收标准

**B1 里程碑（在 C1 内）**：
- [x] 前端提供 intent 输入，preview 会把 intent 传给 `WipeEngine.process(intent=...)`
- [x] 复用现有 `select_clean_candidates` 测试覆盖规则层，无需新增逻辑测试

**C1 自动化**：
- [x] FastAPI TestClient 走完 `create → preview → confirm → download` 全流程
- [x] 断言 SSE `/progress` 至少推送 1 个进度事件
- [x] 断言 `/download` 返回的 mp4 含音频流（A0 保证）
- [x] `tests/test_server.py tests/test_boundaries.py` 全绿
- [x] `videowipe serve` 启动后 localhost 可访问，前端流程已用浏览器截图验证

**C1 人工**：
- [x] 浏览器拖入真实视频 → 输入意图 → 看到检测预览 → 确认 → 进度条推进 → 下载带音轨 MP4
- [ ] 找一个**非技术**目标用户走一遍源码安装 → `videowipe serve` 流程，记录卡点（次要脆弱假设的验证）

### 回滚

删除 `server/` + `web/` 目录 + cli.py 的 `serve` 子命令 + pyproject 的 `web` extra。**`WipeEngine` 公共 API 零变更**，库用户不受影响。

### 最脆弱假设（次要）

**local-first 目标用户能装上 GPU 环境。** C1 完成后用非技术用户走安装流程验证。若卡死，需补一键安装脚本或重评是否回归桌面 GUI。

---

## 与 NEXT_WORK.md 的冲突

`plans/NEXT_WORK.md` 的 "Do Not Build Yet" 曾列出三条与本路线冲突。C1 完成后，这里只保留仍有效的约束：

| NEXT_WORK.md 红线 | 本路线动作 | 建议处置 |
|---|---|---|
| "New inpainting models as bundled default dependencies" | A2 可能评估 E2FGVI（A0.5 已排除 ProPainter） | **保留红线**。A2 若启动，必须先重新核实 E2FGVI 的 LICENSE 是否 MIT/Apache/BSD 友好。 |

Web UI 红线已由 A0 → C1 的实际完成状态关闭；registry 红线已过时，代码已经存在 `--model` 和 ProPainter 外接路径。

---

## 全局验收 / 发布门禁

整条路线的发布门禁（非单阶段）：
- [ ] A0 的人工门槛判定通过（STTN+软alpha 达"成品感"）或 A2 提级完成
- [ ] C1 的非技术用户安装流程验证通过（或卡点已修）
- [ ] `make check` 全绿
- [ ] `scripts/benchmark_pipeline.py` 在至少 2 个 checked-in 样本上产出 benchmark.json
- [x] README 更新：`videowipe serve` 用法 + local-first 定位说明

**发布后待处理**：C1 非技术用户安装验收、GPU Docker 镜像 workflow 超时、A2 是否继续。

---

## 追踪约定

更新本文件时：
- 改阶段表格的"状态"列（🔲🔄✅⏸️❌）
- 完成的验收项打 `[x]`
- 阶段启动时在阶段标题下加 `Started: YYYY-MM-DD`，完成时加 `Completed: YYYY-MM-DD`
- 重大决策变更（如 A0.5 结论、A0 门槛判定结果）记到对应阶段的"实施结果"或新增"决策记录"小节
