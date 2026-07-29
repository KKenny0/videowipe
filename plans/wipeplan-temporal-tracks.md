# VideoWipe WipePlan 与时序轨道实施计划

Date: 2026-07-26
Status: Accepted — Phase A 已通过验收，Phase B 可开始实施

## 结论

推荐实现一个可序列化、可审阅、可确定性执行的 `WipePlan v1`，把当前“一段视频压成一张静态 mask”的流程改成：

```text
视频 + intent
      │
      ▼
现有候选检测（保留采样帧号与逐帧命中事实）
      │
      ▼
WipePlan：track + remove/keep + 生效区间 + 精确空间 mask
      │
      ├── 人 / 本地 agent 只修改语义与时间，不生成像素
      ▼
计划校验（源视频、schema、mask 资产、区间）
      │
      ▼
STTN：union mask 决定处理区域，frame mask 决定每帧是否混合
```

首版轨道不是通用运动目标跟踪器，而是**屏幕叠加物轨道**：每条轨道使用一张稳定空间 mask，加若干半开时间段 `[start_frame, end_frame)`。这是当前失败事实所需要的最小模型：它能关闭字幕空窗误擦，也能把持续存在的顶部台标/署名单独标为 `keep`。

## 当前事实与目标

当前 `detect_clean_candidates()` 已经逐采样帧运行 DBNet，但最终只保留跨帧频率聚合后的 `CleanCandidate.mask`；`WipeEngine.process()` 把所有 selected candidate 合成一张 `auto_mask.png`；`STTNInpainter` 在所有视频帧重复使用该静态 mask。

正式事实基线为：

| 指标 | 当前值 |
|---|---:|
| remove union Jaccard | 0.170858 |
| remove union Boundary F | 0.423151 |
| keep 对象预测覆盖 | 0.936564 |
| 无 remove 帧误擦面积 | 0.066089 |
| 选择意图匹配率 | 0.500000 |

本计划的目标不是更换检测或填充模型，而是让系统拥有一个正确的中间表示：

1. 每个候选成为独立、可寻址的 track。
2. track 明确是 `remove` 还是 `keep`。
3. track 只在有证据的时间段生效。
4. 人或 agent 可以修改计划，执行器只做校验和确定性渲染。
5. 旧静态 mask、手工 mask、`detext` 与现有 SDK 调用保持兼容。

## 不做什么

- 不引入 SAM 2、ByteTrack、DeepSORT、OpenCV contrib tracker 或新运行时依赖。
- 不做运动物体的逐帧 mask propagation；移动水印需要真实失败证据后再进入 schema v2。
- 不让 LLM 生成 mask、坐标或未经校验的执行参数。
- 不做可拖拽时间线编辑器；Web 首版只展示区间并允许整条 track 的 remove/keep 切换。
- 不更换 STTN，不评估新 inpainting 模型，不扩充真实用户数据集。
- 不删除 `clean_candidates.json` 与 `auto_mask.png`；它们保留一个兼容周期。

最小替代方案是只给 `clean_candidates.json` 增加起止帧。它不包含源视频身份、schema 校验、精确候选 mask 或 SDK 执行契约，也无法结束 Web 改选时的 bbox 近似，因此不采用。

## WipePlan v1 契约

### 计划级字段

| 字段 | 约束 |
|---|---|
| `kind` | 固定为 `wipe_plan` |
| `schema_version` | 首个公开版本固定为 `1` |
| `source` | 视频 basename、SHA-256、宽高、fps、frame_count |
| `request` | intent、targets、regions、detect_mode、ocr 的实际解析值 |
| `temporal_resolution` | 最大采样间隔的 frame 数与秒数、理论最大边界误差 |
| `mask_asset` | 同目录 `wipe_plan_masks.npz` 的文件名与 SHA-256 |
| `tracks` | 非重复 track 列表 |
| `warnings` | 粗时间分辨率、静态 fallback 等可执行但必须可见的限制 |

JSON 保持 agent 可读；精确二值空间 mask 存入同目录、压缩且 `allow_pickle=False` 的 `wipe_plan_masks.npz`。每条 track 通过自身 ID 读取对应数组。这样不把数万行 mask 数字塞进 JSON，也不退化成 bbox。

### Track 字段

| 字段 | 约束 |
|---|---|
| `id` | 沿用候选 ID，在同一计划内稳定且唯一 |
| `type` / `label` | 沿用当前候选语义 |
| `action` | 仅允许 `remove` 或 `keep` |
| `bbox` | 供人和 UI 快速理解，不作为 mask 真值 |
| `confidence` | 检测置信度 |
| `presence_fraction` | 成功采样帧中命中该候选的比例 |
| `decision_reason` | 默认规则、intent、region、agent 或人工计划的决策来源 |
| `segments` | 已排序、互不重叠的半开帧区间 |
| `mask_key` | `wipe_plan_masks.npz` 中的精确 mask key |

### 强制校验

- source SHA-256、宽高和 frame_count 必须与待处理视频一致。
- schema、track ID、action、bbox、mask shape/dtype、asset SHA-256 必须合法。
- segment 必须满足 `0 <= start < end <= frame_count`，且排序、不重叠。
- JSON 与 NPZ 必须位于同一计划目录；拒绝绝对路径、父目录穿越和逃逸 symlink。
- 没有 `remove` track 时返回明确输入错误，不启动模型。
- 校验失败由库抛异常，CLI/Web 转换成用户可读错误。

## 轨道生成规则

1. 修改私有采样函数，使其保留真实 frame index；现有均匀采样策略和 detect mode 数量不变。
2. 保留每个成功采样帧的 `TextBox`。现有聚合候选仍负责空间 mask、分类和兼容输出，不另建第二套检测器。
3. 对每个候选，用其 bbox/mask 与逐采样帧 `TextBox` 的空间相交判定 presence；因此一个 `CleanCandidate` 直接对应一条 track，不引入通用身份关联。
4. 每个未采样帧采用最近采样帧的 presence 状态；状态切换点位于相邻采样帧的中点。理论最大边界误差写入计划。
5. 连续 active 帧压缩为半开 segment。用户指定 region、固定 logo 和无法取得逐帧证据的 fallback track 使用全视频 segment，并写 warning。
6. 安全默认规则：顶部区域中心位于 `0.30H` 以上、`presence_fraction >= 0.80`、且没有显式顶部 remove 指令的持久叠加物，默认 `keep`。这条规则直接针对当前 Mango TV 台标、署名和 DOM logo 被当作字幕删除的事实。
7. 决策优先级固定为：载入计划中的人工 action > 显式 region/track 或 agent 选择 > 顶部持久叠加物安全规则 > 当前 `default_remove`。
8. 当最大采样间隔超过 2 秒时，计划仍可生成和执行，但必须写入 warning，并透传到 `WipeResult.warnings`；首版不假装具备精细时间边界。

## 执行语义

- 新增 `WipeEngine.plan(request) -> WipePlan`，只检测和写计划，不加载 inpainting 模型。
- `WipeRequest` 新增可选 `plan`，接受 `WipePlan` 或计划 JSON 路径；`mask` 与 `plan` 互斥。
- `clean` 默认路径内部先生成 WipePlan，再执行它；现有调用者不需要改代码。
- `auto_mask.png` 改为所有 remove track 的空间 union，仅用于兼容和总览；它不再是 STTN 的逐帧执行真值。
- `InpaintJob.mask` 继续保存 union mask，供 STTN 计算需要处理的纵向 crop；新增一个可选 frame-mask callable，STTN 混合第 N 帧时读取第 N 帧 mask。
- STTN 仍按现有 segment 批量推理，不为 inactive 帧重新设计模型循环；只在最终 blend 时按时间关闭 mask。首版优先正确性，不承诺减少推理耗时。
- file-based external inpainter 只有在所有 remove track 都覆盖全视频时才能接受 WipePlan；真正 temporal 的计划明确报不支持，不能静默压平成静态 PNG。

## 阶段 A：核心 WipePlan 与 temporal execution

这是一个不可再拆的核心阶段：只交付 schema 而不能执行会形成第二份影子事实，只改 STTN 而没有可序列化计划又无法审阅。预计涉及约 12 个文件，超过 8 个，但都处在同一条数据流：

- 新增 `src/videowipe/plan.py`
- 修改 `src/videowipe/detect.py`
- 修改 `src/videowipe/api.py`
- 修改 `src/videowipe/engine.py`
- 修改 `src/videowipe/inpainters/base.py`
- 修改 `src/videowipe/inpainters/sttn.py`
- 修改 `src/videowipe/__init__.py`
- 修改 `scripts/eval_clean_detection.py`
- 新增 `tests/test_wipe_plan.py`
- 修改 `tests/test_sdk_api.py`
- 修改 `tests/test_detection_eval.py`
- 更新 `plans/fact-baseline.md`

阶段 A 合并后，SDK 用户和 agent 已能生成、检查、修改并执行计划；即使阶段 B 永不实施，系统仍处于可用状态。

### 自动验收

- WipePlan JSON/NPZ round-trip 字节稳定，NPZ 使用 `allow_pickle=False`。
- 非法 schema、重复 track、越界/重叠 segment、错误 asset hash、错误源视频全部拒绝。
- 合成 60 帧视频中，采样 presence 的时间边界误差不超过相邻采样间隔的一半。
- 同一 track 的静态空间 mask 在计划 round-trip 后逐像素一致。
- STTN 跨自身 `gap` 分段后仍使用全局 frame index，active/inactive 边界没有 off-by-one。
- 无 plan、手工静态 mask 与 `detext` 的现有测试全部保持通过。
- temporal WipePlan 交给 external/file-based backend 时明确失败；全时段计划仍兼容。
- `make check` 全绿。

### 事实基线门槛

检测事实报告因预测语义改变，正式升级为 report schema v2；WipePlan 自身仍是 schema v1。clean commit 上运行 `make fact-baseline-formal`，并同时保留旧静态结果作为对照。

| 指标 | 阶段 A 门槛 |
|---|---:|
| remove union Jaccard | `>= 0.160000` |
| remove union Boundary F | `>= 0.410000` |
| keep 对象预测覆盖 | `<= 0.100000` |
| 无 remove 帧平均误擦面积 | `<= 0.020000` |
| 任一无 remove 帧误擦面积 | `<= 0.030000` |
| 选择意图匹配率 | `>= 0.900000` |
| 分类语义匹配率 | 不低于当前 `0.500000` |

同时逐张比较 temporal mask、indexed annotation 与 `input/detext_examples/mask/*.png`；旧 Golden 继续只作为 calibration，不升级为质量真值。

#### Phase A 验收修订（2026-07-28）

正式 schema v2 事实报告中，七项门槛有五项通过；“无 remove 帧平均误擦面积”和“任一无 remove 帧误擦面积”未达到原阈值。两项失败均来自 `others.mp4` 第 361 帧：该空窗短于 `balanced` 模式的相邻采样间隔，且前后采样点都观察到字幕存在。WipePlan v1 只根据采样证据做最近状态插值，因此没有足够信息恢复这个未被观测到的空窗。

产品验收决定：

- 保留原始阈值、实测值和“未达”事实，不把结果重写为通过。
- 将这两项认定为 **WipePlan v1 已知时序分辨率例外**，不再阻塞 Phase A；对采样证据能够观察到的空窗，预测仍必须正确关闭。
- 原阈值继续保留为引入逐帧 mask propagation 后的 WipePlan 后续版本目标，不为了贴合当前三个样例而调宽阈值。
- 后续复跑不得低于当前五项已通过指标，也不得让已可观察空窗重新产生误擦。
- 本修订只处理质量门槛的范围冲突，不豁免代码审查发现；confirm 动作映射、执行时精确 mask 强制校验和逃逸 symlink 校验已在 `ff1f207` 收口，并通过针对性测试、全量 `make check` 与正式事实基线复验。Phase A 整体已验收。

## 阶段 B：CLI、Web 与 AI-native 入口

阶段 B 只消费阶段 A 已稳定的计划，不再发明另一套时间线数据。

- `videowipe clean VIDEO --preview` 继续作为计划生成入口，同时产出 `wipe_plan.json` 与 `wipe_plan_masks.npz`。
- 新增 `--plan PATH` 执行已经审阅或由 agent 修改的计划；不新增第二个 `plan` CLI 子命令。
- Preview API 返回 tracks、segments、actions 和计划 artifact；兼容 candidates 字段保留一个周期。
- Web 候选列表显示 track 生效时间范围，并允许整条 track remove/keep；不做拖拽时间线。
- confirm 直接更新计划 action 并执行，删除当前 `_mask_from_selected_bboxes()` 近似路径。
- README/README_CN 增加一条 SDK agent 流程：生成计划 → 只编辑 action/segments → 校验执行。VideoWipe 不绑定任何 LLM 或云服务。

预计修改：

- `src/videowipe/cli.py`
- `src/videowipe/server/app.py`
- `src/videowipe/web/index.html`
- `tests/test_server.py`
- `tests/test_boundaries.py`
- `README.md`
- `README_CN.md`

### 阶段 B 验收

- CLI 能生成计划、执行未修改计划、执行修改 action/segment 的计划，并拒绝源视频不匹配。
- Web 默认选择不变时与计划 mask 完全一致；改选后不再退化为 bbox mask。
- 现有 upload → preview → confirm → progress → download 与音频保留测试继续通过。
- 浏览器人工检查 track 区间可理解，窄屏不遮挡，控制台无错误。

## 风险、失败处理与回滚

最脆弱假设是：VideoWipe 当前目标主要是屏幕固定的字幕、台标和署名，因此“静态空间 mask + 时间段”足够。如果真实素材出现移动水印、跟随人物的贴纸或透视变化文字，v1 会产生空间漂移。届时新增 schema v2 的 keyframe geometry/propagation；不在 v1 预埋未使用接口。

长视频的固定采样数会降低时间精度。v1 通过 `temporal_resolution` 和 warning 暴露事实，不偷偷声称精确；真实用户素材证明 2 秒警戒线不够后，再评估 ROI 密集检测或 mask propagation。

阶段 A、B 都没有数据库迁移、外部状态或新依赖。回滚各自 commit 即可；旧静态 mask、手工 mask和现有 API 始终保留。生成的计划、NPZ、报告和预览继续位于忽略目录，不进入 git。

## 依赖、发布与明确延后

- 新依赖/API key/外部服务：无。
- SAM 2：其官方 video predictor 支持以 point/box/mask prompt 建立对象并传播 mask，但要求独立 Torch/模型运行栈；只有移动 overlay 证据出现时再评估。
- ByteTrack：借鉴“检测事实与轨道关联分离”的思想，不引入其面向密集 MOT 的实现。
- OpenCV MultiTracker：官方文档将其列为 legacy，且其多目标实现是逐对象独立跟踪；不作为新公共依赖。
- 两阶段均完成且正式基线通过后，建议作为一个向后兼容的 minor `v0.6.0` 候选；本计划不自动改版本、发布或推送。
- 真实用户长视频分布和移动 overlay 占比由后续试用收集负责，不阻塞 v1，但它们是是否进入 WipePlan v2 的唯一升级依据。

参考实现与边界：

- [Meta SAM 2 官方仓库与 video predictor](https://github.com/facebookresearch/sam2)
- [ByteTrack ECCV 2022 论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)
- [OpenCV 官方 MultiTracker 文档](https://docs.opencv.org/master/d5/d07/tutorial_multitracker.html)
