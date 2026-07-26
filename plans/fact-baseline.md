# VideoWipe 事实基线 v1

Date: 2026-07-26

## 目的与边界

本计划建立可复现的**检测事实基线**，而不是产品通过门槛：三条现有视频是 calibration set，用来校准评估协议并暴露当前失败，不能外推为真实用户素材、商业质量或模型泛化结论。它们的合法、清晰来源说明仍未知；本计划不新增来源声明，也不把它们称为公开 benchmark。

本阶段只增加固定帧 `remove/keep` 标注、检测评估、预览和审阅记录。不会改运行时 API、替换模型、引入 SAM 2、实现 WipePlan、做全量视频数据集、发布、推送或改版本号。README 不因本计划修改。

## 已知工程事实与未知质量

- v0.5 的 SDK-first 嵌入门槛已完成。
- A0 只证明音频、软 alpha、进度和 mask 数学成立。
- mask-level proof 不是“成品感门槛通过”。
- 端到端填充质量、时序稳定性、误擦率和用户接受度尚未通过验证。

因此，在本基线完成前不启动默认模型升级或大型上层产品。完整 inpainting 画质也不在本阶段测量：当前环境速度会污染判断，且没有 clean-plate ground truth；检测事实完成后应另行设计输出质量基线。

## 数据与协议

`input/detext_examples/fact_baseline.json` 固定每条视频五个实际帧号（共 15 张 indexed PNG）。帧号一经写入，评估器不得按时间或采样策略漂移。每张 indexed PNG 使用：`0` 为背景，`1..N` 为 manifest 中稳定对象 ID；每个对象注明 `type`、`action: remove|keep` 和说明。字幕、台标、顶部署名分为独立 ID，标注贴合可见叠加内容，而非整条字幕带。

Jaccard 使用 DAVIS/YouTube-VOS 的区域定义；Boundary F 按 DAVIS 2017 官方 `_seg2bmap`、`ceil(bound_th × diagonal)` 与 disk 膨胀实现。manifest 额外表达标准数据集没有的 `remove/keep` 意图。参考：[DAVIS Jaccard](https://interactive.davischallenge.org/docs/metrics.jaccard/)、[DAVIS 2017 metrics](https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py)、[YouTube-VOS evaluation](https://youtube-vos.org/dataset/rvos/)。

评估命令：

```sh
make fact-baseline
```

它显式需要 DBNet 权重；首次下载失败必须报错，不能静默换成另一种检测逻辑。评估器与生产检测共用 `dbnet_default` 路径；当前支持范围固定为 `opencv-python-headless>=4.5,<5`，OpenCV 5 会明确拒绝运行，而不是切换到评估器专用逻辑。单元测试不联网，真实基线不进入 CI。报告和预览在忽略的 `result/fact-baseline/`，不进 git。

```text
视频 + manifest + indexed masks
              │
              ▼
      detection evaluator
              │
      ┌───────┴────────┐
      ▼                ▼
结构化 metrics      可视检查图
      │                │
      └───────┬────────┘
              ▼
       本文件的当前事实
```

## 报告约定与人工审阅

报告以项目仓库根为准记录 Git HEAD、tracked worktree 是否干净、未跟踪路径、工作区状态哈希、评估器源哈希、VideoWipe/Python/OpenCV 版本、detect/OCR mode、稳定逻辑名称加内容的输入/indexed 标注/legacy calibration 哈希；不再受调用者当前目录或外层 Git 环境影响，也不把 dirty worktree 误归因为 HEAD 可复现。`make fact-baseline-formal` 用于正式基线：要求 tracked worktree 干净，并要求 manifest、视频、indexed 标注和参与计算的 legacy mask 都已跟踪；与基线无关的未跟踪文件（例如本地 `NEXT_WORK.md`）会被记录但不阻塞。

逐帧报告 selected remove union 的 Jaccard/F、keep 误伤、无 remove 目标帧误擦，并独立报告可见对象上的候选分类、选择决策和局部对象质量。宏平均明确区分“有 remove 的帧的 union 质量”与“仅可见对象的局部质量”；一个正确的 remove 对象不会因另一个正确对象而被当作 false positive。旧静态 Golden IoU 仅以 `legacy_calibration_metric` 输出，作为 regression calibration，绝不作为质量事实。

首次报告只建立当前快照，不设“产品通过”阈值。运行后须逐张检查 15 张预览并在此处写回：漏掉内容、误擦内容、被误分类为字幕的 Logo/署名、无目标帧的静态 mask 误擦、以及只能归因于这三条 calibration 样例而不能外推的数字。

## 首次运行事实（2026-07-26）

这是首份 schema v1 报告；审查期间的内部草稿数字不构成已发布 schema。此次运行使用 `balanced` / OCR off、Python 3.10.14、OpenCV 4.13.0，`detector_execution_path` 为生产默认的 `dbnet_default`。本节数字已从 clean tracked worktree 使用 `make fact-baseline-formal` 正式复现；生成报告绑定其记录的 Git HEAD、输入与评估器哈希。与基线无关的本地 `NEXT_WORK.md` 未进入提交，也不影响正式性。

15 张误差叠图已逐张审阅。宏平均按明确单位为：有 remove 目标帧的 union Jaccard **0.170858**、union Boundary F **0.423151**；可见 remove 对象的局部 Jaccard **0.236862**、局部 Boundary F **0.518778**；可见 keep 对象预测覆盖 **0.936564**；无 remove 目标帧误擦面积 **0.066089**；可见标注对象的分类语义匹配率 **0.500000**、选择意图匹配率 **0.500000**。

- 当前检测器能把多数底部字幕归为 `subtitle`，但生成的是明显宽于字形的横向候选带；这同时造成大量漏擦和带外误擦。以上数字只描述这三条样例，不能外推为英语或任何字幕的一般性能。
- Chinese1 的左上 Mango TV 台标在全部五帧都被 `subtitle` 候选覆盖；它是应保留的水印，却有约 93.4% 的标注像素被预测 mask 覆盖。
- Others 的顶部署名、译者署名和 DOM logo 也被 `subtitle` 候选覆盖；存在保留对象的四帧中，keep 覆盖为 80.5%–99.4%。这比单看字幕 IoU 更直接说明当前自动选择不安全。
- 两个固定无 remove 目标帧仍有 3.1882% 与 10.0293% 的预测面积，说明静态候选会在空窗误擦。该数字只描述这两个 calibration 帧。
- 旧静态 Golden IoU 为 Chinese1 0.211894、English1 0.429897、Others 0.388101；它们只保留为 `legacy_calibration_metric`，不构成产品质量结论。

此轮只形成当前三条样例的失败快照。它不证明修复输出质量、时序稳定性、真实误擦率或用户接受度；是否扩充到 20–30 条有合法来源的用户素材，需在此后单独决定。

## 回滚与后续决策

没有数据迁移：文档、manifest、评估器和标注可按阶段独立撤销；`result/` 生成物可安全清理。若这三条样例过于相似，schema 与评估器仍有效，但结论必须停留在“当前三条样例”。扩大到 20–30 条合法用户素材是在本阶段完成后才作出的独立决定。
