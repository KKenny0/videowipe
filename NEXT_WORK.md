# 当前工作入口

核对日期：2026-10-15。最新发布仍为 v0.11.1；当前 main 已包含 #16 的 Agent 选择诊断、#17 的 macOS 安装 CI 及 #22 的帧身份工程修复。当前优先级是清理效果：去掉该去的、保留该留的，减少达到可用结果的修改。

## 开始工作

1. 先核对所在工作树的 `git status --short --branch -uall`、HEAD 和本地差异，保留已有草稿、jobs、素材及输出。
2. 按下表选择当前 Issue，读取正文、评论和原生阻塞关系。本文是入口；具体规格与实时状态以 GitHub 为准。
3. 效果验收先读[记录流程](scripts/CLEANUP_EFFECT_ACCEPTANCE.md)，复用已绑定输入、计划及输出身份的证据。执行成功、抽帧检查和正常播放签收分别记录。
4. 交付前运行 `make check`。检测/蒙版变更还需 fact、decision 基线和固定 PNG 对照；生成产物另存忽略目录。提交、推送、PR、发布及远端状态更新按当前会话授权执行。

## 已发布能力与验收边界

[v0.11.1 Release](https://github.com/KKenny0/videowipe/releases/tag/v0.11.1) 已发布。下列能力已实现，不再作为新功能重复建设：

| 能力 | 代码 / 使用入口 |
| --- | --- |
| Inpainter 协议及 sttn、external、propainter 注册 | [inpainters](src/videowipe/inpainters/__init__.py)、[引擎](src/videowipe/engine.py) |
| 本地 Web 上传、候选审阅、推荐试擦、全片导出及播放 | [Web 接口](src/videowipe/server/app.py)、[工作台说明](docs/cleanup-workspace_CN.md) |
| SDK 规划进度、协作取消；Web 保存、取消、重试与恢复 | [SDK](src/videowipe/engine.py)、[任务状态](src/videowipe/server/jobs.py) |
| 时段与保护编辑、局部空间蒙版及有界预测复用 | [WipePlan](src/videowipe/plan.py)、[审阅](src/videowipe/server/review.py)、[预测缓存](src/videowipe/inpainters/prediction_cache.py) |
| 自动 MPS 选择与空目标段带跳过 | [后端](src/videowipe/backends.py)、[STTN](src/videowipe/inpainters/sttn.py) |

Spec #3 的四段历史样片已获正常播放签收，macOS 安装与实际运行已有发布验收。这些结果只覆盖当时的版本、输入和验证范围，不代表所有视频可用。#17 现已补齐持续 macOS arm64 安装 CI；安装冒烟不等于 MPS 全片效果验收。旧四例无需再次要求用户人工验收。

## 当前任务

#11、#12、#13、#16、#17、#18 已合入并结项。#12 的英文／多语默认输出获用户完整播放接受，中文默认输出明确失败；#13 只覆盖指定一次框编辑的接受，固定 ROI 工程失败及部分未知结果保留。结项不是所有输入可用的证明。

| 顺序 | Issue | 当前边界 / 下一步 |
| --- | --- | --- |
| P0 待播放 | [#22 中文字幕多处闪字](https://github.com/KKenny0/videowipe/issues/22) | [#25](https://github.com/KKenny0/videowipe/pull/25) 已合入，修复错误 seek 的帧身份错配；506 项本地测试、四例 1,809 帧试擦／全片一致性通过。新中文输出仍需正常速度完整播放，票保持开放；英文／多语视频字节与已接受输出相同。长片顺序采样有额外解码成本，不以蒙版或抽帧检查结项。 |
| 并行 | [#15 总成本](https://github.com/KKenny0/videowipe/issues/15) | 既有默认成功、修改后成功及中文失败的机器计时可关联原产物；真人操作／审阅时间、可用结果总耗时和严格冷／热／缓存分组未测，保持开放。按[观察流程](docs/cleanup-user-study_CN.md)前瞻采集，不倒推历史真人时间。 |
| 人工依赖 | [#14 五位用户](https://github.com/KKenny0/videowipe/issues/14) | 等待负责人提供五位真实参与者和授权素材。观察流程与空模板已准备；没有真实试用数据，内部检查不能替代。 |
| 完成 | [#16 Agent 来源](https://github.com/KKenny0/videowipe/issues/16) | [#24](https://github.com/KKenny0/videowipe/pull/24) 已合入，SDK／CLI／保存诊断区分六状态及来源；合法空选成功，失败本地回退。不是视觉 Agent 或跨端任务协议。 |
| 完成 | [#17 macOS CI](https://github.com/KKenny0/videowipe/issues/17) | [#23](https://github.com/KKenny0/videowipe/pull/23) 已合入，macOS arm64 与 Ubuntu 候选包安装检查持续运行，Windows／Ubuntu 测试保留；不代表 Intel 或真实 MPS 全片导出验收。 |
| 历史完成 | [#11](https://github.com/KKenny0/videowipe/issues/11)、[#12](https://github.com/KKenny0/videowipe/issues/12)、[#13](https://github.com/KKenny0/videowipe/issues/13)、[#18](https://github.com/KKenny0/videowipe/issues/18) | 复用已绑定输入、计划、输出及用户判断的记录；保留失败与未知范围，不重复开发已有能力或重做历史签收。 |

[#19 残影诊断](https://github.com/KKenny0/videowipe/issues/19) 已 CLOSED。诊断未改生产算法：保持当前默认输入尺寸，不重做尺寸/时间上下文实验。新动态片段的“基本可用、有一些涂抹”判断只覆盖受控手动目标。原截图的“只擦半行”仍缺原片、版本和操作路径，继续未验证。旧中文输入 seek 偏移与预测残影分别处理。

本地证据入口（均为忽略产物，不要求新 clone 自带）：

- `result/effect-acceptance/default-v2/report.json`：四条历史真实/合成案例的统一记录，人工状态保持 unverified。
- `result/effect-acceptance/screenshot-default/reviewed.json`、`screenshot-corrected/reviewed.json`：静态截图探针及一次选择修正；不是原视频复现。
- `result/effect-acceptance/safety-fix/`：原误选修复验证及残影分层证据。
- `result/effect-acceptance/residue-diagnosis-20261006/REPORT.md`：#19 结论、复跑及最终证据入口。新素材许可和身份读同目录 `material-valid.json`；优先最终 manifest，早期 `material.json` / `static.mkv` 是无效对照。
- `result/effect-acceptance/package-12-13-15-20261011/`：默认证据缺口、一轮框编辑结果、用户确认及阶段计时；用户确认绑定指定输出，技术限制保留。
- `result/effect-acceptance/delivery-11-18-20261011/`：#11/#18 的历史本地交付验证，不替代播放验收。

源素材私人路径仅查本地 manifest，不写进公共文档或自动上传。

## 尚未实施的方向

可选 Agent 视觉理解、受约束的自然语言时段编辑、自动残留/闪烁判断和自动重擦仍需独立规格及证据。网页已有代表帧和手工时段编辑不等于这些 Agent 能力已完成。更换默认模型、云服务及通用聊天入口不在当前八票范围。

## 历史规划

保留[三次交付](plans/PRODUCT_EXPERIENCE_DELIVERIES.md)、[短片试擦](plans/SHORT_TRIAL_MILESTONE.md)、[速度方案](plans/STTN_INFERENCE_SPEED.md)中的原始取舍与证据。文中的“下一步”“未提交”“未签收”属于各自日期的历史状态，不能覆盖本文及对应 Release / Issue 的后续结论。历史性能只适用于其设备、输入、参数和测量范围。

某些旧工作树另有未跟踪的 `NEXT_WORK.md`、`AI_NATIVE_PRODUCT_DIRECTION.md` 及进度回调/Phase C 草稿。它们保留为本地历史，不作为新 clone 的必需依赖；其中注册器、Web、进度取消旧待办已被上述实现取代。保留这些文件、jobs 和素材，不为整理入口清理工作树。
