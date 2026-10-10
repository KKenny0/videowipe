# 当前工作入口

核对基准：v0.11.1 / `1982331`。GitHub 状态核对日期：2026-10-11。当前优先级是清理效果：去掉该去的、保留该留的，减少达到可用结果的修改。

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

Spec #3 的四段历史样片已获正常播放签收，macOS 安装与实际运行已有发布验收。这些结果只覆盖当时的版本、输入和验证范围，不代表所有视频可用，也不代表 macOS CI 已完成。旧四例无需再次要求用户人工验收。

## 当前任务

#11、#18 的实现与本地验收已完成，提交、合入及结项状态见各自 Issue；后续实施从 #12 继续。#12–#17 在核对时均 OPEN。先完成效果证据，再推进辅助工程；关联不代表必须重做已完成上游。

| 顺序 | Issue | 当前边界 / 下一步 |
| --- | --- | --- |
| 完成 | [#11 清理效果记录](https://github.com/KKenny0/videowipe/issues/11)、[#18 统一入口](https://github.com/KKenny0/videowipe/issues/18) | 实现与本地验收完成：验收 runner、误选保护、记录契约、当前入口和历史标记。交付与结项凭据见票据。 |
| 2 | [#12 默认流程](https://github.com/KKenny0/videowipe/issues/12) | 用已许可的新真实素材执行默认选择→推荐试擦→完整导出，并单独记录正常播放判断；已有手动目标结果不能替代。 |
| 3 | [#13 一轮纠正](https://github.com/KKenny0/videowipe/issues/13) | 从本轮默认失败出发，一次保存必要编辑后重试；保留前后身份、耗时及剩余失败。 |
| 4 | [#14 五位用户](https://github.com/KKenny0/videowipe/issues/14)、[#15 总成本](https://github.com/KKenny0/videowipe/issues/15) | #14 需要负责人提供真实参与者；#15 复用计时，只把真正可用结果记作成功，人类思考时间须实测。 |
| 5 | [#17 macOS CI](https://github.com/KKenny0/videowipe/issues/17)、[#16 Agent 来源](https://github.com/KKenny0/videowipe/issues/16) | 尚待相应票据实施；安装保障、决策来源与回退原因记录不能抢占效果验证。 |

[#19 残影诊断](https://github.com/KKenny0/videowipe/issues/19) 已 CLOSED。诊断未改生产算法：保持当前默认输入尺寸，不重做尺寸/时间上下文实验。新动态片段的“基本可用、有一些涂抹”判断只覆盖受控手动目标。原截图的“只擦半行”仍缺原片、版本和操作路径，继续未验证。旧中文输入 seek 偏移与预测残影分别处理。

本地证据入口（均为忽略产物，不要求新 clone 自带）：

- `result/effect-acceptance/default-v2/report.json`：四条历史真实/合成案例的统一记录，人工状态保持 unverified。
- `result/effect-acceptance/screenshot-default/reviewed.json`、`screenshot-corrected/reviewed.json`：静态截图探针及一次选择修正；不是原视频复现。
- `result/effect-acceptance/safety-fix/`：原误选修复验证及残影分层证据。
- `result/effect-acceptance/residue-diagnosis-20261006/REPORT.md`：#19 结论、复跑及最终证据入口。新素材许可和身份读同目录 `material-valid.json`；优先最终 manifest，早期 `material.json` / `static.mkv` 是无效对照。
- `result/effect-acceptance/delivery-11-18-20261011/`：本轮本地交付验证，不替代播放验收。

源素材私人路径仅查本地 manifest，不写进公共文档或自动上传。

## 尚未实施的方向

可选 Agent 视觉理解、受约束的自然语言时段编辑、自动残留/闪烁判断和自动重擦仍需独立规格及证据。网页已有代表帧和手工时段编辑不等于这些 Agent 能力已完成。更换默认模型、云服务及通用聊天入口不在当前八票范围。

## 历史规划

保留[三次交付](plans/PRODUCT_EXPERIENCE_DELIVERIES.md)、[短片试擦](plans/SHORT_TRIAL_MILESTONE.md)、[速度方案](plans/STTN_INFERENCE_SPEED.md)中的原始取舍与证据。文中的“下一步”“未提交”“未签收”属于各自日期的历史状态，不能覆盖本文及对应 Release / Issue 的后续结论。历史性能只适用于其设备、输入、参数和测量范围。

某些旧工作树另有未跟踪的 `NEXT_WORK.md`、`AI_NATIVE_PRODUCT_DIRECTION.md` 及进度回调/Phase C 草稿。它们保留为本地历史，不作为新 clone 的必需依赖；其中注册器、Web、进度取消旧待办已被上述实现取代。保留这些文件、jobs 和素材，不为整理入口清理工作树。
