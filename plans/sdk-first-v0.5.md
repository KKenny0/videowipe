# VideoWipe SDK-first v0.5 实施计划

Date: 2026-07-17

## 产品裁决

VideoWipe 的主线定位是可嵌入生产项目的视频清理引擎：接收视频和清理意图，完成目标检测、mask 生成与视频修复，并通过稳定的 Python 契约交给批处理 Worker、服务或其他上层产品调用。

CLI、Web UI 和 Docker 是 SDK 的适配入口。翻译工作台、时间线、Creator Remix、Codex Workspace、MCP 和发布流程不进入 v0.5 核心范围。此前 M0–M4 保存在 `archive/creator-remix-m0-m4`，不整体 cherry-pick 回主线。

## 版本目标

v0.5 的完成标准不是功能数量，而是“能否可靠嵌入”：

- 老的 `remove_text()` 和 `WipeEngine.process()` 保持兼容。
- 新调用方获得结构化请求、结果、进度、取消和错误。
- 一个 Engine 可以顺序复用模型处理多个视频。
- wheel/sdist 可以在干净、无图形界面的环境安装和启动。
- CLI、Web 和示例都经过同一 SDK 核心路径。
- 仓库和包元数据如实声明 GPL-3.0。

## S1 — 产品边界与授权事实

状态：完成

- README 首屏改为 SDK-first，Python 集成优先于 Web 产品叙事。
- ROADMAP 指向本计划，停止 Creator Remix M5。
- README、LICENSE 和包元数据统一为 GPL-3.0。
- 不更改版本号；版本同步留到正式发布门禁。

验收：文档边界一致，仓库不存在把 VideoWipe 本体声明为 MIT 的公开表述。每阶段结束运行 `waza:check`。

## S2 — 稳定 SDK 契约

状态：完成

新增公共对象：`WipeRequest`、`WipeResult`、`ProgressEvent`、`CancellationToken`，以及稳定的 `WipeError` 异常层次。

新增 `WipeEngine.run(request, on_progress=None, cancellation=None) -> WipeResult`。现有 `process()` 保留签名与字符串返回值，并转调新核心路径。Engine 支持上下文管理器，`cleanup()` 幂等；同一实例支持顺序复用，但 v0.5 不承诺多线程并发安全。

取消只在可观测边界生效：检测前后、模型加载前后、分段处理和外部进程边界。不承诺强制中断一次不可分割的模型推理。

验收：兼容性、复用、结果序列化、进度阶段、取消和异常映射都有不依赖大模型下载的测试。

## S3 — 打包与无界面运行

状态：完成

- 默认 OpenCV 依赖改为 headless，避免默认包与 headless extra 冲突。
- ONNX、Torch、OCR、Web 保持可选依赖。
- 构建 wheel 和 sdist，并检查归档内容。
- 在干净虚拟环境安装 wheel，验证 import、CLI help 和缺少推理后端时的错误。
- CI 覆盖单测、构建与已安装 wheel smoke test；支持版本以 CI 实测为准。

验收：安装证明来自构建产物，而不是 editable install。

## S4 — 嵌入式使用证明

状态：完成

- 增加长生命周期批处理示例，证明模型只加载一次。
- 增加自定义 Inpainter 示例，证明第三方模型可以通过 registry 接入。
- 集成测试确保连续任务不共享 mask、metrics 或取消状态。
- CLI 继续作为适配器，不新增任务平台或新服务。

验收：两个示例均可 smoke test，README 示例与真实公共签名一致，最终执行完整 `waza:check`。

## 非目标

- Creator Remix、LocalizationService、ASR 翻译和字幕烧录。
- Timeline、MediaProject、Variant、Inspector、MCP、Codex Plugin。
- 新的默认修复模型或模型权重再分发。
- Hosted SaaS、任务队列、多租户和鉴权系统。
- 在没有完整来源审计和重新授权依据时改成 MIT 或其他宽松许可证。

## 回滚

S1 仅改变文档和包元数据。S2 的旧 API 始终保留兼容入口。S3/S4 不引入持久化数据格式，因此任一阶段都可以按独立提交回滚。Creator Remix 分支保持不变。

## 最脆弱假设

“现有用户主要把 VideoWipe 当作可嵌入组件”仍是产品推断。v0.5 只投资可复用且可逆的 SDK 基础设施；在出现真实集成需求前，不继续投资大型 Workspace 或编辑 UI。
