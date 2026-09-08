# STTN 推理速度优化方案

状态：已实施自动 MPS 和空目标段带跳过，按用户后续要求整体提交。GPU 特征常驻原型未达到既定收益门槛，没有合入。基线为 `a7a09b1`。

## 实施与验收记录（2026-09-08）

本次涉及 13 个文件，包括运行时、维护脚本、测试和文档。运行时保持原来的 NumPy 后端接口、模型精度、权重及采样参数；新增维护脚本 `scripts/verify_inference_speed.py` 可复跑以下验收。

| 固定三秒试跑 | CPU 三次中位数 | MPS 三次中位数 | 端到端速度比 |
| --- | ---: | ---: | ---: |
| chinese1 | 63.072 秒 | 9.448 秒 | 6.68× |
| english1 | 61.573 秒 | 7.758 秒 | 7.94× |
| others | 125.100 秒 | 18.320 秒 | 6.83× |

像素采集单独运行，不计入三次计时。CPU/MPS 顺序执行以避免相互争用资源。三组编码前保留区全部逐像素一致，移除区 MAE 为 0.000024–0.000032/255，最大通道差为 1；相邻帧差分额外 MAE 均小于 0.000065/255。已查看首、中、末三个时刻的对比图，未见新增视觉差异；这里记录的是采样图检查和全部帧数值检查，不宣称人工连续播放验收。

真实 MPS 上的稀疏时间遮罩对照：6.432→2.362 秒，模型段调用 3→1，编码前整段 SHA-256 完全一致。稠密遮罩对照：6.588→6.691 秒，约 1.6% 开销，低于 5% 门槛，像素 SHA-256 同样一致。固定窗口为 english1 [125,200)，稀疏目标 [151,174)，使用实际 `execution_masks` 和 4 像素羽化。

默认 `device=auto` 的完整视频额外回归全部通过，实际设备均为 MPS：chinese1 56.226 秒（591 帧）、english1 40.068 秒（522 帧）、others 91.788 秒（516 帧）。核对了尺寸、帧数、时长及音轨存在性。chinese1 另有本轮 CPU 整片单次 369.779 秒，约为 6.6 倍速度；完整视频没有 CPU/MPS 三次中位数，因此不把试跑表格推广为所有整片的性能承诺。

GPU 特征常驻原型在无并发基准干扰时复测：现路径热运行中位数 1.897 秒，原型 1.815 秒，耗时仅减少约 4.3%，输出一致；未达到第二步的 15% 门槛，所以保留现有执行路径。

审查修复：内部稳定遮罩共享引用，普通回调一律快照（包括底层缓冲可变的 readonly view）；基准元数据采集失败独立记录并始终清理引擎；比较图写入失败会中止验收。`make check` 为 336 项通过，Ruff 检查通过。本机验证覆盖 CPU/MPS，CUDA/ONNX 未做真实硬件运行；检测算法和遮罩像素生成逻辑未改动。

本地证据：`result/inference-speed-final/report.json`、三个 `*-comparison.png`、`result/speed-auto-full/benchmark_report.json`、`result/speed-resident-probe.json`。原始帧和视频留在忽略目录，不入 git。较早的 `result/inference-speed/` 是已中止的探索运行，不作为最终验收证据。

复跑正式试跑验收：

```sh
PYTHONPATH=src python scripts/verify_inference_speed.py --output result/inference-speed-final
```

以下保留原方案的目标、取舍和实施边界。

## 目标与约束

优先缩短本机短片试跑和完整清理的等待时间。第一步采用最小改动：让 Torch 的 `device="auto"` 在支持的 Mac 上选择 MPS。

保持已有权重、640×120 模型裁剪尺寸、默认 gap=25、ref_length=5、neighbor_stride=5、WipePlan 的移除/保留决定及试跑上下文。运行时代码只进入 `src/videowipe/`。不引入服务、账号、密钥、模型下载要求或用户配置步骤。

## 已验证的瓶颈

当前 `TorchBackend` 自动选择只有 CUDA→CPU；显式传入 `mps` 已能运行。Apple Silicon 上即使选择了 Torch 后端，也不会自动使用 GPU。

本轮临时探针使用现有权重和 english1 的第 125–149 帧、已有 golden mask 的第一个裁剪带，执行真实 `_process_segment`。固定 25 帧，模型已加载，计时包含预处理、编码、注意力、解码和段内合成，不包含视频解码、模型加载和 FFmpeg 输出。

| 设备/线程 | 单段耗时 | 说明 |
| --- | ---: | --- |
| CPU / 默认 8 线程 | 14.94、14.79 秒 | 注意力约占 79%，解码约占 16% |
| CPU / 4 线程 | 17.82、17.87 秒 | 更慢 |
| CPU / 1 线程 | 27.38、27.33 秒 | 更慢 |
| MPS / 首次执行 | 3.74 秒 | 包含首次设备执行开销，不含模型加载 |
| MPS / 后续执行 | 1.747、1.742 秒 | 热运行单段约为 CPU 的 8.5 倍速度 |

MPS 与同一脚本中的 CPU 参考结果平均绝对像素差为 0.00002274/255。这只是一个裁剪带的数值检查，不能替代多视频视觉验收。CPU 探针在沙箱中运行；MPS 探针在沙箱外运行，仍需在同一执行环境完成正式对照。

沙箱内 `mps.is_available()` 为 false，沙箱外为 true，PyTorch 均为 2.12.0。不能把沙箱检测结果解释为本机没有 GPU。

证据保存在未跟踪的生成物目录：`result/speed-probe.json`、`result/speed-probe-mps.json`。临时探针不进入产品代码。

此前真实试跑报告 `result/trial-acceptance/report.json` 中，三秒试跑的 inpainting 耗时分别约为 64.5 秒（chinese1）、64.4 秒（english1）、135.1 秒（others）。这是已有验收产物，本轮没有重跑；不得将单段加速倍数直接乘到这些数字上。

## 推荐实施顺序

### 第一步：自动使用 MPS，独立交付

预计半天至一天，包含三组真实视频验收。主要修改 `backends.py`、`engine.py` 的设备说明、现有 benchmark 脚本与测试，以及 README/README_CN；预计 6–7 个文件，无新服务。

1. Torch 的 auto 顺序改为 CUDA→可用 MPS→CPU。显式设备选择优先，保持 CPU 和 CUDA 的既有路径。MPS 使用当前 FP32，不开启新的混合精度策略。
2. 可用性检查使用 PyTorch 的 MPS API。MPS 不可用时 auto 选择 CPU；显式请求不可用设备时给出可操作错误。实际推理错误仍由库抛出，不用静默 CPU 回退掩盖错误或重试已经输出的片段。
3. 在现有 benchmark 报告中补充实际设备、Torch 版本、线程数和权重 SHA-256。维护脚本增加 `--device`，直接透传引擎已有 device 参数，支持同机 CPU/MPS 对照；不新增面向用户的 CLI 参数或环境变量。
4. 只有通过下述真实质量和性能门槛，才合入 auto 的默认行为。若三组样本未通过，保留当前默认顺序，提交可用的基准能力和诊断改进，不宣称完成默认加速。

这是推荐的最小方案：现有实现已测得显著收益，无需先改注意力算法或重构后端。

### 第二步：让 Torch 中间特征留在设备上，独立交付

预计一至两天。主要涉及 `backends.py`、`inpainters/sttn.py` 和相关测试、benchmark，第一步无需等待它。

目前 encode、transform、decode 之间都经过 NumPy，中间特征反复返回 CPU。增加 Torch 专用的私有张量执行路径，复用同一套邻居/参考帧索引和模型算子：整段特征留在设备上，transform 结果直接送 decoder，仅解码后的图像回 CPU。保持既有 NumPy 方法兼容及 ONNX 路径，避免引入通用张量框架。

保持窗口顺序、每次解码后的 uint8 量化及重叠帧平均顺序，避免将数值变化混入数据搬运优化。使用 `torch.inference_mode()` 覆盖专用推理区域，不改变模型精度。按现有 gap 释放段内张量，不缓存整部视频，不增加并发 worker。

以第一步为基线：GPU 模型处理热运行中位数至少改善 15%，CPU 端到端退化不超过 5%；若收益不足，不合入这层额外实现。单独记录 CPU RSS 和设备内存峰值，GPU 峰值相对第一步不超过 1.2 倍。

### 第三步：整段裁剪带没有待移除像素时跳过模型，独立交付

预计一天。主要涉及 `inpainters/sttn.py` 和试跑/时间遮罩测试，不依赖第二步。

利用现有逐帧执行遮罩，判断当前 gap 内需要输出的帧在某裁剪带是否全部为零；只有全部为零才跳过该带的裁剪、模型推理和回填。静态 mask 继续按原路径执行。

只要存在一个有效目标像素，就保留该段完整参考上下文并执行原流程。不能删掉无字幕参考帧、重排 frame index 或重新划分 gap。羽化区域的非零 alpha 也必须算作有效像素。提前取得的遮罩复用于合成，每帧只求值一次，缓存生命周期限于当前段。

用带空白时间段的固定 WipePlan 验收；静态全程字幕不承诺收益。跳过的段带模型调用次数应为零，编码前结果与旧路径逐像素一致；稠密目标端到端退化不超过 5%。

## 官方实现与取舍

- [PyTorch MPS 文档](https://docs.pytorch.org/docs/2.12/notes/mps.html)：采用官方设备可用性检查和 `.to(device)` 机制；本轮已实际跑通。
- [STTN 原始推理实现](https://github.com/researchmm/STTN/blob/master/test.py)：编码特征保留在设备上，transform 直接接 decoder；借用这个执行边界，保持项目自己的采样参数。
- [video-subtitle-remover 的 STTN 实现](https://github.com/YaoFANGUK/video-subtitle-remover/blob/main/backend/inpaint/sttn_auto_inpaint.py)：同样在设备上连接模型阶段；不照搬动态 gap，因为它会改变我们已验证的试跑上下文。
- [PyTorch inference_mode](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html)：仅用于第二步私有推理路径。

不优先做 ONNX/CoreML 转换、量化、换模型、减小分辨率或参考帧、增加并行推理：这些都比补上已有 GPU 设备选择更重，或会引入新的质量变量。CPU 线程测量已经否定了本机减少线程这一捷径。已有遮罩缓存不重复建设。

最脆弱的假设是：单段 MPS 收益能推广到不同裁剪带、长视频及同机正式运行。若不成立，完整视频加速和稳定性都可能不足；默认切换由三样本验收控制，其他机器仍按实际设备检测选择，不能把 Apple Silicon 型号直接当作可用性证据。

## 验收与交接

使用相同提交基线、输入/遮罩/权重 SHA-256、参数、执行环境及电源条件。CPU/MPS 分别预热一次，再至少运行三次取中位数；冷启动、模型加载、模型处理和端到端耗时分列。GPU 分阶段计时必须同步设备，避免把异步提交时间当推理耗时。内存峰值用独立进程测量，不能把进程累计高水位当作每次运行峰值。

第一步目标：三组固定样本的模型处理均至少比 CPU 快 2 倍，端到端均不回退，至少两组达到 2 倍。它是待达成的验收目标，不是本轮已经证实的结果。

质量验收同时检查：编码前 CPU/MPS 移除区 MAE ≤ 0.5/255；所有未移除像素完全一致；相邻帧差分相对基线的额外 MAE ≤ 0.5/255。并排播放三个视频，检查文字残留、边缘拖影和闪烁，数值通过不能抵消肉眼可见回退。

保留原试跑范围、帧数、尺寸、帧率和音频对应；覆盖跨 gap 边界、结尾不足一个 gap、单帧、多个裁剪带、稀疏/全空时间遮罩、CPU-only、MPS 不可用及显式设备错误。CUDA/ONNX 在可用环境分别做回归；没有相应硬件时明确标为未验证。

现有命令：

```sh
make check
PYTHONPATH=src python scripts/verify_trial.py --output result/trial-speed-after
```

第一步增加维护脚本 `--device` 后，分别执行 CPU 与 MPS 的正式对照：

```sh
PYTHONPATH=src python scripts/benchmark_pipeline.py input/detext_examples --mask-dir input/detext_examples/mask --ocr off --gap 25 --repeat 3 --device cpu --output-dir result/speed-cpu
PYTHONPATH=src python scripts/benchmark_pipeline.py input/detext_examples --mask-dir input/detext_examples/mask --ocr off --gap 25 --repeat 3 --device mps --output-dir result/speed-mps
```

真实权重及视觉验收不能由 mock 测试替代。若实现意外涉及遮罩生成，额外对比 `input/detext_examples/mask/*.png`；原则上本方案不改检测与遮罩质量。

两个运行时优化可分别回退；本次按用户后续要求整体提交，无数据迁移。保留用户已有任务与输出，生成报告和视频不入 git。没有推送或发布。
