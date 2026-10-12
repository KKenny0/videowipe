# 清理效果验收（Issues #11 → #12 → #13）

当前任务与历史规划状态见[统一入口](../NEXT_WORK.md)。下方带日期的验证结论属于对应历史运行；2026-10-11 的交付整理与本轮验证在文末单独记录。

目标是去掉该去的、保留该留的，减少达到可用结果的修改。轻微纹理差异不单独判失败；明显影响使用的变形、闪烁仍须记录。执行成功、蒙版覆盖率和抽帧检查不能替代正常播放判断。

## 当前范围与实施

基于 v0.11.1，复用真实网页上传、默认选择、推荐试擦、完整导出与审阅接口。维护脚本通过进程内 HTTP 传输调用生产处理器，检测器和 STTN 均真实运行；不声称完成浏览器可用性或五位真实用户验证。不新增服务、生产 API、WipePlan schema 或模型依赖。

默认四条历史回归案例：仓库 chinese1、english1、others，以及现有生成器产生的黄白混色描边字幕。others 包含白闪与交叉淡化；中文片包含复杂背景。这些素材均已用于调参，不能称为独立新样本或真实用户案例。原用户截图因缺少原片、输出继续标记未验证，不作为通过/失败分母。

使用前准备现有开发、web、Torch 环境及 FFmpeg，从仓库根运行。真实模型沿用产品默认设备与权重解析，不另选更有利的配置。输出目录必须不存在，避免污染既有证据：

```sh
PYTHONPATH=src python scripts/verify_cleanup_effect.py --output result/effect-acceptance/default
```

脚本默认运行四个案例，也可以用重复的 `--case` 参数限定案例。每个案例记录输入身份、代码与脚本身份、默认选择、计划、实际运行设备/权重、警告、试擦/全片路径与 SHA-256、修改轮数和时长。输入可读后即保存源身份和预期对象；后续失败保留 `stage`、错误和已取得的证据，尚未产生的输出/指标保持缺省，效果状态仍为 unverified。机器检查覆盖完整解码、播放 Range 请求、试擦/全片计划与编码前像素一致、零 alpha 区域编码前不变，以及已有语义保留标注的覆盖情况。保留内容需要独立于执行 mask 检查，不能仅凭 mask 外像素未变宣称无误擦。仓库标注只在文件名与 manifest 的 `source_sha256` 同时匹配时使用；同名异源文件或被替换的样例保持未标注状态。

`review.html` 提供原片/结果上下同步的完整对比视频；源文件与原分辨率输出另有链接。比较视频再次有损编码，只供审阅，数值检查使用原始运行数据。所有生成素材、计划和视频留在忽略的 result 下，不上传用户视频。

## 真实素材的默认流程（#12）

外部真实素材使用独立的 `real_video` 案例名。运行前根据原片写好 expectations JSON，包含源 SHA-256、验收目标及需要移除/保留的对象；这些字段只用于记录和播放审阅，不传入检测、选择或执行接口。例如：

```json
{
  "source_sha256": "源视频的完整 SHA-256",
  "request_goal": "移除双语字幕，保留人物和台标",
  "objects": [
    {"action": "remove", "description": "下方中英文字幕"},
    {"action": "keep", "description": "人物、现场背景及右下台标"}
  ]
}
```

来源许可、裁剪区间、转换方式和既有调参/诊断历史可作为额外字段保存在本地 expectations 中。曾用于诊断的真实片段不能称为独立盲测；裁剪片段的结果也不能推广到完整录屏。

```sh
PYTHONPATH=src python scripts/verify_cleanup_effect.py \
  --case real_video --source /path/to/source.mkv \
  --material-note '已许可的真实素材；说明是否裁剪及既有使用历史' \
  --expectations /path/to/expectations.json \
  --output result/effect-acceptance/real-default
```

缺少移除/保留说明、未提供 expectations 或源哈希不符时，命令在创建输出目录前拒绝执行。完整记录保存默认候选、推荐试擦区间和实际试擦/导出请求；零次修改的运行应使用完全相同的默认选择和推荐区间，不额外指定框、保护区或时间。后续 #13 从 baseline 继承原验收预期，不能换一套预期使失败变成功。

新产物仍须按下节进行正常播放验收。历史输出只能保留其原版本和原判断；如复用，应单独记录当前规划与旧规划的身份对照，不把一次重新规划说成全流程重新通过。

## 判断与记录

`report.json` 初始状态均为 unverified。审阅者应看完完整成片，并重点查看字幕起止、白闪、叠化及以前的失败位置。每个 `effect_review` 填写：

- status：pass / fail / unverified；reviewer：具体审阅者；note：判断依据；源视频与输出的 SHA-256 必须匹配本次产物。
- normal_playback、entire_output_reviewed：是否按正常速度检查完整输出。
- wrong_target、residue、false_removal、unusable_distortion：是否存在对应影响使用的问题。通过需显式全部为 false。
- problems：失败的零基半开帧区间、类别和可见现象。可附辅助截图，但不能拿静态检查冒充播放。

执行 `validate_review` 可检查记录契约；该检查只验证记录完整性，不替代审阅者判断。例如从仓库根检查一份另存的 reviewed.json：

```sh
PYTHONPATH=src:scripts python - /path/to/reviewed.json <<'PY'
import json
import sys
from verify_cleanup_effect import validate_review
with open(sys.argv[1], encoding='utf-8') as stream:
    report = json.load(stream)
for row in report['cases']:
    validate_review(row['effect_review'], row)
print('Review record contract valid; this does not establish playback quality.')
PY
```

保留机器原始数据，报告修改只填写 effect_review；另存最终审阅报告，勿覆盖初始报告。材料不全时保持 unverified。试擦/全片像素检查失败须单独记为工程检查失败，不能因视频仍可播放而改判；这类失败不阻止记录其它案例的独立结果。

当前中文样片有平均/标称帧率不一致的输入警告，原始 OpenCV seek 可出现帧偏移。保留原片运行及警告，不以转码版本替换原始默认路径；效果和精确定位保证分别报告。

用户针对指定输出确认“达到预期”等结果时，可另存 `user_acceptance`，绑定源片、输出和计划哈希，保留原话及审阅请求上下文。它表示该案例的用户结果确认；不要自动补写用户未逐项确认的播放方式、完整观看或缺陷字段，也不要改写原始机器报告。缺少这些字段时，严格 `validate_review` 的 pass 条件仍未建立，但不撤销用户已作出的结果确认。机器检查失败与用户接受可以同时保留。

## 一轮修正

只有默认结果被记录为 fail 的案例才进入 #13。根据原片和失败位置，通过已有目标选择、框、时段或保护编辑制定一份完整 overrides；一次提交可以含多项相互必要的编辑。提交并查看结果后不再根据结果反复调参冒充一轮。

```sh
PYTHONPATH=src python scripts/verify_cleanup_effect.py \
  --baseline result/effect-acceptance/default/reviewed.json \
  --edits result/effect-acceptance/edits.json \
  --output result/effect-acceptance/corrected
```

edits 是案例名到现有 review overrides 的非空 JSON 映射，每个案例也必须有非空编辑对象；空编辑不会退回默认执行或记作一轮修正。脚本在创建输出目录前检查 baseline 中有对应的、零次修改的已审阅失败；重新生成默认计划并比对执行语义身份，避免给变动后的候选套用旧编辑；随后通过现有 PATCH 审阅接口保存一次编辑，试擦、全片与检查沿用同一路径。修改次数在 PATCH 成功后才加一。修改前的默认结果和一轮修改后的结果均保留，记录是否救回及剩余问题。

若全部默认案例通过，#13 结论为本素材范围内不适用，不伪造失败。若仍失败，则记录现有能力上限和复现路径，不扩大本轮为无边界算法修复。

## 验收与回退

代码交付运行 make check。本轮追加的弱依据候选安全修复只调整默认选择，保留候选类型、显式选择和蒙版算法；同时重跑固定 PNG 对照及检测基线。代码可独立撤销；原片、旧结果及任务不会被覆盖。新一轮提交、推送与发布需独立授权，本轮先交付工作区改动与证据。

## 2026-10-02 验证结论

旧四例仅作为回归材料；用户已确认此前多次验证正常，本轮不要求再次人工验收，也不把该历史反馈伪装成本次全片播放签收。当前运行完整解码 1809 帧；三例试擦/全片编码前一致，chinese1 因原片 seek 偏移不一致，原始警告和不一致记录保留。

针对用户关心的截图，裁取右侧原画面并重复为 3 秒、75 帧静态诊断视频。原始版本、操作路径和原视频未知：此探针不是原始故障复现，不能验证运动、字幕切换或时序蒙版。

- 默认选择 c3+c4。c3 把机器印刷文字误判为水印，输出文字变糊；底部字幕有黄色残影。首帧已有明确失败证据，未声称正常播放通过。
- 一次已有选择编辑改为仅 c4，经 PATCH 保存后重新试擦、导出。机器文字得到保留，字幕残影仍在：部分救回，未达到可用结论。机器文字保留是首帧观察，不扩大成所有内容均无误擦。
- 两次均完整解码 75 帧、试擦/全片计划一致、编码前像素一致、零 alpha 区域编码前不变。一轮修正运行约 27.4 秒（含重新规划）；不含人类思考时间，不能当作用户到可用结果的耗时。
- 截图中原来的“只擦掉半行”未复现，原案例继续未验证。当前发现只支持后续调查场景文字误选与静态素材残影，不支持宣称原问题已修复。

本次本地证据在 `result/effect-acceptance/`：`screenshot-probe/probe.py` 保存截图裁切构造，`screenshot-default/reviewed.json` 与 `screenshot-corrected/reviewed.json` 绑定对应源片和结果哈希；`screenshot-edits.json` 保存唯一编辑；`screenshot-comparison.png` 从左到右为源片、默认输出、一轮修正后的同一首帧局部。旧四例在 `default-v2/`。这些忽略产物不进入 Git。

复跑截图探针的产品流程：

```sh
PYTHONPATH=src python scripts/verify_cleanup_effect.py \
  --case screenshot_scene --source /absolute/path/to/screenshot-still.mp4 \
  --material-note '用户截图右侧画面裁切后重复为75帧；仅静态诊断，非原视频' \
  --output result/effect-acceptance/screenshot-default-new
```

代码验证：`make check`，463 passed。仅新增验收脚本、说明和两个记录契约/保护像素检查测试；没有修改生产检测或修复算法，没有提交、推送或发布。


## 2026-10-03 本地修复与残影定位

弱依据误选已修正，尚未提交。`_classify_region` 的“小且靠边”分支原来没有持续性或内容证据，却标为 persistent 并默认移除。现在仍返回 watermark 候选，但默认保留，理由明确标注水印依据不确定。识别到网址的水印、字幕规则及显式 watermark 目标/手选移除保持可用。共享分类器的两处调用均沿用该规则；仅针对 scene_text 的稳定中心覆盖物提升分支不受影响，未扩大改写水印策略。

回归测试在旧生产代码上实际得到 3 failed、2 passed，失败点均为默认误选；修改后 5 passed。日志在 `result/effect-acceptance/safety-fix/red.txt` 和 `green.txt`。新代码重新检测静态截图视频后，c3 为 keep，默认只移除字幕 c4，计划在 `safety-fix/replan/`；全部 75 帧执行 alpha（feather_radius=4）与旧 corrected 计划完全相同，比较记录在 `safety-fix/replan-parity.json`。这项证据验证新默认计划，并非新一轮正常播放签收。

残影诊断复用了原 default/corrected 两份执行计划，各完整运行 75 帧。在 STTN 最终合成入口捕获同一零基首帧的原图、实际模型预测（时间窗口平均及尺寸还原后）、执行 alpha 和编码前最终图；`frame-0.npz` 保存无损数值。每份 `identity.json` 绑定源视频、计划、帧哈希及实际设备/权重；`old-new-parity.json` 确认捕获的首帧最终图和 alpha 哈希与上一轮记录完全一致。因此本次诊断确实对应已有失败产物。

两份 `screenshot-*/subtitle-stages.png` 从上到下为原图、预测、最终合成和 alpha。字幕 ROI 为 x=[180,546)、y=[1044,1086)。以 BGR 条件 R>120、G>100、B<0.7×min(R,G)、|R−G|<100 定位黄色，仅作该局部的可复核诊断：原图 808 个黄色像素全部处于 alpha=1；最终 35 个黄色像素也全部处于 alpha=1，且均已在预测中。满 alpha 区域最终图与预测逐像素差为 0。可见黄色残影来自实际预测自身，不能归因为这块区域覆盖外漏擦或 alpha 混回原图。该证据只定位到预测阶段，尚不能判定是模型能力、模型输入缩放还是上下文导致；没有据此改动生产修复算法。

本地状态：#11 已具备默认路径、证据身份和审阅记录工具；#12 截图静态探针仍有黄色残影，未达到可用结果；#13 已有一次选择修正并保留前后产物，只解决机器文字误擦，未救回字幕残影。工具完成不等于三项验收通过。旧四例不要求用户重复验收；原视频、原版本与操作路径未知，原“半行”故障仍未验证，也未声称修复。远端 Issues 未修改。


本轮验证：`make check` 退出 0，466 passed（1 条既有 Starlette/httpx 警告）；`make decision-baseline` 退出 0，可见保留对象误选率 0、移除候选可用率/选择召回率及动作匹配率均为 1；`make fact-baseline` 退出 0，完成指标计算和固定 PNG 对照，不代表播放质量通过。事实基线 frame union Jaccard=0.319290、Boundary F=0.454246；固定 PNG 的 IoU：chinese1=0.393347、english1=0.464482、others=0.426613。上述汇总与已有 HEAD=1982331 的基线报告一致，输入、标注、PNG、权重身份一致；评估源码身份因本轮检测代码修改而变化。没有修改任何期望或阈值。全部详细报告和比较在 `safety-fix/{check.log,decision-report.json,fact-report.json,fact-comparison.json}`。本次为含未提交改动的本地验证，并非 clean-git 正式发布基线。

## 2026-10-11：#11 / #18 本地可审查交付

以 `1982331` 为基准整理本地候选，未提交、未推送、未修改远端。生产行为仅为前述“小且靠边”的弱依据文字默认保留；显式选择仍可移除。验收工具追加失败阶段/身份保留、按 manifest 源哈希绑定标注，以及空纠正编辑拒绝。manifest 只增加三个现有视频的 SHA-256，原标注、蒙版、期望和阈值均保留。

| 验收条目 | 本地交付状态与证据 |
| --- | --- |
| #11 每例目标、保留内容、原片/输出/计划/版本/问题区间 | runner 保存预期对象与身份，review 契约校验零基半开问题区间。未标注素材明确要求审阅预期，不冒用同名仓库标注。 |
| #11 真实/合成统一记录，错误类型与修改次数 | 复用 `default-v2` 四例和截图探针前后记录；本轮重新核验六例源/输出/执行计划哈希与侧车、审阅契约。不是重新运行四条模型链路。 |
| #11 正常播放与未验证边界 | 通过必须填写正常速度完整播放及四项失败标记；缺原视频的截图案例继续未验证。截图静态探针的 fail 只说明探针失败。轻微纹理差异不单独判失败。 |
| #11 素材处理范围 | 仅本地 HTTP 处理器与忽略目录，无上传或遥测服务。 |
| #18 历史待办与当前入口 | README 中英文指向 `NEXT_WORK.md`；三份已跟踪计划增加历史状态；主仓库四份本地旧草稿仅加状态说明，原文保留。 |
| #18 实现、发布和验收分开 | 入口列已发布能力、#11–#18 当前票据与未实施 Agent 方向；#19 的手动目标判断不泛化为默认流程通过。 |
| #18 引用与材料保留 | 新增/更新的 29 个本地文件链接已解析；Issue/Release 状态已读回。jobs、素材、历史证据和父 Issue 均保留。 |

可提交候选为本工作树中的 11 个文件：`src/videowipe/detect.py`、`scripts/verify_cleanup_effect.py`、本文、`tests/test_cleanup_effect.py`、`input/detext_examples/fact_baseline.json`、`NEXT_WORK.md`、两份 README、三份 `plans/` 历史方案。主仓库另有四份未跟踪本地草稿的状态横幅（`NEXT_WORK.md`、`AI_NATIVE_PRODUCT_DIRECTION.md`、`plans/WIPEENGINE_PLAN_PROGRESS_CALLBACK.md`、`plans/QUALITY_ACCEPTANCE_NEXT.md`），不混入本候选。生成日志、报告、隔离候选及视频不入 Git。

回退可按两组独立处理：撤销误选规则、runner、测试和 manifest 元数据恢复本轮工程行为；撤销 README / 当前入口 / 历史横幅恢复导航。均不需要迁移 WipePlan，也不删除已有源片、任务或输出。历史默认报告、截图失败记录和 #19 诊断继续有效地保留各自范围。

本轮验证（隔离目录仅叠加这份候选；未提交，因此是非正式基线）：

- `make check`：470 passed，1 条既有 Starlette/httpx 弃用警告。旧检测实现上的三个误选回归失败；候选通过。runner 的四个新增反例先失败、修复后通过。
- `make decision-baseline`、`make fact-baseline`：均退出 0，宏平均与已有 safety-fix 记录相同。保留对象误选率 0；fact union Jaccard 0.319290、Boundary F 0.454246。
- 固定 PNG IoU：chinese1 0.393347、english1 0.464482、others 0.426613，均与已有记录一致。输入、权重、评估源码和 PNG 身份一致；给 manifest 补充源哈希使 annotation_sha256 改变，已逐结构核对除此以外无变化。
- `git diff --check`、改动 Python 的 Ruff F 类检查、runner `--help` 及 29 个更新链接检查通过。六份历史案例记录的身份与审阅契约重新核验通过，未把历史结果写成本轮播放签收。

新证据在 `result/effect-acceptance/delivery-11-18-20261011/`：`verification.json` 记录隔离候选和命令退出码；`check.log`、红绿日志、`baseline-comparison.json`、`reused-evidence.json`、`links.json` 保存对应结果。最终候选文件哈希见 `candidate-files.json`；验证后仅补充本文等文档，不改变通过验证的代码和 manifest。

以下为 2026-10-11 提交 #11/#18 前的剩余事项快照：#12 的新素材默认全流程及正常播放签收、#13 从该失败出发的一轮修正尚未执行；当时 runner 仅有固定历史案例入口（当前真实素材入口见上方 #12 流程）。#14 需要五位真实参与者；#15 仅对可用结果计成功成本。中文 seek 偏移和原半行故障仍未解决，本次不以整理交付替代这些验收。

## 2026-10-11：#12 / #13 结果确认与 #15 阶段测量

生产源码基于 main `0245a0c`，本轮仅扩展维护验收入口、测试和文档。真实裁剪片段的默认选择为两条字幕轨迹；人物姓名与英文歌词被检测为同一行并纳入英文轨迹的实际范围。这是默认行为与保留姓名目标的差异，不直接认定为必须修改产品的缺陷。既有多语、白闪、叠化及混色描边案例复核了输入、旧输出和当前规划一致性，历史播放证据仍按原版本范围引用，不重写为当前版本新签收。

通过已有区域框接口对英文轨迹保存一次编辑，选择和时间不变，重新试擦并导出。用户对修正输出回复“达到预期了”。该确认仅覆盖这份输出；没有更换模型、修改检测算法或新增生产编辑能力。没有执行浏览器自动操作，不能将 HTTP 链路验证写成独立真人可用性验收。

技术限制继续保留：试擦与全片计划相同；独立观察区域前240帧中80帧有边缘像素变化；试擦编码前哈希因检查器中断未保存，逐像素一致性未知。检查器中断后使用同一保存编辑重试，操作内容未变，重试没有增加编辑轮数。用户可接受结果不等于这些检查被改判为通过。

本地证据：`result/effect-acceptance/package-12-13-15-20261011/` 下的 `REPORT.md`、`user-acceptance.json`、`accepted-case.json`、`EVIDENCE_GAPS_12.md` 和 `costs-15.json`。原始视频、私人路径及详细报告仍留忽略目录。#15 记录了机器阶段时间、检查器中断和重试成本；真人操作时间、严格冷/热条件及可用结果总成本未测，不能宣布整票完成。票据是否合入结项以 GitHub 实时状态为准。
