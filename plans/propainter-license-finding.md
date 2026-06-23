# ProPainter 授权决策文档（A0.5 产出）

Date: 2026-06-22
Status: **Final** — A0.5 完成

## 结论

**`NO_USE_PROPINTER`**

ProPainter 不可进入 videowipe（MIT pip 包）的任何分发形态——既不能做 pip 默认，也不能做 `--model propainter` 可选档。任何形式的再分发（打包代码或权重）都违反 S-Lab License 1.0。

## 依据（LICENSE 原文，非转述）

### 1. 主代码：S-Lab License 1.0

ProPainter `LICENSE` 文件原文（来源 https://github.com/sczhou/ProPainter/blob/master/LICENSE ）：

```
# S-Lab License 1.0

Copyright 2023 S-Lab

Redistribution and use for non-commercial purpose in source and binary forms,
with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice ...
2. Redistributions in binary form must reproduce the above copyright notice ...
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse ...

[标准 BSD 式免责条款]

4. In the event that redistribution and/or use for commercial purpose in source or binary forms,
   with or without modification is required, please contact the contributor(s) of the work.

---
For inquiries or to obtain permission for commercial use, please consult
Dr. Shangchen Zhou (shangchenzhou@gmail.com) and Prof. Chen Change Loy (ccloy@ntu.edu.sg).
```

关键：**第 4 条 + 末尾联系人声明 = 商用必须取得 Shangchen Zhou / Chen Change Loy 书面许可**。

### 2. README 明示非商用

README License 段原文（来源 https://github.com/sczhou/ProPainter#license ）：

> "The ProPainter is made available for use, reproduction, and distribution **strictly for non-commercial purposes**. The code and models are licensed under NTU S-Lab License 1.0."

### 3. 权重同样非商用

权重与代码统一在 S-Lab License 1.0 下。三个核心权重（`ProPainter.pth`、`recurrent_flow_completion.pth`、`raft-things.pth`）从 GitHub Releases V0.1.0 下载，无单独宽松授权。"代码 Apache / 权重 CC BY-NC" 的常见陷阱在此不存在——因为代码本身就是非商用。

### 4. 依赖无传染问题（但不影响结论）

`requirements.txt`（来源 https://github.com/sczhou/ProPainter/blob/master/requirements.txt ）所列依赖（torch、opencv、einops、timm、scikit-image 等）均为宽松许可证。**没有** `iopath`、`basicsr`、`paniniferg`。但这无关紧要——主代码的 S-Lab License 已是硬性非商用封锁，依赖再干净也无法绕过。

### 5. NTU S-Lab 一贯政策

同作者 sczhou 的 CodeFormer（NeurIPS 2022）LICENSE 与 ProPainter **逐字相同**（除年份外），均为 S-Lab License 1.0。这是该实验室统一政策，不存在"个别项目例外商用"。

## 为什么 MIT 互斥

MIT 许可证承诺：可商用、可再分发、可闭源衍生。把 S-Lab 非商用代码/权重纳入 MIT pip 包（videowipe），MIT 的承诺与 S-Lab 的非商用限制直接矛盾——一旦 videowipe 携带 ProPainter 代码或权重并以 MIT 发布，即属"未经授权的商用再分发"，违反 S-Lab License 第 4 条。

`--model propainter` 可选档也不能解决：可选档的本质仍是 pip 包内再分发 S-Lab 非商用代码/权重，授权冲突依旧。

## 对 videowipe 路线的影响

1. **A2 阶段形态改变**：A2 不再做"ProPainter 内置化"。改为评估 **E2FGVI** 或其他商用友好（MIT/Apache/BSD）的 inpainting 模型作为画质升级候选。
2. **现有 `--external-command` ProPainter 路径保留**：用户自己 clone ProPainter、自己调用——这是用户与 S-Lab 之间的双边授权关系，videowipe 不做再分发，不违规。但**不应在文档里鼓励商用用户走这条路**。
3. **A2 不再是 C1 的阻塞项**：A0（软 alpha + 音频 + 羽化）是 C1 的画质地基，A2 是独立的画质升级线，与 web 前端解耦。

## 唯一合规路径（若未来确需 ProPainter 能力）

邮件 shangchenzhou@gmail.com / ccloy@ntu.edu.sg 谈商用授权。但即便取得授权，授权范围通常仅限申请方，**不能反向 MIT 化再分发给 videowipe 的下游用户**。所以这条路对"MIT pip 包"形态仍是死路。

## 来源

- ProPainter LICENSE: https://github.com/sczhou/ProPainter/blob/master/LICENSE
- ProPainter README (License): https://github.com/sczhou/ProPainter#license
- ProPainter requirements.txt: https://github.com/sczhou/ProPainter/blob/master/requirements.txt
- ProPainter weights/README.md: https://github.com/sczhou/ProPainter/blob/master/weights/README.md
- ProPainter Releases V0.1.0: https://github.com/sczhou/ProPainter/releases/tag/v0.1.0
- CodeFormer LICENSE (S-Lab 惯例对照): https://github.com/sczhou/CodeFormer/blob/master/LICENSE
- RAFT 代码许可证 (BSD-3-Clause): https://github.com/princeton-vl/raft
