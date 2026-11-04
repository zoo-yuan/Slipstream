---
id: "06"
title: 训练配方立场：零演示纯 RL vs 混合对手 vs SPACeR 锚定
labels: [wayfinder:grilling]
status: open
assignee: null
blocked-by: ["02"]
---
## Question

复刻的训练配方取什么立场？三条路线（11-03 文档 §2.3 的"环境现实主义 vs 行为锚定主义"对立）：

- **TerraZero 姿态**：零演示纯 RL，靠环境现实主义（场景生成层 + 异质 + 真实规则）让 human-likeness 涌现。
- **现行混合对手**：沿 `../PufferDrive` 的 ADR-0003 方向（replay/IDM/policy 混合 Σ，日志保真做锚点），场景生成层做增量。
- **SPACeR 锚定叠加**：参考模型似然奖励 + KL，把 self-play 拉回人类分布（65K 参数轻量策略，顺带产出 sim-agent）。

裁决依据应包含：02 票披露的 TerraZero reward/规模细节、9_bernhard_dev 上 90 CLS-R 的配方（混合路线的现有上限）、以及与 `../PufferDrive` 仓 ADR-0002/0003 未决项的归口关系（该仓裁决还是本图吸收后回写）。

**补充（02 票）**：TerraZero reward **无路线 progress 项**（Table 8 题注置零，InterPlan 进度分最低 67.1）——零演示路线须裁决是否补 progress shaping；「评测期条件化」（同 checkpoint 免重训切基准）是否采纳为我们的验收/风格切换机制，也归本票。
