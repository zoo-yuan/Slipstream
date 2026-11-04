---
id: "08"
title: 异质 agent 保真档位：行人/骑行者/卡车各做到什么程度
labels: [wayfinder:grilling]
status: open
assignee: null
blocked-by: ["02"]
---
## Question

异质道路使用者的建模档位裁决（当前底子：VEHICLE/PEDESTRIAN/CYCLIST 三类常量 + 类型化尺寸采样，行人/骑行者动力学简化，无挂车卡车）：

- 行人：质点+社交力？轨迹级？横穿/jaywalker 场景需要什么最小能力？
- 骑行者：与行人共享模型还是车辆简化版？
- 卡车：尺寸分布扩展够不够，还是需要挂车运动学（TerraZero 的高保真档）？统一策略控异质（其 in preparation 状态）要不要等？

**修正（02 票）**：「统一策略 in preparation」只对**车+卡车**成立；**车+行人+骑行者统一策略已在报告发布**（per-type action heads：车 jerk / 行人 unicycle grid / 骑行者 bicycle grid，6.7M 参数，WOSAC 2023 0.632 > Gigaflow 0.619）——本票不得再按页面口径低估。卡车保真实况 = 自行车模型 + 轮胎侧偏力（参数未披露）；骑行者无脚本 NPC（仅 log 回放）。

裁决依据：02 票披露的保真档位 + 复刻范围票的勾选（若未选异质则本票关闭并移出范围）。
