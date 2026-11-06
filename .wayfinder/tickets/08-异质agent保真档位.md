---
id: "08"
title: 异质 agent 保真档位：行人/骑行者/卡车各做到什么程度
labels: [wayfinder:grilling]
status: closed
assignee: zoo（已裁决）
blocked-by: ["02"]
---
## Question

异质道路使用者的建模档位裁决（当前底子：VEHICLE/PEDESTRIAN/CYCLIST 三类常量 + 类型化尺寸采样，行人/骑行者动力学简化，无挂车卡车）：

- 行人：质点+社交力？轨迹级？横穿/jaywalker 场景需要什么最小能力？
- 骑行者：与行人共享模型还是车辆简化版？
- 卡车：尺寸分布扩展够不够，还是需要挂车运动学（TerraZero 的高保真档）？统一策略控异质（其 in preparation 状态）要不要等？

**修正（02 票）**：「统一策略 in preparation」只对**车+卡车**成立；**车+行人+骑行者统一策略已在报告发布**（per-type action heads：车 jerk / 行人 unicycle grid / 骑行者 bicycle grid，6.7M 参数，WOSAC 2023 0.632 > Gigaflow 0.619）——本票不得再按页面口径低估。卡车保真实况 = 自行车模型 + 轮胎侧偏力（参数未披露）；骑行者无脚本 NPC（仅 log 回放）。

## Resolution

已裁决（zoo，2026-11-05，全档偏满——服务「真实世界可部署 SOTA」目标与 sim-agent 双重角色）：

1. **行人 = 脚本横穿者内核化**：Python 原型（票 07 M2）→ C 侧 scripted walker（放置 + 直线/班次横穿，unicycle 速度模型）；社交力/ORCA 与策略控行人后置到 sim-agent 阶段。
2. **骑行者 = bicycle 动作网格，卡车 = 尺寸/动力学分布扩展**，两者进统一策略 sim-agent 线（per-type heads，对齐 TerraZero 已发布口径）；挂车运动学不做。
3. 执行顺序：行人内核化先行（长尾场景急需）；骑行者/卡车随 sim-agent 线。

裁决依据：02 票披露的保真档位 + 复刻范围票的勾选（若未选异质则本票关闭并移出范围）。
