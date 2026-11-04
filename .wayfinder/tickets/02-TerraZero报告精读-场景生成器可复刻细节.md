---
id: "02"
title: TerraZero 报告精读：场景生成器全部可复刻细节
labels: [wayfinder:research]
status: closed
assignee: council-minimax（子代理，第二次发射）
blocked-by: []
---
## Question

技术报告 arXiv 2607.13028 + 项目页（含 demo 可配参数）里，关于以下七项到底披露了什么、怎么做的？逐项给出可复刻级别的细节，未公开处明确标注"未披露"：

1. **程序化场景生成器**：起终点/goal 采样分布、对手数量与行为分布、信号配时随机化参数、hazard（施工/静态障碍/横穿行人/parked/collided 车）的类型清单与放置规则、场景编辑接口形态。
2. **异质 agent 建模**：车/卡车/行人/骑行者各自的动力学保真档位；"统一策略控异质"（页面标注 in preparation）与当前发布物的边界。
3. **地图摄取（"任何 log 当地图"）**：输入格式、从 log 中抽取什么（仅几何？交通控制配时从哪来？）、3D 表示的具体含义。
4. **仿真架构**：C 引擎 + GPU 零拷贝推理的机制细节、吞吐测量口径（agent steps/s 的定义）。
5. **零演示 recipe**：reward 结构与随机化、PPO 设置、规模（8×10¹¹ 步）、稳定性手段、动作空间与观测表示。
6. **评测协议**：val14 / InterPlan / WOSAC 的提交口径与复现要求。
7. **可迁移工程判断**：哪些细节是"段位差"的承重墙（对照 11-03 文档五成分框架），哪些是锦上添花。

产出：`.wayfinder/research/terrareport-findings.md`（每条论断附来源：报告章节/原文引句/页面段落）。

## Resolution

已解决（2026-11-03，council-minimax 精读全文，成果见 [terrareport-findings](../research/terrareport-findings.md)）：

- **八生成器清单可直接当 07 票规格**：反应式车（IDM+pure-pursuit / PDM 式闭环）/ parked 车（三尺寸类）/ 碰撞簇（disc/rear-end/T-bone/fan 四布局）/ 施工区（tiled cone grid、taper、lane-block 三封道+静止工人）/ 静态碎片 / 斑马线行人（Poisson+ORCA）/ jaywalker / log 回放骑行者；放置=lane 上拒绝采样+交叉口 lane 排除；「新组合是配置变更而非代码变更」。
- **信号层全规格**：NEMA 八相位 / Christmas（每 stop line 独立红→绿→黄，LogNormal dwell）/ round-robin；逐路口初始相位随机 + 保护左转翻 permissive（都 undisclosed 参数除外）。
- **reward 全表公开但无 progress 项**：Table 8 逐项区间/per-type 权重/23 维三模式；goal progress 与 target speed 置零剔除 → InterPlan 进度项最低（67.1）；**复刻须裁决补 progress shaping**。
- **评测期条件化机制**：同一 checkpoint 免重训，靠 reward/运动学条件化观测切换 val14/InterPlan 风格；但评测期权重向量未公开。
- **承重墙重估**：几何规则执法（红灯=swept path 跨 stop bar、停牌 dwell、elevation gate、heading residual）升为承重；“3D”仅为 z 感知几何、吞吐零拷贝为工程形态、卡车保真=自行车+轮胎侧偏力——均降权。
- **三处口径修正**：8×10¹¹ 步出自 TerraTransfer/++ 页而非报告；车+人+骑统一策略已在报告发布（仅车+卡未发布）；两侧均无场景编辑器，demo 只是生成器上的筛选器。
