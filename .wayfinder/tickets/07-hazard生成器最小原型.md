---
id: "07"
title: hazard 生成器最小原型：在 gigaflow reset 上放长尾元素
labels: [wayfinder:prototype]
status: open
assignee: null
blocked-by: ["01", "02", "11"]
---
## Question

做一个便宜、粗糙、可反应的原型回答"它该长什么样"：在 gigaflow 模式 reset 路径上放置第一批长尾元素——parked vehicles（沿边界半占道）、collided vehicles（交叠姿态）、construction zone（polygon + 降速区）、jaywalker（行人 agent + 横穿 goal）、cut-in 制造者（激进 IDM 变体）——并出可视化（viz 工具链已有）。

原型要回答的设计问题：放置时机（reset 全量 vs episode 中途动态注入）、合法性检查是否复用 `check_spawn_collision/offroad`、hazard 概率表的配置面形态、观测是否需要新特征列。产物链接进本票，供「复刻范围」与后续设计票收敛。

**补充（02 票，规格已可执行）**：八生成器 = 反应式车（IDM+pure-pursuit / PDM 式）/ parked（car/bus/truck）/ 碰撞簇（disc、rear-end、T-bone、fan）/ 施工（tiled cone grid、taper、lane-block + 静止工人）/ 静态碎片 / 斑马线行人（Poisson 班次+ORCA）/ jaywalker / log 回放骑行者。放置规则：lane 上拒绝采样、对齐行进方向、低速起步、交叉口 lane 排除；混合为配置面（非代码变更）。报告未披露：goal 弧长界、IDM 模式重标定系数、Poisson/ORCA 参数——原型自行定参并记录。

**再补（10 票）**：反钻空子补丁先例可复用（`check_lane_change_red_light`，drive.h:1526-1586）；IDM NPC 停灯**不停牌**（idm.h:185-223）——原型若用 IDM 做反应式背景车，需补停牌阻尼或明录口径差；红灯点扫越→footprint 扫越是顺手可做的最小补丁（0.5-1d）。
