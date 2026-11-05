---
id: "07"
title: hazard 生成器最小原型：在 gigaflow reset 上放长尾元素
labels: [wayfinder:prototype]
status: open
assignee: zoo + 会话代理（M0 启动 2026-11-03）
blocked-by: ["01", "02", "11"]
---
## Question

做一个便宜、粗糙、可反应的原型回答"它该长什么样"：在 gigaflow 模式 reset 路径上放置第一批长尾元素——parked vehicles（沿边界半占道）、collided vehicles（交叠姿态）、construction zone（polygon + 降速区）、jaywalker（行人 agent + 横穿 goal）、cut-in 制造者（激进 IDM 变体）——并出可视化（viz 工具链已有）。

原型要回答的设计问题：放置时机（reset 全量 vs episode 中途动态注入）、合法性检查是否复用 `check_spawn_collision/offroad`、hazard 概率表的配置面形态、观测是否需要新特征列。产物链接进本票，供「复刻范围」与后续设计票收敛。

**补充（02 票，规格已可执行）**：八生成器 = 反应式车（IDM+pure-pursuit / PDM 式）/ parked（car/bus/truck）/ 碰撞簇（disc、rear-end、T-bone、fan）/ 施工（tiled cone grid、taper、lane-block + 静止工人）/ 静态碎片 / 斑马线行人（Poisson 班次+ORCA）/ jaywalker / log 回放骑行者。放置规则：lane 上拒绝采样、对齐行进方向、低速起步、交叉口 lane 排除；混合为配置面（非代码变更）。报告未披露：goal 弧长界、IDM 模式重标定系数、Poisson/ORCA 参数——原型自行定参并记录。

**再补（10 票）**：反钻空子补丁先例可复用（`check_lane_change_red_light`，drive.h:1526-1586）；IDM NPC 停灯**不停牌**（idm.h:185-223）——原型若用 IDM 做反应式背景车，需补停牌阻尼或明录口径差；红灯点扫越→footprint 扫越是顺手可做的最小补丁（0.5-1d）。

## Milestones（验收契约，owner 目测验收 / 代理实现）

每个 M 都满足：**一条命令可复现** + 产物落在 `artifacts/<name>/`（图片/GIF/HTML + summary.json）+ 同 seed 无/有 hazard 对比。M 完成即停，等 owner 目测验收后再进下一个。

### M0 基线可视化管线（先行）
- 命令：`PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode baseline --seed 42`
- 产物：top-down 渲染帧 ≥3 张 + GIF + index.html + summary.json（agents 数/步数/速度分布/碰撞计数）
- 看什么：gigaflow 随机撒车 + IDM 行车是否正常（车道行驶、无穿墙、信号停行）

### M1 静态长尾元素（parked / collided 簇 / 施工锥桶）
- 命令：同上 `--mode static-hazards --seed 42/43/44`
- 产物：同 seed 无/有 hazard 并排对比图 + 每类计数/放置成功率/放置合法性（无 off-road、与动态 agent 无重叠；碰撞簇内部重叠是有意的）+ 3 类×多 seed 计数直方图
- 看什么：路缘 parked 车、双车交叠碰撞簇、锥桶渐缩封道是否肉眼可辨

### M2 行为型长尾元素（jaywalker 优先；cut-in 视 per-agent IDM 参数可行性）
- 产物：行人横穿轨迹可视化 + 完成横穿率 + 与车辆最小距离分布；cut-in 若不可行明录原因与替代方案
- 看什么：行人从路侧横穿至对侧，车流是否有让行/减速反应

### M3 配置面 + 回归 + 混合演示
- 产物：`hazard_*` 概率旋钮进 config（init 前校验）+ C/Python 测试绿 + `--mode mix` 混合演示
- 看什么：一次 episode 内多类元素同场，且关掉旋钮即回到纯 gigaflow（回归不变）
