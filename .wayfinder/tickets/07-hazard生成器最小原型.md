---
id: "07"
title: hazard 生成器最小原型：在 gigaflow reset 上放长尾元素
labels: [wayfinder:prototype]
status: closed
assignee: zoo + 会话代理（2026-11-05 M0-M3 全验收）
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

### Owner 验收长期标准（2026-11-05 起，M1 两轮反馈）

- **图例单独成图**（legend.png，index.html 首项），不得挤入渲染帧遮挡地图元素。
- **验收渲染按类别着色，不按 agent id 彩虹着色**（id 色板会让车辆被误读成行人/骑行者）。
- 亚米级 hazard（锥桶等）标记有**强制最小像素尺寸**（全图 ≥1​2px），交付前做像素存在性审计。
- 图例不得列出未 spawn 的类别；预留条目须标「M2 未启用」或删除。
- 关键图（compare/zoom）加文字标注直接指认 hazard 位置；无 CJK 字体时用英文，不得豆腐块。

### 进度记录

- 2026-11-05：M0 ✅、M1 ✅（含两轮渲染修复）、M2 ✅（jaywalker 6/6，减速事件诚实零，cut-in abandoned+替代方案已录）；内核钉死+全量重渲（新锚点：baseline 200/31/0；static 增量 +116.7%/+133.3%/+16.0%）；kernel 溯源块+非本仓拒跑守卫入所有 summary。剩 M3。
- 2026-11-05（终）：M3 ✅（配置面 hazard_demo.yaml + 32 tests + mix 三 seed + 回归双门像素级全绿 + 既有内核测试 13+1 全绿）。票闭。

## Resolution

已解决（2026-11-05，owner 四里程碑全验收；实现 = worker 子代理 glm-5.3-flash，提交链 d2b09f4→9b65005→df7b8a8→de1b953→534508b→baf5e4f→b5a60c4）：

1. **交付**：`scripts/hazard_demo.py` 四模式（baseline/static-hazards/jaywalker/mix）+ `config/hazard_demo.yaml` 旋钮（init 前 fail-fast 校验）+ `tests/test_hazard_demo.py`（32 passed）。产物在 `artifacts/{baseline,static-hazards,jaywalker,mix}/`，全部带 kernel 溯源块，遵守长期渲染标准（独立图例/类别着色/像素审计/标注）。
2. **锚点（内核钉死后）**：baseline s42 = 200 步/31 碰撞/0 offroad；static 增量 +116.7%/+133.3%/+16.0%；jaywalker 6/6 横穿、min 1.012m、1 接触（s44）；mix 碰撞 b→m = 12→26/9→26/25→29。
3. **回归双门像素级**：旋钮全关 → 世界渲染与 baseline 逐字节一致（仅标题条 572px 差）；既有内核测试全绿 → 零内核改动铁证。
4. **已知偏差（全录 summary notes）**：行人 type=1（内核小车语义）；cut-in abandoned（替代 = per-agent IDM 参数）；贴边锥桶 lane-assoc 失败（无 gameplay 影响，源码已核）；减速事件诚实零。
5. **设计问题回答**：放置时机 = reset 后 Python 步进；合法性检查 = lane 锚定 + Python 侧 OBB 重叠（内核 checker 无 binding，未加薄封装）；配置面 = YAML 旋钮 + init 校验；观测 = 静态槽自动进 partner obs（recompute_observations）。
6. **复现**：`PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode {baseline|static-hazards|jaywalker|mix} --seed {42,43,44}`（每 seed ~20-60s）。
7. **后续出界项**：内核子系统化（C reset 内生放置/真 PEDESTRIAN type/per-agent IDM）→ 已入地图雾区，规格依赖 06/08 裁决。
