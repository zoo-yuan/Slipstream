> Wayfinder 10 票调研产物。执行：council-glm 子代理（逐行读码，全部行号实测于 ../PufferDrive 当前工作树）。日期 2026-11-03，主会话代为落盘。

# 规则几何执法核对（票 10）

基准：TerraZero §3.3（`.wayfinder/research/raw/terrareport-fulltext.txt:577-597`）——违规三通路：reward 惩罚 + 进评测指标 + 可配置后果（stop/remove）。以下行号均实测于 `../PufferDrive/pufferlib/ocean/drive/` 当前工作树。

## 1. 红灯 = swept path 跨 stop bar —— **已具备（一处小口径差）**

- 判定路径：`check_stop_line_crossing_event`（drive.h:1451-1508）。**边缘触发**：后保险杠中点 prev→cur 的符号跨越（s_prev<0 → s_cur≥0，drive.h:1489-1497）+ 当前 tick 信号态为 RED（drive.h:1468-1470）+ 10 m 距离门（1475）+ π/4 heading 门（1478）+ 横向落点在延长 1.5× 线段内（1504）。红线内停车、绿灯再走 → 不罚，与 TerraZero「not merely for waiting within the region」一致。
- 反钻空子补丁：`check_lane_change_red_light`（drive.h:1526-1586）——连续两 tick 红（1551-1553）期间换道进入已越线的受控车道即违规。
- 口径差：TerraZero 是 **footprint（swept path）** 跨线；PufferDrive 扫的是**后保杠单点**（注释自认 CARLA 惯例，drive.h:1452-1453）。另有前置 `current_lane_idx != -1`（1590-1592）——该步未匹配到车道的 agent 逃逸；执法还耦合观测配置 `obs_slots_traffic_controls_n`（3546）。
- **工作量**：点扫越改 OBB 扫越可复用 `check_segment_crosses_moving_box`（idm.h:221 已有样例盒×延长线段先例）+ 解开 lane/obs 耦合 ≈ **0.5-1 工程日**。

## 2. 停牌 = hold-below-speed + sampled dwell —— **已具备（dwell 未采样化）**

- `check_stop_sign_violation`（drive.h:1642-1709）：速度阈值 `AGENT_STOPPED_SPEED_THRESHOLD = 0.2 m/s`（constants.h:310，应用于 1683-1684）；驻停计数 `stop_sign_stopped_timestep_count` 仅在车头在线前且低于阈值时累加（1680-1684, 1703-1704），要求 `STOP_SIGN_REQUIRED_STOP_DURATION_SECONDS/dt` 步（constants.h:182 = **固定 0.5 s**，1687）；前保杠越线事件时结算（`agent_front_crossed_stop_line` 1607-1639 → 1692-1694）；带 z 门（1659）与 heading 门（1651-1652）。
- 口径差：TerraZero 是 **sampled dwell time**（按实例采样）；PufferDrive 为全局常数。
- **工作量**：常数改 per-control 采样字段（`TrafficControlElement` 加一字段 + reset 侧采样）≈ **0.5 工程日**。

## 3. off-road 带 elevation gate —— **已具备**

- 主判定（drive.h:3340-3344）：road-edge 线段先过 z 门 `|element->z − agent->sim_z| > Z_BUFFER(4.0 m, constants.h:204) → 跳过`，再做 `check_segment_crosses_moving_box`（swept footprint×线段）。spawn 检查同款 z 门（2441-2445）。**off-road 判定确实带 elevation gate**，票面的疑虑可消除。OBB 碰撞 z 早拒在 `check_obb_collision`（1715-1719）与 swept 版（`check_moving_obb_collision` 开头，≈1780-1785；票面 1709 是旧行号）。
- 兜底路径：无邻居实体即 off-road（3319-3321）、offgrid（3281-3285）——无高差误报面。
- 口径差：TerraZero 测 road-edge **和 sidewalk 边界**；PufferDrive 边界类型仅 `ROAD_EDGE_BOUNDARY/MEDIAN`（constants.h:51-53），无 sidewalk 类型（misc 仅 CROSSWALK/SPEED_BUMP，constants.h:55-57）。
- **工作量**：若地图源含 sidewalk 数据，加类型+摄取 ≈ **1-2 工程日**；无数据则 0。

## 4. lane direction 执法 —— **已具备，与 TerraZero 口径同构**

- 现行即 heading-residual 路线：θ_f = agent heading vs 最近车道多段加权 heading（drive.h:3368-3447 段，`element->headings` 加权平均）；`LANE_ANGLE_IDX = cos(θ_f)`（3476）；`wrong_way_distance` 在速度 >2.5 m/s 且 cosθ<0 时按里程累积（3502-3508）；评测层 `direction_compliance` 分档乘入 score multiplier（2089-2093）；奖励层 `lane_align`（against-lane 惩罚 = min(cosθ,0)，3647-3655）、`lane_center`（3659-3665）、倒车惩罚（3690-3694）。
- **无 masking**：`apply_infraction_behavior` 只挂 collision/offroad/red-light/stop-sign（3527-3556），逆行仅罚不禁——正是 TerraZero「penalized but never masked」。handedness 由车道几何 heading 携带，亦一致。现有 `lane_align/lane_center` 奖励项与执法口径同向，无需另立「heading residual 执法」新件。
- **工作量**：0（可选：把 wrong_way 从评测指标升为 reward 项，0.5 日）。

## 5. per-class 差异 —— **半具备（行人豁免缺失）**

- `compute_metrics` 在 c_step 对所有 active agent 无类型过滤调用（drive.h:4673-4679）；红灯/停牌检查体内无任何 `agent->type` gate（1451-1508、1642-1709 全文核对）。类型常量 `VEHICLE/PEDESTRIAN/CYCLIST` 存在（datatypes.h:33-35）但执法路径不用。
- 现状：骑行者按车辆处理 = **默认即符合**；行人**无显式豁免**——只有几何副作用（行人横穿时 heading 与 tc->heading 差 >π/4 被 1478 跳过），沿车道方向走的行人可被误判。
- **工作量**：`type == PEDESTRIAN` 早退 gate + 回归测试 ≈ **0.5 工程日**。
- 附注（供 07 票）：NPC 侧 `idm_sample_hits_red_light`（idm.h:185-223）把红+黄都当虚拟前车（198），NPC 会停灯；但 IDM **无停牌处理**——NPC 闯停牌无阻尼。

## 6. 违规后果 stop/remove —— **已具备**

- `INFRACTION_BEHAVIOR_{IGNORE,STOP,REMOVE}`（constants.h:99-101）；`apply_infraction_behavior`（drive.h:480-486）置 `stopped/removed`；四独立旋钮 `collision/offroad/traffic_light/stop_sign_behavior` 分别在 offroad（3529）、碰撞（3541）、红灯（3548）、停牌（3555）调用；binding.c:1979-1982 unpack，Python 默认全 `"ignore"`（drive.py:41-44, 172-175）。三通路齐备：reward（3617-3631，红灯/停牌共用 `REWARD_COEF_STOP_LINE`）、eval 指标（3621/3629 → 汇总 2149-2153 → multiplier 2086/2093）、后果。优先级互斥注释 3522-3524。
- 小瑕疵（非缺口）：红灯与停牌共用一个惩罚系数；惩罚为事件步一次性。
- **工作量**：0（拆分系数 0.5 日，可选）。

---

## 摘要（≤15 行）

1. 总判定：**六项中 4 已具备、1.5 半具备、0.5 视数据而定**——02 票「几何执法升承重墙」在 PufferDrive 基底上基本是修补而非新建。
2. 实缺口仅三处，合计 ≈2-4 工程日：红灯点扫越→footprint 扫越（0.5-1d）；停牌 dwell 采样化（0.5d）；行人路口豁免 gate（0.5d）；sidewalk 边界类型视地图源（0-2d）。
3. 票面疑虑「off-road 是否带 elevation gate」：**带**（drive.h:3340-3344，Z_BUFFER=4 m），spawn 检查亦带（2441-2445）；OBB z 早拒现行位于 1715-1719（票面 1709 为旧行号）。
4. lane direction 无需新件：现行 heading-residual + 罚不禁 + 几何携带 handedness，与 §3.3 同构；建议只把 wrong_way 升权进 reward。
5. 反钻空子已有先例（`check_lane_change_red_light` 1526-1586），说明「RL 找执法漏洞→打补丁」的迭代模式在此代码库成立，07 票 hazard 原型可复用该手法。
6. 立场：优先删繁就简——不要为对齐 TerraZero 措辞新建执法框架，只做上述三个最小补丁；最大反方论点：footprint 扫越在密集路口的算力成本与点扫越差异未测，若成本显著可维持现状并文档化口径差。置信度：高（全部结论基于逐行读码）；会改变结论的证据：`obs_slots_traffic_controls_n` 耦合的实际影响面、地图源是否含 sidewalk 几何。
