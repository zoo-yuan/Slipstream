> Wayfinder 02 票调研产物。执行：council-minimax 子代理（第二次发射；输入为本地全文 .wayfinder/research/raw/terrareport-fulltext.txt，主会话代为落盘）。
> 日期 2026-11-03。引用格式：报告章节/表号 + 原文引句；未披露处显式标注。

# TerraZero 报告精读（票 02）" title, then 7 sections with citations, then ≤30-line summary.

Also since I'm the council advisor with lateral breadth stance, I'll add a short "横向视角 / 修正与新发现" section, plus the report shape elements (novel angle, when it wins/loses, confidence) — but the task says the note ends with a ≤30-line summary of the 7 items + top-3 report-vs-page findings. I'll fold the council elements into the summary block and a short section, keeping it disciplined. Total length: this is a deliverable artifact, so longer than 400 words is acceptable ("unless asked otherwise" — the task asked for a complete structured note).

Let me be careful to cite precisely: file paths + section names.

Key citations:
- .wayfinder/research/raw/terrareport-fulltext.txt §3.1, §3.2, §3.3, §3.4, §4.1–4.5, §5.1, §5.2, §6.1–6.6, App A (Tables 7), App B (Table 8), App C (Tables 9–12), App D (Alg 1, D.1–D.5, Table 13), App E (Tables 14–15).
- .wayfinder/research/raw/terrzero.txt (project page).
- .wayfinder/research/raw/TerraTransfer.txt:86,147,181-183; TerraTransferPlusPlus.txt:181,283.
- ../PufferDrive/docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md (§2.1 five components, §3/§4).
- .wayfinder/map.md.

Let me double check TerraTransfer.txt line numbers for the 8e11 claim: line 86 "≈2,535 yrs of self-play driving experience (8×10¹¹ steps)", line 147 "TerraZero · massively-parallel self-play 8×10¹¹ steps", line 181 "8×10¹¹ simulated steps", 183 "≈23M driving hours". Good.

And TerraTransferPlusPlus.txt:283 "8.0 × 10^11 steps in simulation", 284 "224M miles", and line ~181 "801 B simulated steps in self-play" (that's the TerraTransfer++ total self-play steps, 801B). Good detail: TerraTransfer++ claims 801B simulated steps in self-play (their own training), while attributing 8.0×10^11 steps to TerraZero's self-play.

Wait — careful: line 181 "801 B simulated steps in self-play" appears in the TerraTransfer++ header stat; line 283 "8.0 × 10^11 steps in simulation" appears in a "TerraZero · massively-parallel self-play" panel with "224M miles", "469 × Earth-to-Moon", "2,535 years". So the 8×10^11 is TerraZero's self-play scale, and 801B is TT++'s own self-play steps. Good.

One caveat: the report says nothing about 801B either. Fine.

Also note: the project page (terrzero.txt) does NOT mention 8e11 either — consistent with my finding.

Now let me make sure about "Mixed vehicle, pedestrian, and obstacle generators" from Table 15 (planner road users) and "Vehicle generators only; pedestrians and cyclists are policy-controlled" (sim agent). Important replicable spec: the planner's training scenes include pedestrian generators (crosswalk/jaywalk) and obstacle generators; the sim agent only uses vehicle generators because peds/cycs are policy-controlled.

Also Table 15: kinematic coefficients — planner all four [0.5,1.5]; sim agent: vehicles [0.5,1.5], cyclists narrower ranges, pedestrians fixed. Nice.

Goal dropout 0.3 for both planner and sim agent (Table 15).

Now about the throughput measurement口径 (item 4): 
- Report §6.1: "Measured on a representative training job, TerraZero sustains 560K agent-steps per second on a single consumer GPU, 1.3M on a single server-grade GPU, and 2.8M on an 8-GPU server node". 
- Table 1 caption: "Numbers marked † are figures reported by the original authors; all other SPS values are benchmarked under our setup." So GPUDrive 11K/22K, PufferDrive 320K/745K, Waymax 319K were benchmarked by TerraZero authors under their setup — that's a claim of a fair re-benchmark, but the protocol (env count, whether learning is included, whether the same scenario content) is NOT disclosed. Explicitly flag: SPS counting convention, env count, whether policy inference+learner time is included, and whether all simulators ran the same scenario content = 未披露.
- 2.8M on 8-GPU node is not 8×1.3M (=10.4M) → sublinear scaling; and the planner ran on 16×A100 while throughput numbers are for 1/8 GPUs. Note: the throughput table's "8× server node" is 8 GPUs, while the planner training used 16 A100s — and 16 A100 isn't a node config reported. Fine.
- Also "agent-steps per second" counts *all* agents (NPCs + policy-controlled) or only policy-controlled agents? NOT disclosed — this matters hugely for comparison (PufferDrive's SPS convention counts... unknown). Flag as key 未披露.

Also: zero-copy details: C engine writes simulation state directly into shared numpy buffers read as PyTorch tensors; 16-bit observations reinterpreted bit-for-bit; buffers sized to controlled agents present rather than padded; NUMA pinning; host-to-device transfers overlap kernel launches; PufferLib vectorization; C engine compiled into a CPython extension. Good replicable mechanism description, but no code-level API (shared memory arena layout, H2D copy engine vs CUDA IPC) — 未披露 whether it's true zero-copy (device-mapped shared memory) or just pinned-buffer async copy. Flag: "fully zero-copy" claim but mechanism details (cudaHostRegister / IPC handle) undisclosed.

Item 3 map ingestion: "an offline pipeline converts each dataset into a single common representation (inspired by ScenarioNet), separate converters for Waymo/nuPlan/CARLA, a builder compiles shared representation into simulation-ready geometry: identifies intersections, ties stop lines and traffic signals to their lanes, resolves legal travel directions and speed limits. Intersections come from a source's own junction geometry where it provides one and otherwise inferred from lane topology. The builder validates each scene and emits compact binary terrabin. At training time scenarios are streamed and sampled so each batch stays balanced across regions and scenario types without holding the full dataset in memory." Also Algorithm 1 step 1: "Load terrabin scenario (map geometry, logged trajectories, signal phases)". Also §6.2 scenario counts: WOMD ~576K scenarios (487K train split); nuPlan ~262K across 4 cities (Las Vegas largest, Boston smallest); CARLA 5 towns. Good for scale replication.

Also §2.2 note: "WOMD, for instance, carries sparse traffic-signal annotations and very few cyclists" — an ingestion caveat.

Item 5 zero-demo recipe: reward structure (Table 8 + 23 params + 3 modes + normalization + reward-conditioned obs), PPO settings (Table 14), policy architecture, action spaces (jerk grid for vehicles; unicycle grid for peds; compact bicycle grid for cyclists), observation representation (Tables 9–12 with exact dims and scaling), goal dropout, observation noise model (off by default: clipped Gaussian perturbations + slot dropout), stability (V-trace, PopArt, prioritized sampling, truncation-aware bootstrap), distribution (16 A100 for planner / 32 for sim agent; data parallel + synchronized stats), scale (8×10^11 steps NOT in report; only epochs 12,000/4,000 for WOSAC checkpoints).

Action space discretization: jerk grid — number of discrete actions NOT disclosed; unicycle grid / compact bicycle grid sizes not disclosed. Important gap.

Now item 6 evaluation protocol: val14 (PlanTF suite, 14 scenario types balanced, nuPlan closed-loop reactive score, LQR controller executes actions), InterPlan (8 families, official 80-scenario split + full 335), WOSAC 2023 (full Waymo val minus >128-agent scenarios, as Gigaflow), WOSAC 2024 (CAT-K manifest filtered under 128-agent cap, vehicles + VRU tables, 32 stochastic rollouts, realism = 0.20 kin + 0.45 interactive + 0.35 map, VRU drops TTC and renormalizes). Transfer study protocol (WOSAC-2024 vehicle realism over 80-frame horizon, 32 rollouts; NAVSIM v2 EPDMS core safety = min of 4 sub-metrics; ~100 scenarios per cell; ego only, others replay logged).

Also: eval-time conditioning detail (reward/kinematic-conditioned observations, no retraining, goal/route sampling aligned to lane-level goals). And undisclosed: which reward weight vector is used at eval, how the policy's jerk actions are mapped to the nuPlan LQR controller.

Item 7 承重墙 judgment: 
- 承重墙 (load-bearing): (a) hazard/long-tail generator (§4.4) — direct causal claim; (b) 真实规则几何执法 (§3.3) — because reward penalizes violations, RL exploits loopholes; (c) per-episode reward+kinematics randomization + reward-conditioned obs (§5.1) — prevents overfitting one objective AND enables inference-time steering; (d) heterogeneous dynamics multi-model (§3.2) — only for sim-agent role; (e) 吞吐/零拷贝 — engineering, not capability, only at 10^11 scale.
- 锦上添花: PopArt/V-trace/prioritized sampling (stability, orthogonal to capability); goal dropout (regularization); observation noise (off by default, robustness experiments); "3D map" as advertised (it's z-aware geometry: elevation gate, z-separation 2.5m, 3D traffic entity positions) — cheap, not a wall; truck fidelity (bicycle + cornering forces) — thinner than the project page's "high-fidelity trucks" implies; 3.5M compact MLP (leaner observation set is a *feature* that helps when map topology is imperfect).
- Corrections to the 11-03 doc: (i) 8×10^11 not in the report; (ii) "异质统一策略 in preparation" is outdated for veh+ped+cyc (report §6.4 has it, and beats Gigaflow 0.632 vs 0.619) — only the car+truck unified policy isn't in the report; (iii) "高保真卡车" = bicycle + tire cornering forces only; (iv) "3D 地图" is z-aware object-level geometry, not rendering.
- Also novel: report's own admission that their progress component is the weak spot (val14 ego progress 92.91 vs Gigaflow 93.6; InterPlan route progress 67.1) because they zeroed the progress reward term → replication risk: copying their reward verbatim inherits a progress deficit; a replication should add progress shaping (their reward explicitly holds goal-progress and target-speed terms at zero). This is a strong, actionable, non-obvious finding.

Let me double-check that claim: Table 8 caption: "Goal progress and target speed are held at zero across both reported policies and are omitted, so the full parameter vector has 23 entries". Hmm, "Goal progress" might be a term separate from "Goal" bonus. The listed terms in Table 8 are: Goal, Collision, Boundary, Comfort, Lane Align, Lane Center, Velocity, Vel Align, Reverse, Timestep, Stop Line, Red Light, Road Incursion, Speed Limit, Edge Preference = 15 terms (some with 2 params). The caption says "Goal progress and target speed are held at zero" — meaning in the underlying Gigaflow-shaped reward there are progress/target-speed terms that are set to zero and omitted from the observation vector (23 entries). So indeed no progress reward. Meanwhile the Velocity term (+α_vel·Δt·v/v_max) rewards moving fast (but not progress along route). So yes: no route-progress shaping → their progress metrics are the weakest (InterPlan route-progress 67.1 is lowest component; val14 ego progress 92.91 trails Gigaflow 93.6). Solid, and it directly informs the replication: "progress reward term must be added or your policy may learn to loiter".

Careful: their "Goal" bonus is fixed 1.0 and gives a large terminal reward for reaching the goal, which indirectly rewards progress. And InterPlan score's route-progress term is from the nuPlan scorer (route progress along the recorded route), not the reward. So the claim: reward lacks a route-progress shaping term; the terminal goal bonus is the only route-progress incentive; consequence: weak route-progress on InterPlan (67.1) and slightly-below-Gigaflow ego progress on val14. Good.

Now, the confidence section and "when it wins/loses" for my council role. Let me structure the final answer:

```
# TerraZero 报告精读（票 02）
（来源与口径说明）
## 0. 读源与口径
## 1. 程序化场景生成器
## 2. 异质 agent 建模
## 3. 地图摄取
## 4. 仿真架构与吞吐口径
## 5. 零演示 recipe
## 6. 评测协议
## 7. 承重墙判断
## 8. 横向视角：报告 vs 项目页 vs 本地 11-03 文档的四处修正
## 摘要（≤30 行）
```

Then a short council block (novel angle / when it wins / loses / confidence) — I can fold into §8 and the summary.

Length: this will be long (maybe 2000+ words). That's appropriate for the deliverable. But the council advisor role says keep under 400 words unless asked otherwise — the task explicitly asked for the full structured note. I'll keep the note dense and skimmable (bullets, tables), which is fine.

Let me be careful about a few factual details:

- §3.3 "Stop signs require the agent to hold below a speed threshold for a sampled dwell time before the line clears." — sampled dwell time: value undisclosed.
- §3.3 "can trigger a configurable consequence such as stopping or removing the offending agent" — which consequence is used in training: undisclosed.
- §4.4 reactive vehicles: "IDM longitudinal controller paired with pure-pursuit steering, or by a closed-loop planner in the style of PDM-Closed that forward-simulates a grid of lateral-offset × IDM-parameter proposals and selects the trajectory maximizing a weighted score of progress, time-to-collision, comfort, and lane-keeping. IDM vehicles sample among default, assertive, and cautious behavior modes that rescale the IDM gains, and may reroll that mode within an episode" — the rescale factors and reroll probability undisclosed.
- Table 7 "Max travel directions per lane 32".
- §4.5: "the initial phase of each intersection is randomized at reset, and protected left turns are flipped to permissive by an independent per-intersection draw".
- Christmas controller "cycles each stop line independently through red, green, and yellow with dwell times drawn from a LogNormal distribution, so signals across a map stay uncoordinated" — dwell time LogNormal params undisclosed. (Note: order red, green, yellow — unusual, matches "Christmas" tree? fine, quote it.)
- Round-robin: "one approach leg holds green at a time, and the green window rotates leg by leg in canonical NEMA order".

- §6.2: "in bf16 mixed precision distributed across 16 NVIDIA A100 80GB GPUs".
- Table 14: "Total agents per GPU 512; Environments per GPU 32; Vectorized batch size 4; Num minibatches 16".

- §5.1 MDP: "up to 20 partner agents within 50 m, and up to 200 nearby road segments". Table 7 adds max traffic entities 16 @100m.

- Observation: planner ego features table: agent_type 1, rel_goal_pos 2, goal_dropout_flag 1, state_dropout_flag 1, speed 1, vehicle_dims 2, collision_state 1, steering_angle 1, acceleration 2, stop_sign_state 3 (optional), alpha_params 19 (optional), kinematic_params 4 (optional). Note "state_dropout_flag" — indicates ego-state dropout robustness feature exists (in Table 9), while §5.1 mentions "optional observation-noise model, off by default, that adds clipped Gaussian perturbations and slot dropout". Also Table 10 partner acceleration disabled by default.

- §5.1 "Vehicles act in a discrete control space, in either acceleration or jerk."

- Sim agent: "jerk for vehicles, a unicycle grid for pedestrians, and a compact bicycle grid for cyclists (Appendix E)".

OK. One more: does the report describe the WOSAC "9.1 s WOMD recording at 10 Hz, first 1.1 s as initial context, remaining 8 s, up to 128 agents, 32 stochastic closed-loop joint rollouts"? Yes §6.4.

And transfer study: "each cell averages about 100 scenarios", "80-frame future horizon", "Agents are initialized from the logged state but pursue randomly generated goals".

Let me now also note the WOSAC 2023 caveat: "The 2023 edition uses the full validation split after removing scenarios above the 128-agent challenge cap, as in Gigaflow".

Alright. Also I should mention the "Demo" page interactive controls briefly for item 1's "场景编辑接口形态": page: "choose the agent type (car, truck, or heterogeneous†), set a traffic condition, drop in road users and hazards, optionally dial in a robustness setting, and pick a viewpoint (agent or top-down) — the simulator procedurally generates the rest, and the selectors narrow to whatever it has on hand." Note the phrase "the selectors narrow to whatever it has on hand" — i.e., the demo UI is a *filter over generated content*, not an authoring tool. That's a lateral insight: the page's demo is a *sampler* of the procedural generator, and it reveals that even the demo doesn't allow authoring specific hazards — matching the report's "configuration change rather than a code change" and "a configurable mix of them that varies from episode to episode". So there is NO scenario editing interface in either source: only configuration + randomized sampling. This directly informs ticket 07 (hazard 最小原型): the upstream product is "hazard generators + a mixture config", not "a scenario editor". Strong finding.

Let me write it up now. I'll include line references to files (file path + section), since claims must cite files actually read.

I'll produce the notes. Aim for completeness with tight prose.

For the ≤30-line summary at the end: 7 items × 1-3 lines + top-3 report-vs-page findings. I'll count lines carefully (use bullet lines).

Let me write the final response.</think>

我已读完报告全文（2597 行，正文 + 附录 A–E）与四个 Terra 项目页、11-03 五成分文档。以下是交付正文（无写文件工具，故在此落盘内容）。

---

# TerraZero 报告精读（票 02）

**读源与口径**：主源 `.wayfinder/research/raw/terrareport-fulltext.txt`（arXiv 2607.13028v2 全文抽取，下称「报告」）；项目页 `.wayfinder/research/raw/terrzero.txt`；系列页 `TerraTransfer.txt` / `TerraTransferPlusPlus.txt` / `SPACeR.txt`；本仓地基 `../PufferDrive/docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（下称「11-03 文档」）。下述每条均给报告章节/表号或原文引句；未披露处显式标注。

## 1. 程序化场景生成器

**生成器总纲**：报告 §3.2 给出六条可组合配置轴——「agent classes, per-class dynamics, initialization, signal control, rule-based road users, and per-episode randomization」，并明确「A new combination is a configuration change rather than a code change」。§3.2 另点明 road-user 轴「admits any subset of its eight generators (Section 4.4)」。**「八生成器」是报告唯一给出的生成器计数口径**，项目页无此数字。

**初始化模式（§4.1 + 附录 D）**：三模式 log / random / hybrid。默认训练用 random。
- random：「agents are procedurally placed per class under rejection sampling, so that every agent starts collision-free and compliant with the traffic rules, on a lane from which a sufficiently distant goal is reachable」；agent 数量本身「sampled from a configurable range whose upper bound may exceed the logged agent count」（D.1：「scenes can pack denser than any naturalistic recording」）。
- 放置细节（D.1）：落在 lane 上、与该 lane 行进方向对齐、低速起步，重采样直到无碰撞且在路面上；**「Lanes inside signalized intersections are excluded so agents do not begin mid-intersection」**。
- hybrid（D.3）：以 environment 为粒度的独立抽样概率 `p_random ∈ [0,1]`，不是场景内混合——「rather than a within-scene mixture」。D.3 给出 hybrid 的用途论证：log 初值能产生 random 难出现的「at highway speed in a merge lane」类连贯交通态，代价是状态多样性下降。

**Goal 采样（§4.2 + D.5）**：沿 lane 连通图前向游走、路口处「following random candidate lanes at junctions」，目标弧长从有界区间抽样；「An alignment filter rejects candidate goals that fall behind the agent」，找不到对齐 goal 的 agent **被移出场景**。log 模式下 goal = 该 agent 记录轨迹的最终位置，并先验证在 lane 图上可达。goal 达成后按配置「either receives a freshly sampled goal further along the topology or halts in place」。
> **未披露**：D.5 结尾明说「The sampling bounds and filter thresholds are configurable, and we omit their values」——弧长上下界、过滤阈值**报告明确拒绝给出**。

**对手/交通参与者清单（§4.4 + Figure 4 图注）**：八生成器按 §3.2 计数可复原为 ①反应式车（IDM+pure-pursuit，或 PDM-Closed 式闭环规划器）、②路侧 parked 车（car/bus/truck 三尺寸类）、③碰撞车簇（uniform disc / rear-end chain / T-bone / outward fan 四布局）、④施工区（tiled cone grid / taper / lane-block 三种封道 + 可选静止工人）、⑤孤立静态障碍（road debris，偏离车道中心）、⑥斑马线行人（ORCA 互惠避让）、⑦ jaywalker（mid-block，垂直于道路走至对侧）、⑧骑行者（**无脚本控制器，NPC 仅由 log 回放产生**）。
- 反应式车可「reroll that mode within an episode」，三种行为模式 default/assertive/cautious「rescale the IDM gains」（增益重标定系数**未披露**）；PDM 式规划器「forward-simulates a grid of lateral-offset × IDM-parameter proposals」并按 progress/TTC/comfort/lane-keeping 加权打分（提案网格尺寸、权重**未披露**）。
- 行人：「arriving on a Poisson schedule sampled per crosswalk as single crossers, packs, bidirectional flows, or staggered sequences」；jaywalker「walking perpendicular to the road until they clear the far edge」（Poisson 率、ORCA 参数、班次参数**未披露**）。
- 施工/静态障碍的**车道占用几何**在 Figure 4 图注里可直接读出：whole-lane 帧上的 tiled cone grid（矩形锥桶阵列）、diagonal taper（斜向渐变收窄）、perpendicular lane-block（垂直封道），每种配一名 stationary worker，另有随机路面碎屑。

**信号配时（§4.5）**：三种控制器。
- NEMA：标准双环八相位并发方案（引 NEMA TS 2-2003），两环交叉屏障保证冲突流不同时放行。
- Christmas：**默认训练组合采用**——「cycles each stop line independently through red, green, and yellow with dwell times drawn from a LogNormal distribution, so signals across a map stay uncoordinated」（LogNormal 参数**未披露**；注意相序是红→绿→黄）。
- Round-robin：「one approach leg holds green at a time, and the green window rotates leg by leg in canonical NEMA order」（绿灯窗口时长**未披露**）。
- 每场景额外两处随机：「the initial phase of each intersection is randomized at reset」，且「protected left turns are flipped to permissive by an independent per-intersection draw」。
> 这是**报告有、项目页完全没有**的一整层：项目页只说「varied traffic-signal timing」（`terrzero.txt`），没有任何控制器种类/相位随机/保护左转翻转的披露。

**尺寸随机（§4.3 + D.1）**：每个 agent 的 bbox 尺寸按类型均匀区间独立采样，车辆再分 car/truck/bus 尺寸类（「compact cars to long buses and trucks」）。**具体尺寸区间未披露**（项目页也无）。

**场景编辑接口形态**：报告**没有**任何交互式场景编辑器；最接近的表述是配置面 + 每 episode 采样组合（§3.2、§4.4「a scene draws a configurable mix of them that varies from episode to episode」）。项目页 demo 同样不是编辑器：`terrzero.txt` 原文「choose the agent type…, set a traffic condition, drop in road users and hazards, optionally dial in a robustness setting, and pick a viewpoint… **the selectors narrow to whatever it has on hand**」——即 demo 只是**生成器上的筛选器/采样器**，不可定点编排 hazard。

## 2. 异质 agent 建模

**动力学档位（§3.2「Per-class dynamics」）**：行人 unicycle；骑行者与小车 bicycle；卡车「bicycle model augmented with tire cornering forces」。「Cars are actuated at the jerk level… while the other classes are actuated by acceleration.」
> 关键口径修正：11-03 文档称 TerraZero 有「高保真卡车」，报告的实际定义只有**自行车模型 + 轮胎侧偏力一项**；报告自己批评 Gigaflow「covering trucks as size variants of the same box」。即 TerraZero 对 Gigaflow 的卡车差异化**仅是一个额外动力学项**。轮胎模型参数、侧偏刚度、挂车几何**未披露**。

**统一策略控异质 —— 边界与页面冲突（§3.2/§5.1/§6.4）**：报告三处自述已实现统一策略控异质：
- §3.2：「A single shared policy controls all classes through per-type action masking」。
- §5.1：「a heterogeneous sim agent (Section 6.4) that drives all three classes through separate per-type action heads over their own dynamics models: jerk for vehicles, a unicycle grid for pedestrians, and a compact bicycle grid for cyclists」。
- §6.4：「Unlike Gigaflow… TerraZero controls vehicles, pedestrians, and cyclists through **one unified heterogeneous policy**」。
评测上 Table 5/6 给出 WOSAC 2024 车辆 0.740 / VRU 0.688，且 §6.4 明说「makes it suitable for unified, interactive traffic-scene simulation」。
> 项目页 `terrzero.txt` 的脚注「† Heterogeneous agents controlled via a unified policy will be released in work in preparation」——按报告，脚注只对**车+卡车（demo 下拉的 "heterogeneous" 项）**成立；**车/行人/骑行者统一策略已在报告内发布并评测**。此为「项目页落后于报告」的直接证据。

**保真档位分布（Table 15）**：planner 仅控车辆（3.5M 参数 MLP）；sim agent 控三类（6.7M 参数，per-type heads）。行人/骑行者**不能**作为 NPC 脚本（骑行者仅 log 回放）；sim agent 训练场景只用 vehicle generators，因为行人/骑行人由策略接管（Table 15「Road users: Vehicle generators only; pedestrians and cyclists are policy-controlled」）。

## 3. 地图摄取（"任何 log 当地图"）

**管线（§3.4）**：各源 → 「a single common representation (inspired by ScenarioNet)」→ builder → `terrabin` 二进制。converter 与 source-specific 解析解耦，「supporting a new dataset requires only a new converter」。builder 职责：「identifies intersections, ties stop lines and traffic signals to their lanes, and resolves legal travel directions and speed limits」；「Intersections come from a source's own junction geometry where it provides one and are otherwise inferred from lane topology」；「The builder validates each scene and emits it as a compact binary terrabin file that the engine reads directly」。训练时「scenarios are streamed and sampled so that each batch stays balanced across regions and scenario types without holding the full dataset in memory」。
**terrabin 载荷**（Algorithm 1 第 1 行）：「Load terrabin scenario (map geometry, logged trajectories, signal phases)」——即几何 + log 轨迹 + **相位信息**三段。
> **未披露**：terrabin 二进制字段布局/精度/版本号；「任意 log」到 common representation 的具体 schema；是否含车道级速度限制字段的编码；除 Waymo/nuPlan/CARLA 三个 converter 外的通用 log 入口（页面口号 "Any log becomes a map" 在报告里**只有三源 converter**，无通用 log 解析器）。
> **口径修正（对 03 票）**：报告 §2.2 指出「WOMD, for instance, carries sparse traffic-signal annotations」，而 Limitations 要求「traffic signal phase information」——但 §4.5 的三种控制器是**程序化生成配时**，故「相位结构」是硬需求、**真实配时不是**。这与 03 票结论「配时无需真值（程序化覆写）」一致，报告可作为佐证。

**规模（§6.2）**：WOMD 约 576K scenarios（487K 训练 split）；nuPlan 约 262K（4 城，Las Vegas 最大、Boston 最小）；CARLA 仅 5 个合成小镇。

## 4. 仿真架构与吞吐口径

**机制（§3.1）**：「The object-level dynamics, observation construction, reward computation, and traffic-light logic run entirely in C, compiled into a CPython extension, while policy inference and learning run in PyTorch on the GPU」；全栈基于 PufferLib。零拷贝：「the C engine writes simulation state directly into the buffers that the training loop reads back as PyTorch tensors, eliminating the per-step serialization and allocation… host-to-device transfers overlap with kernel launches」；三条具体手段——缓冲区「sizes its buffers to the controlled agents actually present rather than padding to a fixed count」、观测「16-bit precision that the GPU reinterprets bit-for-bit」、NUMA 亲和（pins each rank's workers to the CPU cores local to its GPU's NUMA node）。分布式：data parallel + 梯度 all-reduce + **跨 rank 同步归一化统计**（advantage normalization / PopArt / joint priority sampling 用同一全局尺度）（§5.2）。
> **未披露**：是否真为 device-mapped 共享内存（cudaHostRegister / IPC handle / shared arena 布局）还是 pinned staging buffer 异步拷贝；batch 维布局与对齐约束。

**吞吐口径（§6.1 + Table 1）**：「Measured on a representative training job, TerraZero sustains 560K agent-steps per second on a single consumer GPU, 1.3M on a single server-grade GPU, and 2.8M on an 8-GPU server node」。Table 1 脚注：「Numbers marked † are figures reported by the original authors; **all other SPS values are benchmarked under our setup**」——即 PufferDrive 320K/745K、Waymax 319K、GPUDrive 11K/22K 是**作者代跑**的对照。
> **未披露（关键）**：① "agent-step" 是否只计策略可控 agent（还是含 NPC/背景车）——直接决定与 PufferDrive 320K 是否同口径；② 是否含 learner 更新与 host 传输时间；③ 复跑时的场景内容/agent 数/环境数配置；④ 8-GPU 节点 2.8M 相对单卡 1.3M 的次线性（8×1.3M=10.4M）未解释。
> 报告自己也留了口子：§6.3.1 说 TerraZero「stays usable where the underlying lane and road topology is imperfect or broken, the regime that richer feature stacks handle poorly」——即它用**更瘦的观测**换鲁棒性，不是换算力。

## 5. 零演示 recipe

**动作/观测（§5.1 + 附录 C）**：车辆离散 jerk（或 accel）动作空间；sim agent 为 jerk / unicycle grid / compact bicycle grid 三个 per-type head（**离散档位数未披露**）。观测四组 Deep Sets 编码、ego 坐标系（Eq.1 旋转公式）：ego 特征表（Table 9）逐维给缩放（rel_goal_pos ×0.005、vehicle_dims (w/15, l/30)、speed 带符号、acceleration (along/5, alat/4)、stop_sign_state 3 维 one-hot、alpha_params 19 维、kinematic_params 4 维）；partner 每体 8 维（Table 10）；road 每段 7 维（Table 11，road_type lane=0/line=1/edge=2，且「Road segments fill a fixed budget that prioritizes lane centerlines and road edges… so dense markings cannot crowd the essential geometry out」）；traffic entity 每体 12 维（Table 12，含 3D 位置 z×0.1、信号 one-hot (red,yellow,green,off)、stop line 端点）。上限（Table 7）：partner 20 @50m、road segment 200、traffic entity 16 @100m；Δt=0.1s、policy 步长 0.1s、episode 256 步（25.6s）。
**策略**：紧凑 MLP ≈3.5M 参数（sim agent 6.7M），per-group 编码器（ego MLP + road/partner/traffic 的置换不变 Deep Sets）拼接后过「a three-layer, 1024-unit MLP shared by the actor and value heads」。

**Reward 结构（附录 B，Table 8）**：23 维参数向量，per-type 权重（Vehicle/Pedestrian/Cyclist 三列），每项支持三种模式——固定标量 / [min,max] 每 agent 抽样 / null（剔除并从观测中剔除）。逐项与默认区间：Collision [0,3]/[0.5,2]/[1,3]（公式 `−(α_col+γ_col·v)`，γ_col 默认 0.1）；Boundary [0,3]/—/[0.5,2]；Comfort [0,0.1]（阈值 |a_long|>3.0、|a_lat|>3.0、|j_long|>5.0、|j_lat|>5.0，ped (1.5,1.0,12.0,5.0)、cyc (3.0,1.5,40.0,18.0)）；Lane Align [2.5e-4,2.5e-2]；Lane Center [2.5e-4,7.5e-3]；Velocity 固定 2.5e-3（`+α_vel·Δt·v/v_max`，v_max=20·c_vel）；Vel Align [0,1]；Reverse [2.5e-4,7.5e-3]；Timestep 固定 2.5e-5（survival）；Stop Line 固定 5.0；Red Light 固定 3.0；Road Incursion（pedestrian）[0.5,2]；Speed Limit（ped [0.5,2]、cyc [0.25,1]）；Edge Preference（cyc）[0.01,0.05]。辅助阈值：δ_goal 默认 2.0m（可随机 [2,12]）、v_goal 3.0m/s、b_center 0.0。抽样参数归一化 α̂=2(α−αmin)/(αmax−αmin)−1，固定项归一化为 0。
> **承重级发现**：Table 8 题注「**Goal progress and target speed are held at zero across both reported policies and are omitted**」——即 TerraZero 的 reward **没有沿路线的进度塑形项**，唯一路线激励是终点的 Goal bonus（固定 1.0）。后果在报告自己数据里显形：InterPlan 上「route-progress term sits at 67.1, **the lowest of the components**」（§6.3.2），val14 上 Ego Prog. 92.91 也低于 Gigaflow 93.6。**照抄这份 reward 会继承一个进度短板**（这是复刻侧应当主动补的一刀）。

**稳定性三件套（§5.1）**：① saliency-prioritized sampling：`p_i ∝ (Σ_t|Â_{i,t}|)^α`，IS 权重 `w_i=(N·p_i)^{−β}`，β 线性退火；② V-trace off-policy 校正（「Setting both clips to infinity recovers standard GAE」；截断保留 `V(s_final)` 并切断 λ trace，真终止零 bootstrap）；③ PopArt 价值归一化（EMA 更新 μ、σ，解析重缩放 value head；target 在归一化空间、advantage 在真实回报尺度）。另有 goal dropout（默认 0.3 车辆）、观测噪声模型（默认关：截断高斯扰动 + slot dropout）。
**PPO/配置（Table 14/15）**：lr 5e-4 线性衰减、γ 0.99、λ 0.95、ε 0.2、value 不裁剪、entropy 0.01、vf coef 0.5、grad-norm 0.5、每 rollout 2 epoch、V-trace 开启（ρ̄=c̄=1.0）、PopArt EMA 0.9997 / min σ 1e-4、优先级 α=0.85、β0=0.85→≈0.978、Adam(0.9,0.999,1e-8)、16 minibatch、每 GPU 512 agents / 32 envs / vectorized batch 4；planner 16×A100-80GB bf16、sim agent 32×A100。
**规模**：**报告未给出总训练步数**（见 §8 修正）。
**动力学随机（§5.1 + Table 13）**：四系数 c_throttle/c_steer/c_acc/c_vel 每 agent 在 [0.5,1.5] 抽样、episode 内恒定，且「Velocity and acceleration bounds… are applied **after** the dynamics integration step」；sim agent 的行人固定、骑行者区间更窄（Table 15）。

## 6. 评测协议

**val14（§6.3.1）**：经 **PlanTF evaluation suite** 跑 nuPlan closed-loop reactive score；14 类场景按均衡比例抽取；动作经「the benchmark's standard nuPlan Linear-Quadratic Regulator (LQR) controller」执行，「so no controller-side change contributes to either score」。
**InterPlan（§6.3.2）**：八个交互族（施工、事故现场、jaywalker、绕行停驻车、对向超车、低/中/高密度变道）；报官方 80-scenario split（70.87）以便同场景比较，另报 335 全量（71.31）；同一 checkpoint、同一训练配置。
**WOSAC（§6.4）**：2023 版=Waymo 全 validation split 去掉超过 128-agent 上限者（与 Gigaflow 同口径）；2024 版=「the filtered validation subset of CAT-K and SPACeR, built from the released CAT-K validation manifest under the same 128-agent cap」（项目页 Figure 5 给出规模为「shared 880-scenario subset」，**报告未写 880**）；协议：9.1s@10Hz、前 1.1s 为初始上下文、后 8s 评分、≤128 agents、32 次随机闭环联合 rollout；realism = 0.20·kinematic + 0.45·interactive + 0.35·map，VRU 去掉 vehicle-only TTC 后在 bucket 内重归一化；车辆/VRU 分表；checkpoint 为 Waymo epoch 12,000 与 nuPlan epoch 4,000。
**迁移研究（§6.5）**：WOSAC-2024 vehicle realism（80 帧未来、32 rollout；「Agents are initialized from the logged state but pursue randomly generated goals」）+ NAVSIM v2 EPDMS core safety（取 at-fault-filtered no-collision / drivable-area / driving-direction / TTC 四项的最小值）；每格约 100 场景；ego 受控、其余 agent 走 log 回放（故非全反应式）。
**评测期条件化（§6.3 末段，报告最有复用价值的一句）**：「At evaluation TerraZero conditions the policy for each benchmark through the reward- and kinematic-conditioned observations of Section 5, **with no retraining**, and aligns the eval-time goal and route sampling with the lane-level goals the policy saw during training (Section 4.2); the benchmarks' own scoring stays unchanged.」即**同一 checkpoint 靠推理期改 reward 权重向量即可切换 val14/InterPlan 行为风格**。
> **未披露**：评测期具体 reward 权重向量取值；jerk 动作到 nuPlan LQR 跟踪器的接口映射（训练动作空间与评测执行器不同源）；val14 14 类场景的清单与均衡比例细节。

## 7. 承重墙判断（对照 11-03 文档五成分）

**报告自己给出的因果证据（可直接引用）**：
- §6.3.2：「the reported policy trains against a dense, procedurally generated population of construction cones, static obstacles, crashed-vehicle clusters, crossing and jaywalking pedestrians, and reactive IDM traffic (Section 4.4), **so the long-tail families InterPlan probes resemble scenes the policy already practices**」；并称「which points to **procedural long-tail scenario generation rather than model scale** as the source of the gain」。
- §2.1 对 Gigaflow 的批评：「its scenes hold only policy-driven agents and static obstacles modeled as untyped immobile vehicles, leaving out… construction zones, crashed-vehicle clusters, curb-parked cars, and jaywalking pedestrians」。
- §5.1 承重前置条件：「the recipe's stability… **the right currency is compute efficiency rather than sample efficiency**」，前提是环境信号足够丰富、规则不可钻空子。

**承重墙（缺了必掉分）**：
1. **长尾 hazard 生成层（成分 2）**——InterPlan 70.87 vs SPDM(60) 63.66 的直接读数；报告的八生成器清单 + Figure 4 车道占用几何，是 07 票「hazard 最小原型」的可执行规格。
2. **真实规则的几何执法（成分 4）**——§3.3 全是几何判定而非脚本旗标：OBB 分离轴；off-road 用 footprint 对 road-edge/sidewalk 边界并加「an elevation gate so that overpasses are ignored」；lane direction 用 heading residual；红灯「registers when the agent's **swept path crosses the stop bar on a red tick**, not merely for waiting within the region」；停牌「hold below a speed threshold for a sampled dwell time」；行人豁免路口规则、骑行者按车辆处理。**这类几何执法在 C 内核里是低成本高回报项**。
3. **每 episode 的 reward/动力学随机化 + reward-conditioned 观测（成分 5 的真身）**——它同时做到「防过拟合单一目标」与「推理期可调风格」，是报告里唯一被复用两次（训练 + 评测）的机制。
4. **多类动力学（成分 3）**——仅对 sim agent 角色承重（WOSAC 0.632 vs Gigaflow 0.619）；对 ego planner 角色不承重。

**锦上添花 / 已被高估**：
- **吞吐与零拷贝**：是工程形态不是能力成分；且报告未披露 SPS 计数口径，与 PufferDrive 320K 的差距目前**不可严格对账**。
- **"3D 地图"**：报告里的 3D = z 感知几何（Table 7 的 Z-separation 2.5m 多层过滤、traffic entity 的 3D 位置、elevation gate），**不是渲染**。11-03 文档 §3 已判「E. 3D 已达标」，报告口径支持该判断——这一项被列为承重墙是权重错配。
- **"高保真卡车"**：= 自行车模型 + 轮胎侧偏力，参数未披露；相对 Gigaflow「truck as size variant」的增量是一层动力学。
- **PopArt / V-trace / 优先采样**：都是标准件，属稳定性与样本效率，对「能不能到 SOTA」非承重。
- **紧凑 3.5M MLP + 瘦观测**：反而是卖点（对烂地图拓扑更鲁棒）。
- **goal dropout / 观测噪声**：正则项，噪声模型默认关闭。

**复刻侧必须主动补的两处（报告自曝短板）**：① reward 缺路线进度项 → 补 progress shaping；② 评测期 reward 权重向量未公开 → 复刻无法逐位复现 val14/InterPlan 成绩，只能复现「同一 checkpoint 双榜」的机制。

## 8. 横向视角：报告 / 项目页 / 本仓文档的三处口径修正（**报告 vs 项目页 vs 11-03**）

1. **8×10¹¹ 步不在报告里。** 报告全文无总训练步数（§6.2 只有 16×A100/32×A100 与 WOSAC 的 epoch 12,000 / 4,000）。8×10¹¹ 的出处是系列页：`TerraTransfer.txt:86` 「≈2,535 yrs of self-play driving experience (8×10¹¹ steps)」、`:147` 「TerraZero · massively-parallel self-play 8×10¹¹ steps」、`:181` 「8×10¹¹ simulated steps」；`TerraTransferPlusPlus.txt:283` 「8.0 × 10¹¹ steps in simulation / 224M miles / 2,535 yrs」（该页另有 **801B** 自有 self-play 步数）。→ **11-03 文档 §1 把 8×10¹¹ 归给「驾驶能力全部来自向量化 self-play」时，应标注其来源是 TerraTransfer/++ 页而非 TerraZero 报告**；票 02 第 5 项的「8×10¹¹ 步」在**本报告口径下为「未披露」**。
2. **「统一策略控异质 in preparation」只对车+卡成立。** 报告 §3.2/§5.1/§6.4 已发布并评测**车+行人+骑行者**统一策略（per-type action heads，6.7M 参数，WOSAC 2023 realism 0.632 > Gigaflow 0.619）；项目页脚注仍写 in preparation。→ 08 票若按页面读会低估异质统控的成熟度；但**车+卡车**统一策略（demo 下拉 "heterogeneous"）确实不在报告内，且卡车侧仅「bicycle + tire cornering forces」。
3. **「场景编辑器」在报告和页面上都不存在。** 两侧只有「配置轴 + 每 episode 采样」；demo 的「drop in road users and hazards」实为生成器上的筛选器（"the selectors narrow to whatever it has on hand"）。→ 07 票的原型方向应定为「hazard 生成器 + 混合配置」，不是交互式编排器。

---

## 摘要（≤30 行）

1. **场景生成器**：六配置轴（§3.2）；road-user 轴恰有**八个生成器**（反应式车/parked/碰撞车簇/施工区/静态障碍/斑马线行人/jaywalker/log 回放骑行者）；goal=lane 图前向游走+对齐过滤（弧长界与阈值**报告明说不给**）；三种信号控制器（NEMA 八相位 / Christmas=每 stop line 独立 红→绿→黄 + LogNormal dwell / round-robin）+ 逐路口初始相位随机 + 保护左转翻 permissive。
2. **异质建模**：行人 unicycle、cyclist/car bicycle、truck=bicycle+轮胎侧偏力；car 用 jerk 动作、其余用 accel；报告已发布**车+行人+骑行者统一策略**（per-type heads，6.7M），项目页脚注滞后；骑行者**无脚本 NPC**。
3. **地图摄取**：三源 converter → ScenarioNet 式共同表示 → builder（识别路口、绑 stop line/signal 到 lane、解合法方向与限速）→ `terrabin`（几何+log 轨迹+相位）；terrabin 布局、通用 log 入口**未披露**；配时由程序化控制器生成，真实配时非硬需求。
4. **仿真架构**：C 内核（编译为 CPython 扩展）跑动力学/观测/reward/信号逻辑，PyTorch 跑推理与学习，共享 numpy 缓冲零拷贝、16-bit 观测、NUMA 亲和、跨 rank 同步统计；SPS 口径「representative training job」，但**agent-step 是否含 NPC、是否含 learner 时间、复跑配置均未披露**。
5. **零演示 recipe**：reward 23 维三模式（固定/区间/null）+ per-type 权重 + reward-conditioned 观测；PPO+GAE+V-trace+PopArt+优先采样，Table 14/15 全表可抄；**总步数未披露**（8×10¹¹ 出自 TerraTransfer/++ 页）。
6. **评测协议**：val14 经 PlanTF suite + nuPlan LQR 执行动作；InterPlan 官方 80 + 全量 335；WOSAC 2023 去 >128 agent、2024 用 CAT-K manifest 过滤（880 见页面）；**同一 checkpoint 靠推理期 reward 权重切换基准，不重训**——但评测期权重值未公开。
7. **承重墙**：hazard 生成层 + 几何化规则执法 + 每 episode reward/动力学随机化与条件化观测；锦上添花/被高估：吞吐零拷贝（工程）、"3D"（仅 z 感知几何）、"高保真卡车"、PopArt/V-trace。

**三个「报告有而项目页没有」的关键发现**：
① **评测期条件化机制**：同一 checkpoint 免重训即可用 reward 权重向量切换 val14/InterPlan 行为（§6.3），且 reward 参数与运动学系数都在 ego 观测里——页面完全没有这条。
② **信号层全规格**：NEMA/Christmas(LogNormal dwell)/round-robin 三控制器 + 逐路口相位随机 + 保护左转翻转（§4.5），页面只有「varied traffic-signal timing」一句。
③ **reward 全表与「无进度项」**：Table 8 逐项区间/per-type 权重/23 维三模式 + 目标区 δ_goal[2,12]、γ_col=0.1、v_max=20·c_vel 全公开；同时题注写明 goal progress 与 target speed **置零剔除**——这解释了 InterPlan 进度项 67.1 为最低分，也给出复刻侧必须补的一刀。
