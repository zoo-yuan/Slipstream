# FAQ · 架构与数据口径问答（活文档）

> 性质：持续补充/更新/修正的活文档（非冻结叙事）。每条标注 as-of（代码/数据状态锚点）。
> 更新规约：新问答追加；认知修正时**改答案并在尾部 changelog 记录旧口径**；来源会话/实验落 ref。
> 登记：docs/STATE.md 台账。

## Q1 · 训练数据来自哪里？engine 内置地图吗？ [as-of 2026-11-18]

**是**。训练全程用 CARLA 8 城 map bin（`pufferlib/resources/drive/binaries/carla/`，随仓内置，由 CARLA OpenDRIVE 经 `data_utils/mirror_map_bin.py` 转换）；bin 含 lane graph + 逐 lane 限速 + 边界，训练 env 直接加载。
- nuPlan 4 城整城 bin（us-nv-las-vegas-strip 等）= 2026-11-18 自建（`scripts/val14_build_city_bins.py`），**仅评测未训练**。
- Apollo 分块 base_map（7235 临沂等）= 第三地图域，G1 转换器目标（见 Q4）。

## Q2 · 有 behavior cloning / 模仿学习吗？GT 是什么？ [as-of 2026-11-18]

**没有，纯 RL 自博弈**（PuffZero/TerraZero 范式：零专家数据，规模换智能——TerraZero val14 94.19 未用一滴专家数据）。
- 训练信号全部来自奖励（progress/goal/offroad/collision/comfort/ent…），obs 821 维中**无专家动作通道**。
- 内核 `Agent.log_trajectory` 结构仅用于 cosim 回放评测；val14 指标 `ego_progress_along_expert_route` 中的 expert route 是 **metric 分母**（进度基准），不是训练监督。
- 相关裁决：ADR-0010（任务定义对齐 > 模仿信号——E3 lane_limit 零样本 +31% progress 实证）。

## Q3 · Apollo normal_cases 该作 IL 训练 GT 还是测试数据？ [as-of 2026-11-18]

**裁决：测试先行（强必要），IL 缓行（留升级路径）**。owner 2026-11-18 认同放手用 matched 袋。
- 测试 = ② 北极星验收主链路：G1 转换器 → 实运场景零样本评测（效率/安全/顶格率三方读数）。
- IL 缓行三理由：①分布 gap 主因是任务不是模仿（Q2 裁决实证）；②GT 质量（安全员日常驾驶非专家级、20s 短段）+ 映射有损（Apollo ControlCommand 油门/刹车/转向 → 12 类 jerk 离散动作）；③若 G1 评测 gap 显著且 RL 难闭合 → IL 那时有明确靶子（BC/蒸馏/辅助损失按 gap 维度选）。

## Q4 · Apollo 数据资产口径（bag / map_match / 地图） [as-of 2026-11-18 17:27]

- bag：`normal_cases/` 2726→2971+（持续增长，下载清单 3806）；Cyber Record 格式（非 ROS bag）。
- 四通道 presence（三门验证）：四有率 **99.30%** [97.95, 99.76]（424/427 抽样，分层 99.38%）；变体通道 `/apollo/perception/safetyobstacles` 待 G1 裁决是否并入 perception。
- map_match（`map_match/*.tsv`，bbox+150m 口径非车道级）：**matched 285**（唯一命中）/ ambiguous 268（待车道级确认）/ no_map 2140（31 图不覆盖，owner 裁决不纠结）/ pose_err 13。matched 最大群 = 7235_huolala临沂 124 袋。
- 地图：`apollo_map/data/maps/7235_huolalalinyi-1588/`（1875 分块 base_map.bin，15G，rx-2996 tar 解压）在位；其余 30 张 bbox 用图位置待确认（不阻塞 G1 微链路）。
- G1 微链路起步材料：124 袋（7235 群）+ 7235 图。

## Q5 · val14 外部标定现状 [as-of 2026-11-18]

- 链路全通（P0/P1a/P2 三门闭环）；16× 冠军 32 vegas 场景全计分；E3 对照：progress +31%/comfort +12.5%/direction 满分/碰撞更好（drivable −3%）。
- 数据缺口：本地 val 镜像 1191/1381 log（缺 190 + 11 坏）→ token 解析率 48.6%（543/1118）；ModelScope 无 val、HF 无镜像、AWS 桶 403 需注册——**源头待 owner**（缺单 `artifacts/val14/missing_chunks.json`）。
- 已知口径注意：48.6% 子样城市分布有偏，子样分只能标 partial；聚合器 KeyError（carl pin devkit 1.2.2 vs editable 1.2.0）由 collect 脚本绕行。

## changelog
- 2026-11-18 首版（Q1-Q5，来源：owner 问答会话 + val14/bag 两战役产物）。
