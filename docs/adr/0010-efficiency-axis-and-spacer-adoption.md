# ADR-0010：效率轴任务定义改造 + SPACeR 锚定采纳（owner 双裁决）

Status: accepted-2026-11-17（效率轴口径）/ accepted-2026-11-17（SPACeR 采纳，更新票 06 旧口径）/ **accepted-2026-11-18：E3@4× 封 v5 基座（owner 一字裁决「封」；同场授权：此后技术决策全部由 agent 三门自主闭环，不再请示）**。一手来源：owner 会话原话「封。但是要强调，技术类的决策，以 three-gate 策略实现自动设计、决策、review、实现和独立三方验证」。

- **效率口径**：实测证明 goal_speed=3.0（GIGAFLOW v_goal=3 硬钉）使「贴限速」结构不可能（16× 冠军巡航 2.8 m/s vs 8 城限速 8.33-29.06，attainment=0.0 @N=64 三门验证）。裁决=v5：goal_speed_mode=lane_limit（内核单点门控写，`c8ad22f`）。
- **E 系首验裁决**：E3（progress 回 v1 剂量联动）> E1（纯 mode）> E2（压 goal）；E3@4× = v5 最强（速度 4.71 m/s +88%、碰撞 0.0018、ret 5.32）→ **已封 v5 基座（2026-11-18）**：S 系 α 校准、E 系剂量迭代、lane_limit 长跑均从 E3 配方出发；lane_limit 峰位精定位由剂量迭代免费携带。
- **SPACeR 采纳**：β·KL(π_θ‖π_ref) + α·log π_ref(a|s) 已实现（`4b46cdf`，校准 mean 1.944 nats，S2 α 回填 0.0064）；与 snapshot-pool 伙伴混合（v4）正交——命名去 SPACeR 化的旧理由失效，两旋钮并存：伙伴混合管对手分布，KL 锚定管 learner 漂移。
- 跨任务可比性铁律：lane_limit 与 constant 的 return 不可比；可比量=速度/碰撞/dnf/顶格率（物理口径）。
