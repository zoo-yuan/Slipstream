# ADR-0005: 奖励区间旋钮谱系与内核重锚纪律

Status: 2026-11-10 · 一手出处：T3 退化诊断 → owner 批准（2026-11-10 晚会话）

- 训练 regime `reward_randomization: true` 下，静态奖励键（如 `reward_offroad`）**惰性**——内核 `generate_reward_coefs`（drive.h）按 agent 从 REWARD_BOUNDS 采样，静态键只在 randomization=false 分支被读（v2 扫臂 B≡C 位级全同为证）。要按臂操纵系数，走 T1 progress 开创的**区间旋钮特例**：env 传 `[lo,hi]`，采样 draw 槽位与无条件性不变——默认区间 = 旧行为，RNG 流位级一致（vs 旧 .so 逐位验证通过，3 seeds × 16 agents × 19 系数列）。
- 谱系：T1 `reward_progress_lo/hi`（d5ce8ad）→ T6 `reward_offroad_lo/hi`（9502e95，binding sha `9615216c…`→`11c130c6…`）。下一个天然候选 = **collision 区间**（v2 C 臂「宁撞不出路」证据：offroad 钉 3.0 后碰撞成了相对便宜的出口，训练面碰撞 0.345 高企）。
- 落地纪律：init 前 fail-fast 校验（ValueError，Python/C 双侧，对齐全仓 knob 惯例）；热路径零 malloc（区间只在 init 读入 state）；默认位级不变 ⇒ 跨代内核的训练结果可互比。
- 重锚：动内核必变 binding sha。t2-maintrain 锚点仍是旧 sha；后续训练以 `--allow-kernel-drift` 显式放行（owner 明示口径）。**锚点何时重定为新 sha 待 owner 裁决**（建议：v2 全量定案后一并重锚 + K4a 静态锚复跑确认——RNG 位级保持下静态锚数值理应不变，但按锚点纪律需复跑留证）。
- 关联方法论裁决（同会话）：**配方判读以 eval 双面（carla_fast clean+mix）为准，不以训练曲线为准**——T3「末段退化」在 eval 语义下被否决（e168 准瘫痪 vs e1682 全面占优）；训练 return 含 shaping 视角偏差。
