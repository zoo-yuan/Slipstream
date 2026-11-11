# STATE

As-of: 9502e95（内核 T6 offroad 旋钮 + v2 扫 C 臂）
（2026-11-10 晚：T3 hazard-on 长训完成 + v2 三臂前沿 + 内核旋钮谱系开枝；11/11 票全闭维持）
约定：AGENTS.md 是放置规则权威；strata 分层（STATE / adr / lessons）自 2026-11-03 启用；TerraZero 复刻 effort 的活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0005`**（基底/配方/执法/运行时独立/奖励区间旋钮谱系）。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；tag `baseline/pufferdrive-3.0-2c1ca3e4`。remotes：`origin` = zoo-yuan/Slipstream；`upstream` 只读。
- **内核 sha 现状（T6 后）**：本树 binding = `11c130c6…`（T6 offroad 区间旋钮，默认位级=RNG 流保持）；**t2-maintrain 锚点仍为旧 sha `9615216c…`**，训练跑以 `--allow-kernel-drift` 显式放行；锚点重定待 owner（ADR-0005）。
- **T3 hazard-on 长训完成**（`artifacts/t2-haztrain/`）：单臂 C=配方 v1 + mix 面，2.205e8 步/1682 epochs/1.8h 零失败，face 8×128 独占 GPU + expandable_segments 全程稳。
- **eval 基建**：`scripts/t3_eval_ckpt.py`（carla_fast 双面 ckpt A/B 评测）。T3 结论：早停否决（e168 准瘫痪）；T2C/T3 = 互补失败模式（T2C 极保守 dnf 0.72 / T3 机动但借出路 return 崩）。
- **v2 三臂探针完成**（`artifacts/t2-v2sweep/`，含每臂自动 eval）：A_v1ctl / B_proghalf / C_prog_off3（=B+offroad 钉[3,3]）构成安全↔机动 Pareto 前沿：B 机动王（goals 1.43，碰撞 0.21 偏高）；**C 首个 eval 正收益（+0.24 clean/+0.10 mix）、出路≈0、碰撞最低**，代价 dnf 0.54 趴窝。无单一赢家，全量三选一待 owner（见 Next）。
- 训练/收割资产：`t2_sweep`（index 自动重建，手工页退役为 index.manual.html 备份）/ `t2_hazsmoke`（点火前 smoke 门）/ `t2_haztrain`（单臂长训）/ `t2_v2sweep`（多臂+每臂 eval，臂覆盖机制泛化到任意 env 键）。
- 兄弟仓 `../PufferDrive`：Nova bridge 线 + strata 文档语境，只读引用（ADR-0004）。
- 评测资产：`external/interplan`（80+335 场景）；val14 官方闭环栈选型在案。

## Docs 台账（冻结检查依据）

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` — 冻结分析（自兄弟仓拷入）。
- `.wayfinder/research/` — 四份调研产物，随票 Resolution 索引。
- （本会话 docs/ 根无新增日期前缀文件）

## Next（下一步）

- **待 owner 点火（明早第一问）：v2 全量 2.2e8 三选一**——① 先探膝点臂 D（proghalf + offroad [1.5,3]，~25min）再全量最优；② 直接全量 C（正收益最安全，赌 dnf 随步数缓解）；③ 直接全量 B（能力线 goals 优先）。引擎 `t2_v2sweep` 直接复用（done 臂自动跳过），需 `--allow-kernel-drift`。
- 候选后续：collision 区间旋钮（ADR-0005 谱系下一员，C 臂「宁撞不出路」证据）；v2 定案后内核锚点重定 + K4a 静态锚复跑留证。
- **push 未落地**：远端落后本地 35+ commits，owner 自推（好上行窗口）。
- 雾区候切（wayfinder 地图已完成，执行序）：异质 sim-agent 线；外部验收（nuPlan→val14→InterPlan）；log→map（Apollo G1-G3）；E1a/E1b（Nova 权重解锁）。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②在 TerraZero 类环境上训练出实际可部署、真实运营 log 上 SOTA 的能力（RL 确定性路径 + 混合手段增概率）。
- 与兄弟仓集成：rino 数据湖 + 混合智能体闭环，Nova 19入/4出契约内训练 planner；TerraZero 复刻 = 场景多样性与 self-play 供给线。
- 上游纪律：`upstream` 只读；`9_bernhard_dev` 挖矿不并基线；AGPL 只读。内核钉死 + 热路径零 malloc + 信任边界校验（ADR-0005 延续）。
