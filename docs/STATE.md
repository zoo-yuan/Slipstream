# STATE

As-of: 2896d3e（B 全量收官 + t2_bfinal）
（2026-11-11：v2 配方弧线闭合——五臂矩阵 + 三次全量，保守吸引子定症，ADR-0006）
约定：AGENTS.md 是放置规则权威；strata 分层自 2026-11-03 启用；TerraZero 复刻 effort 活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0006`**（基底/配方 v1/执法/运行时独立/奖励区间旋钮谱系/保守吸引子与 v2 终局）。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；`upstream` 只读。内核旋钮谱系：T1 progress → T6 offroad → T7 collision（区间式，默认位级=legacy RNG 流；本树 binding `02786a05…`，**t2-maintrain 锚仍 `9615216c…`**，训练跑 `--allow-kernel-drift` 显式放行，锚点重定待 owner）。
- **v2 弧线已闭合（ADR-0006）**：五臂探针矩阵 A(v1)/B(proghalf)/C(+offroad[3,3])/D(区间[1.5,3])/E(+collision[3,3]) + 三次全量（T3-v1 2.2e8、C 2e8、B 2.43e8）。**安全侧交付**：C +0.29/+0.10、B +0.17/+0.06 双面正收益，出路≈0，碰撞 0.009-0.029。**效率侧结构性不可达**：全臂收敛保守盆地（dnf 0.66-1.0、速度 ≤0.55、progress→0）；病灶=熵衰减；E 全瘫证明「eval 权重钉进训练」方向错误。
- 评测资产齐备：`t3_eval_ckpt`（carla_fast 双面 ckpt A/B）+ `t2_v2sweep`（多臂+每臂自动 eval）+ `t2_v2final`/`t2_bfinal`（单臂全量）。证据在 `artifacts/`（t2-haztrain / t3-eval{,-mix,-t2c} / t2-v2sweep / t2-v2final / t2-bfinal，各含 index+summary 带 kernel 溯源）。
- 兄弟仓 `../PufferDrive`：Nova bridge 线 + strata 语境，只读（ADR-0004）。
- 外部评测资产：`external/interplan`（80+335 场景）；val14 cosim 栈选型在案。

## Docs 台账（冻结检查依据）

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` — 冻结分析。`.wayfinder/research/` — 四份调研。
- （2026-11-11 会话 docs/ 根无新增日期前缀文件）

## Next（下一步）

- **v3 方向待 owner 开新图裁决**（ADR-0006 证据链指向）：熵下限/退火探针（首选）、reward_velocity 剂量对齐（训练 0.05 vs eval 0.0025 的 20× 错配）、hazard 密度课程、lr 退火。验收口径已升级：**安全+效率双维（物流/载人，贴车道限速顶格，拒绝瘫痪型保守）**；限速基建在位（车道级限速+随机化+compliance 指标），「顶格率」专用指标待建。
- 内核锚点重定（T6/T7 后）+ K4a 静态锚复跑留证——v3 定案时一并。
- **push 未落地**：远端落后本地 45+ commits，owner 自推。
- 雾区候切（执行序）：异质 sim-agent；外部验收（nuPlan→val14→InterPlan）；log→map（Apollo G1-G3）；E1a/E1b。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②在 TerraZero 类环境训练出**实际可部署、真实运营 log 上 SOTA** 的 planner。**验收口径（2026-11-11 owner 补充）：物流/载人场景，安全+效率并重，速度贴车道限速顶格运行，瘫痪型保守策略不合格。**
- 与兄弟仓集成：rino 数据湖 + 混合智能体闭环，Nova 19入/4出契约；复刻 = 场景多样性与 self-play 供给线。
- 方法论铁律：配方裁决 = 全量 + eval 双面（探针只筛瞬态，ADR-0005/0006）；上游只读；内核钉死 + 热路径零 malloc + 信任边界校验。
