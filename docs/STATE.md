# STATE

As-of: 91e7bbf（夜间战役轮 6 收官 + 蒸馏提交前）
（2026-11-12 晨：ADR-0007 保守吸引子破解，配方 v3 定案候选，R@2× 双面纪录 1.333/1.052）
约定：AGENTS.md 是放置规则权威；strata 分层自 2026-11-03 启用；TerraZero 复刻 effort 活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0007`**（基底/配方v1/执法/运行时独立/奖励区间旋钮谱系/保守吸引子/熵破解配方v3）。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；`upstream` 只读；**lab42 双机线**：`zoo@192.168.202.42:/home/zoo/rino_we/808-Scale-AD/Slipstream`（2×RTX4090，同步走 `git push lab42` + `scripts/sync_lab42.sh`；父目录有他人项目勿动）。
- **配方 v3（ADR-0007，候选定案）**：B 基座 + `ent_coef 0.05` + `offroad[2,3]` + 4.87e8 步 → eval 双面 **1.333/1.052**（基线 B 0.170/0.059 的 7.8×/18×）、碰撞 **0.0007/0.0047**（反优于 1× 15×）、速度 2.28/1.77、dnf 0.14/0.25。剂量平台 ×4-×6；velocity 方向证伪；双钉价过度约束。冠军权重在 lab42 `artifacts/t3-R2x/ckpt/R_ent5_off23/models/`。
- 夜间战役 6 轮 × 3 GPU（3060 + 2×4090）14 个全量训练：证据在 lab42 `artifacts/t3-*`（t3-ent2 本地）；复现：R@1× 三种子 0.86-1.28 安全带紧密；R@2×@seed43 + S@seed43 在跑（09:40 收）。
- 内核旋钮谱系：T1 progress → T6 offroad → T7 collision（默认位级=legacy；本树 sha `02786a05…`，lab42 自建 sha 不同——跨机相对比较成立，锚点重定仍待 owner）。
- 评测资产：`t3_eval_ckpt`（carla_fast 双面）；v2 弧线证据（t2-*）在案（ADR-0006）。

## Docs 台账

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（冻结）；`.wayfinder/research/` 四份。
- 本会话 docs/ 根无新增日期前缀文件；夜间战役过程叙事在 `.scratch/night-2026-11-11.md`（易碎层）。

## Next（下一步）

- 收 R@2×@seed43 / S@seed43 复现 → v3 定案（或再迭代）。
- 「顶格率」指标（速度/车道限速比；限速基建在位）——owner 验收口径（物流/载人、贴限速）。
- v3 外部基准验证（InterPlan 80+335 / val14 cosim）→ 朝「SOTA」主张推进。
- 内核锚点重定（T6/T7）+ K4a 静态锚复跑；push（owner 自推，本地领先 60+ commits；lab42 有完整副本）。
- 雾区序不变：异质 sim-agent；外部验收；log→map；E1a/E1b。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②可部署、真实运营 log SOTA 的 planner。**验收口径（owner）**：安全+效率并重（物流/载人），速度贴车道限速顶格，瘫痪型保守不合格。
- 方法论铁律：全量+eval 双面判读（探针读瞬态）；内核钉死+热路径零 malloc；上游只读。
