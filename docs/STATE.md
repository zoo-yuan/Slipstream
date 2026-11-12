# STATE

As-of: ab5e531（白昼战役 II-III：双冠军 2.113/2.100 N=2，规模未饱和）
（2026-11-12 晚：ADR-0008 步数=硬通货；量级改进提案待评审；两卡已让出，断点在盘）
约定：AGENTS.md 是放置规则权威；strata 分层自 2026-11-03 启用；TerraZero 复刻 effort 活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0007`**（基底/配方v1/执法/运行时独立/奖励区间旋钮谱系/保守吸引子/熵破解配方v3）。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；`upstream` 只读；**lab42 双机线**：`zoo@192.168.202.42:/home/zoo/rino_we/808-Scale-AD/Slipstream`（2×RTX4090，同步走 `git push lab42` + `scripts/sync_lab42.sh`；父目录有他人项目勿动）。
- **配方 v3 双冠军 N=2（ADR-0008）**：R@4× s42（8×128）**2.113/1.825**、s43（16×256）**2.100/1.909**；碰撞 0.0004-0.0015、dnf 0.011-0.021。规模曲线未饱和（1.20→1.33→2.11）；**总步数=硬通货，面轴=墙钟优化（+20%）**，双卡并跑另有 SPS 争用（87K→45K）。8×（1.95e9）断点在盘等卡（artifacts/t3-f16-8x，323M/1946M）。冠军权重本地+lab42 双份（含 config.yaml）。
- 渲染验收在位：`artifacts/NIGHT-REPORT.html` v3 + 冠军回放画廊 render-CHAMP{,2}（同 8 场景 vs B 基线）。
- **GPU 现状（11-12 晚）**：lab42 两卡均已让出（1 号→strata 服务，0 号→高优任务），本地 3060 已让；全部战役进程消零。
- 夜间战役 6 轮 × 3 GPU：证据在 lab42 `artifacts/t3-*` + 本地镜像；过程 `.scratch/night-2026-11-11.md`；复现（R@1× 三种子 0.86-1.28 安全带紧密；R@2× N=2；S@s43 紧密）。
- 内核旋钮谱系：T1 progress → T6 offroad → T7 collision（默认位级=legacy；本树 sha `02786a05…`，lab42 自建 sha 不同——跨机相对比较成立，锚点重定仍待 owner）。
- 评测资产：`t3_eval_ckpt`（carla_fast 双面）；v2 弧线证据（t2-*）在案（ADR-0006）。

## Docs 台账

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（冻结）；`.wayfinder/research/` 四份。
- `docs/2026-11-12-量级改进路径提案.md`（冻结提案，待 owner 评审；决策点 5 项）。
- 夜间战役过程叙事在 `.scratch/night-2026-11-11.md`（易碎层）。

## Next（下一步）

- **待 owner 评审**：`docs/2026-11-12-量级改进路径提案.md`（四路径 + 5 决策点；建议序：①外部 SOTA 标定→③断点续跑→②自博弈 v4→③容量/16×→④数据轴）。
- 8× 断点续跑命令在案（卡空即接，剩 ~2.7h）；V/W 剂量臂断点在盘可补。
- 「顶格率」指标待建；v3 外部基准验证（InterPlan/val14）未做。
- push（owner 自推，本地领先 65+ commits；lab42 完整副本）。
- 雾区序不变：异质 sim-agent（与提案②自博弈合流）；外部验收；log→map；E1a/E1b。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②可部署、真实运营 log SOTA 的 planner。**验收口径（owner）**：安全+效率并重（物流/载人），速度贴车道限速顶格，瘫痪型保守不合格。
- 方法论铁律：全量+eval 双面判读（探针读瞬态）；内核钉死+热路径零 malloc；上游只读。
