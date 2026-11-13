# STATE

As-of: 261fd86（夜间规模战役 III：16× 新冠军 2.978/2.792 @400 场景双面，32× 回落定位峰位）
（2026-11-13 晨：owner 授权 GPU 窗口收尾；cap2048 OOM 待独占卡；val14 数据三缺口在案）
约定：AGENTS.md 是放置规则权威；strata 分层自 2026-11-03 启用；TerraZero 复刻 effort 活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0007`**（基底/配方v1/执法/运行时独立/奖励区间旋钮谱系/保守吸引子/熵破解配方v3）。

## Where（现状）

- **规模轴 III 完成（2026-11-12/13 夜，owner 授权窗口）**：R 配方 s42 @16x256 face，**16× = 新双面冠军 ret 2.978/2.792（400 场景大样本，coll 0.0000/0.0023，dnf 0.0005/0.0182，goals 2.89，速度 2.83）**；规模曲线 1.20→1.33→2.11→2.42→**2.98（16×）**→**2.49（32× 回落）**——**步数硬通货汇率在 16× 拐头**，32× 后期训练 ret 波动提示超长程 lr 不退火不稳。权重：lab42 `artifacts/t3-f16-16x/`（16×s43 复现断点在盘 76% 停）；32× `artifacts/t3-f16-32x/`；8× `artifacts/t3-f16-8x/`。交接页 `.scratch/handoff/2026-11-13-night-scale-III.md`。
- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；`upstream` 只读；**lab42 双机线**：`zoo@192.168.202.42:/home/zoo/rino_we/808-Scale-AD/Slipstream`（2×RTX4090，同步走 `git push lab42` + `scripts/sync_lab42.sh`；父目录有他人项目勿动；192 核）。
- **配方 v3 双冠军 N=2（ADR-0008）**：R@4× s42（8×128）**2.113/1.825**、s43（16×256）**2.100/1.909**；碰撞 0.0004-0.0015、dnf 0.011-0.021。规模曲线未饱和（1.20→1.33→2.11）；**总步数=硬通货，面轴=墙钟优化（+20%）**，双卡并跑另有 SPS 争用（87K→45K）。8×（1.95e9）断点在盘等卡（artifacts/t3-f16-8x，323M/1946M）。冠军权重本地+lab42 双份（含 config.yaml）。
- 渲染验收在位：`artifacts/NIGHT-REPORT.html` v3 + 冠军回放画廊 render-CHAMP{,2}（同 8 场景 vs B 基线）。
- **GPU 现状（11-12 晚）**：lab42 两卡均已让出（1 号→strata 服务，0 号→高优任务），本地 3060 已让；全部战役进程消零。
- **夜间链与断点**：`scripts/night_chain.sh`（8×→16×→cap2048→16xs43，硬停 10:15）；**cap2048 容量臂 CUDA OOM×2 DEAD**（backbone 2048 与 strata 共卡无 headroom；重试需独占卡或 backbone 1536 折中）；夜间战役过程 `.scratch/night-2026-11-11.md`；复现（R@1× 三种子 0.86-1.28 安全带紧密；R@2× N=2；S@s43 紧密）。
- 内核旋钮谱系：T1 progress → T6 offroad → T7 collision（默认位级=legacy；本树 sha `02786a05…`，lab42 自建 sha 不同——跨机相对比较成立，锚点重定仍待 owner）。
- 评测资产：`t3_eval_ckpt`（carla_fast 双面）+ `t4-16x-bigeval`（16× 冠军 400 场景大样本）；v2 弧线证据（t2-*）在案（ADR-0006）。本地 CPU eval 不可行（13.6min/场景）。
- **侦察定论（.wayfinder/research/）**：`val14-data-recon.md`（路径① = 数据工程日：val split/地图/carl_nuplan 三缺口，公开镜像全查无 val，AWS 需注册；nuPlan train 1.5TB 在 lab42 `/media/pc/data2/rino_we/nuplan`）；`selfplay-v4-findings.md`（提案②修正：训练实为同策略 naive self-play（非 IDM），真实缺口 = 群体同质化，v4 = SPACeR 参考锚定快照池，pufferl Python 层改造 1-2 天）。

## Docs 台账

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（冻结）；`.wayfinder/research/` 六份。
- `docs/2026-11-12-量级改进路径提案.md`（冻结提案，待 owner 评审；决策点 5 项；夜间执行后两处修正见 research notes）。
- `.wayfinder/research/selfplay-v4-findings.md` + `val14-data-recon.md`（2026-11-13，路径②①实况修正）。
- 夜间战役过程叙事在 `.scratch/night-2026-11-11.md`（易碎层）。

## Next（下一步）

- **待 owner**：①nuPlan AWS 注册（val14 外部标定前提；或从上游集群搬 carl_nuplan+地图）；②ADR-0009 候选裁决（规模轴 III 战果：16× 峰位 + 新冠军，是否封冠军/继续容量轴）；③push origin（本地领先 70+ commits）；④GPU 归还后：16xs43 断点续完（76% 在盘，~1.5h）、cap2048 独占卡重试、超长程 lr 退火臂。
- 增益轴排序（夜战后更新）：容量（OOM 待解）> 自博弈 v4（工程 1-2 天，research note 在案）> 超长程 lr 退火 > 继续堆规模（已过峰）。
- 「顶格率」指标待建；v3/16× 外部基准验证（val14）数据未到位。
- 雾区序不变：异质 sim-agent（与 v4 合流）；外部验收；log→map；E1a/E1b。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②可部署、真实运营 log SOTA 的 planner。**验收口径（owner）**：安全+效率并重（物流/载人），速度贴车道限速顶格，瘫痪型保守不合格。
- 方法论铁律：全量+eval 双面判读（探针读瞬态）；内核钉死+热路径零 malloc；上游只读。
