# STATE

As-of: 夜班 2026-11-18 晨（E3@4×=v5 最强 ret 5.32/速度 4.71(+88%)/coll 0.0018；16× N=2 闭环 2.964/2.791；Y 首跑通；cap2048-v2 与 mix1600 在途）
（2026-11-17：owner 立项物流效率口径；v4 自博弈与 v5 效率轴均代码就绪等 GPU；SPACeR KL 锚定已采纳待提案；lab42 内核已重建同步）
约定：AGENTS.md 是放置规则权威；strata 分层自 2026-11-03 启用；TerraZero 复刻 effort 活图在 `.wayfinder/map.md`（已完成）。**裁决真值在 `docs/adr/0001-0007`**（基底/配方v1/执法/运行时独立/奖励区间旋钮谱系/保守吸引子/熵破解配方v3）。

## Where（现状）

- **规模轴 III 完成（2026-11-12/13 夜，owner 授权窗口）**：R 配方 s42 @16x256 face，**16× = 新双面冠军 ret 2.978/2.792（400 场景大样本，coll 0.0000/0.0023，dnf 0.0005/0.0182，goals 2.89，速度 2.83）**；规模曲线 1.20→1.33→2.11→2.42→**2.98（16×）**→**2.49（32× 回落）**——**步数硬通货汇率在 16× 拐头**，32× 后期训练 ret 波动提示超长程 lr 不退火不稳。权重：lab42 `artifacts/t3-f16-16x/`（16×s43 复现断点在盘 76% 停）；32× `artifacts/t3-f16-32x/`；8× `artifacts/t3-f16-8x/`。交接页 `.scratch/handoff/2026-11-13-night-scale-III.md`。
- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`；`upstream` 只读；**lab42 双机线**：`zoo@192.168.202.42:/home/zoo/rino_we/808-Scale-AD/Slipstream`（2×RTX4090，同步走 `git push lab42` + `scripts/sync_lab42.sh`；父目录有他人项目勿动；192 核）。
- **配方 v3 双冠军 N=2（ADR-0008）**：R@4× s42（8×128）**2.113/1.825**、s43（16×256）**2.100/1.909**；碰撞 0.0004-0.0015、dnf 0.011-0.021。规模曲线未饱和（1.20→1.33→2.11）；**总步数=硬通货，面轴=墙钟优化（+20%）**，双卡并跑另有 SPS 争用（87K→45K）。8×（1.95e9）断点在盘等卡（artifacts/t3-f16-8x，323M/1946M）。冠军权重本地+lab42 双份（含 config.yaml）。
- 渲染验收在位：`artifacts/2026-11-11-NIGHT-REPORT.html` v3 + 冠军回放画廊 render-CHAMP{,2}（同 8 场景 vs B 基线）。
- **GPU 现状（11-12 晚）**：lab42 两卡均已让出（1 号→strata 服务，0 号→高优任务），本地 3060 已让；全部战役进程消零。
- **夜间链与断点**：`scripts/night_chain.sh`（8×→16×→cap2048→16xs43，硬停 10:15）；**cap2048 容量臂 CUDA OOM×2 DEAD**（backbone 2048 与 strata 共卡无 headroom；重试需独占卡或 backbone 1536 折中）；夜间战役过程 `.scratch/night-2026-11-11.md`；复现（R@1× 三种子 0.86-1.28 安全带紧密；R@2× N=2；S@s43 紧密）。
- 内核旋钮谱系：T1 progress → T6 offroad → T7 collision（默认位级=legacy；本树 sha `02786a05…`，lab42 自建 sha 不同——跨机相对比较成立，锚点重定仍待 owner）。
- 评测资产：`t3_eval_ckpt`（carla_fast 双面）+ `t4-16x-bigeval`（16× 冠军 400 场景大样本）；v2 弧线证据（t2-*）在案（ADR-0006）。本地 CPU eval 不可行（13.6min/场景）。
- **侦察定论（.wayfinder/research/）**：`val14-data-recon.md`（路径① = 数据工程日：val split/地图/carl_nuplan 三缺口，公开镜像全查无 val，AWS 需注册；nuPlan train 1.5TB 在 lab42 `/media/pc/data2/rino_we/nuplan`）；`selfplay-v4-findings.md`（提案②修正：训练实为同策略 naive self-play（非 IDM），真实缺口 = 群体同质化，v4 = SPACeR 参考锚定快照池，pufferl Python 层改造 1-2 天）。

## Docs 台账

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（冻结）；`.wayfinder/research/` 六份。
- `docs/2026-11-12-量级改进路径提案.md`（冻结提案，待 owner 评审；决策点 5 项；夜间执行后两处修正见 research notes）。
- `docs/2026-11-13-北极星对齐与数据轴裁决输入.md`（冻结：两星距离评估 + val14 四问 + Apollo bag 格式要求/owner 清单；北极星距离条首条存档）。
- `docs/2026-11-13-SOTA模型规格与ONNX导出及评测方法.md`（实证：obs 821 布局/12 类动作/7.6M 参数；ONNX 导出脚本 `scripts/export_onnx.py` + ORT 对拍 8.6e-6；评测面构造与指标计算逻辑）。
- `.wayfinder/research/selfplay-v4-findings.md` + `val14-data-recon.md`（2026-11-13，路径②①实况修正）。
- 夜间战役过程叙事在 `.scratch/night-2026-11-11.md`（易碎层）。

## Next（下一步）

- **待 owner**：①v5 效率轴已立项（owner 2026-11-17：物流配送，效率与安全同等，效率=流畅+顶限速；goal_speed=3.0 与真实需求相去甚远）——门1 提案进行中（.scratch/proposals/v5-goal-speed-proposal.md，goal_speed_mode=lane_limit 动态化+奖励联动+扫臂）；②v4 首跑排期（代码三门就绪；GPU 回归链 watcher 挂 lab42）；③**SPACeR KL/似然奖励锚定已采纳**（更新票 06 旧口径，与 snapshot-pool 正交可叠加，待门1提案）；④nuPlan AWS 注册 / Apollo 采样 bag+地图（双缺口不变，G1 转换器须保留 lane speed_limit 字段）；⑤push origin（本地领先 87+）。
- **三门夜班交付（cb4b99e）**：SelfplayMixer（8 单测+冒烟，off 路径 RNG 位等价，Y/Z 臂）；speed_attainment.py（三门闭环：独立重算一致+三重交叉全零）；render-CHAMP16 画廊；nuplan-py39 环境（import+读 train db 验证）；64 场景顶格率样本跑中。
- 已知缺口（验证者发现，见 handoff §3）：smoke 层伙伴来源无断言；Y/Z 与 v3 SPS 不可直比（compile 分裂+probe 不含自博弈）；臂 epoch<interval(50) 时快照池退化。
- 增益轴排序不变：容量（OOM 待解）> 自博弈 v4（代码就绪）> 超长程 lr 退火 > 堆规模（已过峰）；**新增：v5 效率轴（goal_speed）待裁决**。
- 「顶格率」指标待建；v3/16× 外部基准验证（val14）数据未到位。
- 雾区序不变：异质 sim-agent（与 v4 合流）；外部验收；log→map；E1a/E1b。

## Long-term（长期不变量）

- **双目标**：①复现 TerraZero；②可部署、真实运营 log SOTA 的 planner。**验收口径（owner）**：安全+效率并重（物流/载人），速度贴车道限速顶格，瘫痪型保守不合格。
- 方法论铁律：全量+eval 双面判读（探针读瞬态）；内核钉死+热路径零 malloc；上游只读。
