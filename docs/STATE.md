# STATE

As-of: adb8dc3
（2026-11-05 票 09 收章：评测资产就位；hazard 原型票已闭）
约定：AGENTS.md 是放置规则权威；strata 分层（STATE / 台账）自 2026-11-03 启用；TerraZero 复刻 effort 的活图在 `.wayfinder/map.md`。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`（含上游 6 个新提交：Longest6v1/Alpasim eval、self-play eval 改进、overspeed reward 降至 comfort 量级、speed limit penalty 实验+revert）；tag `baseline/pufferdrive-3.0-2c1ca3e4`。
- remotes：`origin` = zoo-yuan/Slipstream（本仓）；`upstream` = Emerge-Lab/PufferDrive（只读）。
- 本仓自有资产：`.wayfinder/`（TerraZero 复刻地图）+ `scripts/hazard_demo.py`（四模式：baseline / static-hazards / jaywalker / mix）+ `config/hazard_demo.yaml`（hazard 旋钮，init 前校验）+ `tests/test_hazard_demo.py`（32 passed）+ `docs/`。hazard 原型票已闭（M0-M3 全验收）。
- **内核钉死纪律（2026-11-05 起）**：运行只用本仓自编译 .so（`NO_TRAIN=1 python setup.py build_ext --inplace --force`，可复现构建，sha256 `443d302f…`）；兄弟仓解释器仅作 Python 借用（`PYTHONPATH=$PWD`）；所有 summary 带 kernel 溯源块（pufferlib 路径 + .so sha256 + git sha），非本仓内核 RuntimeError 拒跑。
- **仿真锚点（K4a 新执法面，2026-11-09 重定）**：执法默认 = footprint 扫越 + dwell [0.5,1.5]（TerraZero 口径；legacy 逃生门可显式传旧旋钮复现）。baseline = s42 200/26/1（红灯 latch 31、停牌 45）/ s43 192/21/0 / s44 200/11/0；kernel-static 增量 +140%/+41%/+64%；mix s42 = 10→25。完整表：artifacts/reanchor-k4/comparison.md。（旧锚 200/31/0 属 legacy 执法面，可经逃生门复现）
- 已裁决（详见地图 Decisions so far）：Slipstream 为主仓；能力包 = 程序化场景生成层 + 零演示配方 + sim-agent（吞吐改造出界）；验收 = 内部优先 + 外部榜单为辅；算力 = 单服务器 GPU；`9_bernhard_dev` = 挖矿源。
- 兄弟仓 `../PufferDrive`：Nova bridge 线已落生产代码（`4875b53f`：ACTION_TYPE_CONTROL + acc wash-in + Apollo lane matching）+ 全套 strata 文档；E1a/E1b 桥依赖现状见 `.wayfinder/research/e1-dependencies.md`（公共 7 具备/4 缺/3 阻塞；E1a 另缺 Nova 权重）。
- 评测资产：`external/interplan`（Apache-2.0，80+335 场景，缺 nuPlan 数据本体）；val14 官方闭环 = 仓内 MIT cosim/nuplan 栈（选型见 `.wayfinder/research/val14-harness-options.md`）。

## Docs 台账（冻结检查依据）

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` — 自 `../PufferDrive` 仓拷入的冻结分析（五成分因果链、缺口分析、逐工作落地映射）；原仓有同源版本与 ADR 语境。
- `.wayfinder/research/` — 四份调研产物（TerraZero 报告精读 / log→map 管线 / 可借件盘点 / 规则执法核对），随票 Resolution 索引。

## Next（执行清单，2026-11-05 地图完成后移交，按优先级）

1. **hazard 内核子系统化**（K 系列，验收纪律同票 07 长期标准）：
   - **K1 内核静态放置**：parked/簇/锥桶移入 C reset 路径，env config 驱动（init 前校验/热路径零 malloc）；验收 = `--placement kernel` 产出同款渲染产物 + 合法性 0 违例 + C 测试绿 + python 路径回归逐位不变。
   - **K2 内核脚本行人**：walker 由内核驱动（unicycle 步进+班次）；验收 = M2 同款度量（横穿率/最小距离/接触）从纯 env 产出。
   - **K3 配时开关 + 执法三补丁**：真值配时保真开关（bin states 生效）；红灯 footprint 扫越/停牌 dwell 采样/行人豁免 gate；验收 = 单测 + 行为差异数字（违规计数前后对比）。
   - **K4 训练就绪整合**：hazard 旋钮进 puffer_drive.yaml + 全量回归 + **PPO 短跑 smoke（1e6 步 reward 曲线图，hazard 开/关对比）**——zero-to-one 的第一信号。
2. **零演示 RL 主轴训练 run**：progress shaping 补齐 + 评测期条件化 + 既有 reward conditioning；混合对手旋钮保留非默认。
3. **异质 sim-agent 线**：骑行者 bicycle 网格 + 卡车分布扩展 + per-type heads（65K 轻量线起点，WOSAC 角色）。
4. **外部验收**：nuPlan 数据获取（license 核验）→ 仓内 cosim/nuplan 栈跑通 val14 → InterPlan 接入。
5. **log→map 摄取**：Apollo G1-G3 几何转换器 + WOMD/nuPlan strip-agents 工具（单一中间表示 + bin writer，五源共享）。
6. E1a/E1b：补桥三件套 G8/G9/G3 + Nova 权重（owner 侧解锁）。

## Long-term（长期不变量）

- **双目标（owner 战略框架，2026-11-05 定）**：①复现 TerraZero（始终如一）；②在 TerraZero 类环境上训练出**实际可部署、在真实世界（运营 log）拿到 SOTA** 的能力——RL 是确定性路径，replay/IDM/policy 等混合手段增大这个概率。
- 与 `../PufferDrive` 仓集成目标一致：rino 数据湖 + 混合智能体闭环世界，在 Nova 19 输入 / 4 输出契约内训练 planner；TerraZero 复刻 = 场景多样性与 self-play 能力供给线。
- 上游跟进纪律：`upstream` 只读；`9_bernhard_dev` 挖矿不并基线；AGPL（behavior-bench）只读不并入。
