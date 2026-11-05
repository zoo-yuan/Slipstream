# STATE

As-of: baf5e4f
（2026-11-05 hazard 原型会话：M0/M1/M2 完成 + 内核钉死 + 全量重渲；master = 上游 3.0 全历史 + 本仓资产）
约定：AGENTS.md 是放置规则权威；strata 分层（STATE / 台账）自 2026-11-03 启用；TerraZero 复刻 effort 的活图在 `.wayfinder/map.md`。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`（含上游 6 个新提交：Longest6v1/Alpasim eval、self-play eval 改进、overspeed reward 降至 comfort 量级、speed limit penalty 实验+revert）；tag `baseline/pufferdrive-3.0-2c1ca3e4`。
- remotes：`origin` = zoo-yuan/Slipstream（本仓）；`upstream` = Emerge-Lab/PufferDrive（只读）。
- 本仓自有资产：`.wayfinder/`（TerraZero 复刻地图）+ `scripts/hazard_demo.py`（M0-M2 三模式：baseline / static-hazards / jaywalker）+ `docs/`。
- **内核钉死纪律（2026-11-05 起）**：运行只用本仓自编译 .so（`NO_TRAIN=1 python setup.py build_ext --inplace --force`，可复现构建，sha256 `443d302f…`）；兄弟仓解释器仅作 Python 借用（`PYTHONPATH=$PWD`）；所有 summary 带 kernel 溯源块（pufferlib 路径 + .so sha256 + git sha），非本仓内核 RuntimeError 拒跑。
- **仿真锚点（新内核，2026-11-05 重定）**：baseline seed42 = 200 步/31 碰撞/0 offroad（旧内核 135/78/11 已废）；static-hazards 碰撞增量 s42 +116.7%/s43 +133.3%/s44 +16.0%；jaywalker 6/6 横穿完成、min 距离 1.012m、1 次接触（s44）、减速事件 0（诚实零）、cut-in 3×abandoned（替代方案已录）。
- 已裁决（详见地图 Decisions so far）：Slipstream 为主仓；能力包 = 程序化场景生成层 + 零演示配方 + sim-agent（吞吐改造出界）；验收 = 内部优先 + 外部榜单为辅；算力 = 单服务器 GPU；`9_bernhard_dev` = 挖矿源。
- 兄弟仓 `../PufferDrive`：Nova bridge 线（`ACTION_TYPE_CONTROL` WIP 未提交，归那边收尾）+ 全套 strata 文档（STATE/ADR/Terra 分析原稿）。

## Docs 台账（冻结检查依据）

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` — 自 `../PufferDrive` 仓拷入的冻结分析（五成分因果链、缺口分析、逐工作落地映射）；原仓有同源版本与 ADR 语境。
- `.wayfinder/research/` — 四份调研产物（TerraZero 报告精读 / log→map 管线 / 可借件盘点 / 规则执法核对），随票 Resolution 索引。

## Next（下一步）

- P0：hazard 原型 M3（配置面 hazard_* 旋钮 + init 前校验 + C/Python 测试 + mix 混合演示 + 关旋钮回归不变）——原型票最后一块。
- P1：训练配方立场（票 06，含 progress shaping 补刀裁决）；异质保真档位（票 08）。
- P1：评测资产就位（票 09：内部 E1a/E1b 依赖清单 + InterPlan 克隆核验 + val14 harness 选型）。
- 随后：规则执法三补丁（红灯 footprint 扫越 / 停牌 dwell 采样化 / 行人豁免 gate，≈2-4 工程日，见 rule-enforcement-checklist）。

## Long-term（长期不变量）

- 与 `../PufferDrive` 仓 STATE 的集成目标一致：rino 数据湖 + 混合智能体闭环世界，在 Nova 19 输入 / 4 输出契约内训练 planner；TerraZero 复刻 = 场景多样性与 self-play 能力的供给线。
- 上游跟进纪律：`upstream` 只读；`9_bernhard_dev` 挖矿不并基线；AGPL（behavior-bench）只读不并入。
