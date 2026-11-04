# STATE

As-of: a920284
（2026-11-03 基线导入 + strata bootstrap 会话；master = 上游 3.0 全历史 + 本仓资产）
约定：AGENTS.md 是放置规则权威；strata 分层（STATE / 台账）自 2026-11-03 启用；TerraZero 复刻 effort 的活图在 `.wayfinder/map.md`。

## Where（现状）

- master 基线 = 上游 PufferDrive `3.0` @ `2c1ca3e4`（含上游 6 个新提交：Longest6v1/Alpasim eval、self-play eval 改进、overspeed reward 降至 comfort 量级、speed limit penalty 实验+revert）；tag `baseline/pufferdrive-3.0-2c1ca3e4`。
- remotes：`origin` = zoo-yuan/Slipstream（本仓）；`upstream` = Emerge-Lab/PufferDrive（只读）。
- 本仓自有资产：`.wayfinder/`（TerraZero 复刻地图：6 项 decisions、4 张前沿票）+ `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`。
- 已裁决（详见地图 Decisions so far）：Slipstream 为主仓；能力包 = 程序化场景生成层 + 零演示配方 + sim-agent（吞吐改造出界）；验收 = 内部优先 + 外部榜单为辅；算力 = 单服务器 GPU；`9_bernhard_dev` = 挖矿源。
- 兄弟仓 `../PufferDrive`：Nova bridge 线（`ACTION_TYPE_CONTROL` WIP 未提交，归那边收尾）+ 全套 strata 文档（STATE/ADR/Terra 分析原稿）。

## Docs 台账（冻结检查依据）

- `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` — 自 `../PufferDrive` 仓拷入的冻结分析（五成分因果链、缺口分析、逐工作落地映射）；原仓有同源版本与 ADR 语境。
- `.wayfinder/research/` — 四份调研产物（TerraZero 报告精读 / log→map 管线 / 可借件盘点 / 规则执法核对），随票 Resolution 索引。

## Next（下一步）

- P0：hazard 生成器最小原型（wayfinder 票 07，随基线导入完成已解锁；规格 = 八生成器 + 放置规则，见 `.wayfinder/tickets/`）。
- P1：训练配方立场（票 06，含 progress shaping 补刀裁决）；异质保真档位（票 08）。
- P1：评测资产就位（票 09：内部 E1a/E1b 依赖清单 + InterPlan 克隆核验 + val14 harness 选型）。
- 随后：规则执法三补丁（红灯 footprint 扫越 / 停牌 dwell 采样化 / 行人豁免 gate，≈2-4 工程日，见 rule-enforcement-checklist）。

## Long-term（长期不变量）

- 与 `../PufferDrive` 仓 STATE 的集成目标一致：rino 数据湖 + 混合智能体闭环世界，在 Nova 19 输入 / 4 输出契约内训练 planner；TerraZero 复刻 = 场景多样性与 self-play 能力的供给线。
- 上游跟进纪律：`upstream` 只读；`9_bernhard_dev` 挖矿不并基线；AGPL（behavior-bench）只读不并入。
