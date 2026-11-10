# STATE

As-of: f34dcc9（纯文档会话：蒸馏落层）
（2026-11-05 票 09 收章：评测资产就位；hazard 原型票已闭）
约定：AGENTS.md 是放置规则权威；strata 分层（STATE / adr / lessons）自 2026-11-03 启用；TerraZero 复刻 effort 的活图在 `.wayfinder/map.md`（已完成，11/11 票全闭）。**裁决真值在 `docs/adr/0001-0004`**（基底/配方/执法/运行时独立）。

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

## Next（下一步）

- 待 owner 点火：**hazard-on 长训**（配方 v1 + mix 面，T2 引擎直接复用）→ 内部 18 指标评测 + 失败挖掘回放。
- 待 owner 裁决：index 生成接进 t2 收割器（20 行，使收割=全套仪表盘）。
- 未完基建：**push 仍未落地**（远端落后本地 30+ commits；645MB 全历史首推需好上行窗口手动完成，见 lessons）。
- 雾区候切（wayfinder 地图已完成，此处仅执行序）：异质 sim-agent 线；外部验收（nuPlan 数据 → val14 cosim → InterPlan）；log→map 摄取（Apollo G1-G3）；E1a/E1b（Nova 权重 owner 侧解锁）。

## Long-term（长期不变量）

- **双目标（owner 战略框架，2026-11-05 定）**：①复现 TerraZero（始终如一）；②在 TerraZero 类环境上训练出**实际可部署、在真实世界（运营 log）拿到 SOTA** 的能力——RL 是确定性路径，replay/IDM/policy 等混合手段增大这个概率。
- 与 `../PufferDrive` 仓集成目标一致：rino 数据湖 + 混合智能体闭环世界，在 Nova 19 输入 / 4 输出契约内训练 planner；TerraZero 复刻 = 场景多样性与 self-play 能力供给线。
- 上游跟进纪律：`upstream` 只读；`9_bernhard_dev` 挖矿不并基线；AGPL（behavior-bench）只读不并入。
