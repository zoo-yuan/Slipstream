# Wayfinder Map — Slipstream 复刻 TerraZero 路线

## Destination

在 Slipstream 得到一套**已锁定的复刻路线 spec**：基底承接方式、程序化场景生成层设计（含异源 log→地图管线）、异质 agent 与 hazard 范围、训练配方立场、验收口径全部裁决完毕——开工前无需再做路线级大决策。

## Notes

- 领域：自动驾驶闭环仿真（复刻 TerraZero 类能力：程序化场景生成/编辑、更丰富的异质元素、异源 log 降级为地图来源、self-play 达到强效果）。工作语言：中文。
- 必读上下文（在 `../PufferDrive` 仓）：`docs/STATE.md`（strata grounding）、`docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md`（五成分因果链 + 缺口分析，本图的地基）、`docs/2026-10-31-PufferDrive-vs-TerraZero-仿真器对比.md`。
- 事实基座（2026-11-03 逐分支核验）：**gigaflow 已在 3.0 家族完整实现**（`3.0`、`dev/zoo_v3.0` 代码一致；`origin/9_bernhard_dev` 另含 pose/brownian 噪声与 val14 90 CLS-R 最强模型）；`2.0`/`main` 无 gigaflow。Slipstream 本仓 = 空壳（unborn master，remote zoo-yuan/Slipstream）；`../PufferDrive` = 真实工作仓（官方 Emerge-Lab clone + 本地分支），工作树含**未提交** Nova bridge WIP（`ACTION_TYPE_CONTROL` 直控模式 + 测试）——与 TerraZero 复刻是两条并行线，关系在「基底裁决」票内定。
- 用户既定倾向：复用 PufferDrive 拓展，风险确定性优先于从零；不设 token 预算限制。（基底与范围已裁决，见 Decisions so far）
- tracker 操作口径见 `.wayfinder/TRACKER.md`（frontier 查询、认领、Resolution 约定）。

## Decisions so far

- [异源 log→地图管线：TerraZero 的"任何日志当地图"如何落地](tickets/03-异源log到地图管线调研.md)：纯地图 bin（0 agents）在 gigaflow 下成立且是默认形态；配时无需真值（程序化覆写）；「任何 log 当地图」= 一个几何转换器 + 一个 bin 重打包工具（writer 模板 mirror_map_bin.py），五源共享；最大缺口 = Apollo HD map 转换器。
- [可借件盘点：生成器、参考模型与评测资产生态](tickets/04-可借件盘点-生成器参考模型评测资产.md)：仓内 MIT 评测/WOSAC 栈成色好可直接接线；behavior-bench（AGPL-3.0）藏 SMART 权重+官方 val14 闭环 runner，只可读不可并入；hazard 放置逻辑全生态无件可借（只能自建）；InterPlan 待克隆核验。
- [TerraZero 报告精读：场景生成器可复刻细节](tickets/02-TerraZero报告精读-场景生成器可复刻细节.md)：八生成器+信号三控制器+reward 全表可直接当规格；**reward 无 progress 项**（复刻须补）；评测期条件化=同 checkpoint 免重训切基准；承重墙重估：几何执法升权，3D/吞吐/卡车保真降权；无场景编辑器（只有配置轴+采样）。
- [规则几何执法核对：PufferDrive 现状 vs TerraZero §3.3 清单](tickets/10-规则几何执法核对.md)：六项中 4 已具备、实缺口仅三小补丁（红灯 footprint 扫越、停牌 dwell 采样化、行人豁免 gate，合计 ≈2-4 工程日）；off-road 确带 elevation gate；lane direction 与 §3.3 同构无需新件。
- [基底裁决：复用 PufferDrive 3.0 家族还是别路，Slipstream 如何承接](tickets/01-基底裁决-复用PufferDrive还是别路.md)：Slipstream 为主仓；基线 = `3.0` 干净上游，**保留历史**导入；`9_bernhard_dev` 为挖矿源（噪声/权重按需 cherry-pick）；Nova bridge WIP 留 PufferDrive 仓。执行见「基线导入执行」票。
- [复刻范围与验收口径：能力矩阵、验收标准、算力包线](tickets/05-复刻范围与验收口径.md)：能力包 = 场景生成层 + 零演示配方 + sim-agent（吞吐出界）；验收 = 内部优先 + 外部榜单为辅；算力 = 单服务器 GPU（1e9–1e10 步）。
- [基线导入执行：3.0 历史进 Slipstream master + 工程约定初始化](tickets/11-基线导入执行.md)：已执行——拉最新上游后基线 = `3.0@2c1ca3e4`（比 80a681b4 新 6 提交：Longest6v1/Alpasim eval、overspeed reward 调整）；保留全历史；tag `baseline/pufferdrive-3.0-2c1ca3e4`；资产/strata 两提交（a920284、0ebd01d）。

## Not yet specified

- SPACeR 参考锚定与场景生成层的叠加关系——若「训练配方立场」选混合/锚定路线才展开。
- 场景生成的配置面形态（hazard 概率表 vs 场景模板 DSL vs 交互式编辑器——报告已确认 TerraZero 侧只有配置轴+采样，无编辑器）——等 hazard 原型反馈。
- 大规模训练稳定性细节（lr/entropy/batch 随步数的调度）——等配方立场落定（算力已定单服务器 GPU）。
- gigaflow 真值配时保真开关（若验收要求自有 ODD 真实信号配时；小改 drive.h:3009 块）——03 票反方论点，验收已定内部优先，倾向不做，待配方票一并确认。
- 源→中间几何表示→单一 bin writer 的统一摄取架构（五源共享）——03 票执行者建议，随生成层设计展开。

## Out of scope

- **吞吐形态改造（C 引擎 + GPU 零拷贝推理）**——2026-11-03 范围裁决：单服务器 GPU 包线（1e9–1e10 步）下不立项；若未来升多节点另开新 effort。
- TerraTransfer / TerraTransfer++ 的 vision 对齐与轨迹蒸馏管线（「学会看/学会规划」是另一个 effort，分析见 11-03 文档 §5，不因本图复活）。
- Photorealistic 渲染闭环 RL（CARLA/3DGS 路线）。
- 商业仿真器采购或闭源栈集成（Applied Intuition 栈不可得，只能借配方）。
