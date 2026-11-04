# 可借件盘点：生成器、参考模型与评测资产生态

> Wayfinder 04 票调研产物。执行：council-deepseek 子代理（2026-11-03）。
> 方法限制：该子代理无 bash/curl/git（仅 read/grep/find/ls），外部仓库只凭本地克隆+仓内文档核验；git diff 未执行。升级路径见文末。
> 基准引用：`../PufferDrive`（MIT，工作树=`dev/zoo_v3.0`）、`../behavior-bench`（AGPL-3.0）、`../RAP-valeo-wod`（Apache-2.0）。

## 1. InterPlan — 未克隆，接入路径已有文档
- 本地唯一证据：`../behavior-bench/NUPLAN.md` §"interPlan Evaluation"（L194-230）。仓库 `github.com/mh0797/interPlan`；335 场景 8 类（jaywalkers/overtaking/nudging/lane changes/accident sites/construction zones 等）；**80 官方 split = `scenario_filter=interplan10`（每类 10）**，全量 = `scenario_filter=benchmark_scenarios`；自带独立 runner（非 nuPlan 默认）+ 自有 scenario filters；PufferDrive 以 `pufferdrive_planner` Hydra 插件接入。
- 判定：**场景集与 metrics runner 本次未克隆、license 未知、维护状态未知**。接入方式已有（behavior-bench 的 `pufferlib/nuplan_integration/config/simulation/planner/pufferdrive_planner.yaml` 就是现成 Hydra 挂点）。解锁 09/05。

## 2. val14 闭环评测 — 仓内口径 ≠ 官方 val14
- `../PufferDrive/pufferlib/config/evaluation/benchmark.yaml`：`nuplan_multi/nuplan_single` 是 **replay 模式**，读预转 `.bin`（`binaries/nuplan`），1000 场景、seed 42、`scenario_length:200`——**不是 nuPlan 官方 val14 闭环 runner**。
- 官方 val14 闭环 runner + `val14_split` 在 behavior-bench：`../behavior-bench/pufferlib/nuplan_integration/config/simulation/scenario_filter/val14_scenarios.yaml`（14 类场景 ×1/类）+ `planner/pufferdrive_planner.yaml`；`nuplan_to_binary.py` 可把 val14 转 PufferDrive `.bin`（解锁 03/09）。
- "90 CLS-R val14" 栈在 `origin/9_bernhard_dev`（`packed-refs` SHA `09ca44c`），文档（`STATE.md` L14、gap-analysis L73/91）指其含 pose/brownian 噪声 + 最强 self-play 模型；**git diff 未执行，具体评测配置未核**。解锁 01/05/09。

## 3. 参考模型 — SMART 有实现+权重（AGPL），CAT-K 无公开
- **SMART**：`../behavior-bench/pufferlib/planning/smart.py`（SMART 预测模型作 ego/交通控制器，tokenized motion+map codebook，比例控制器转 (acc,steer)）+ `pufferlib/prediction/smart/`（`prediction_model.py` 等）+ `weights/SMART_epoch_030.pt`。**实现齐全、权重在仓**。但整个 behavior-bench 是 **AGPL-3.0**（`../behavior-bench/LICENSE`）。
- **CAT-K**：本地无实现、无权重；仅 11-03 文档转引 TerraZero 论文数字（CAT-K* IL SFT, WOSAC 0.766）。**未找到公开仓库**。解锁 06（SMART 可作 SPACeR π_ref 或 traffic agent）。

## 4. 程序化场景生成库 — 无本地克隆，hazard 放置逻辑均无
- **scenarionet / ScenarioNet**：仅 `../RAP-valeo-wod/process_data/create_openscene_metadata*.py` import `scenarionet.converter.nuplan`（Apache-2.0 仓库引用，scenarionet 本体未克隆）。价值在 **nuPlan→向量化地图/场景转换器**（借给 03 的 log→map），**无 hazard 放置/长尾采样器**。
- **CommonRoad / KB3D**：本地零痕迹（find 无结果）。**未找到公开仓库**；CommonRoad 是场景格式+规划基准，KB3D 未见实现。
- 结论：**hazard 放置逻辑无处可借**，须自建（印证 `docs/2026-11-03...md` §4A：C 内核 ~600-1000 行）。解锁 07/02。

## 5. 上游分支挖矿 — 无法 diff，仅 SHA+文档
- `origin/9_bernhard_dev` = `09ca44c...`：pose/brownian 噪声 + nuPlan 10Hz cosim + val14 最强模型（`STATE.md`、`docs/integration/2026-10-09-gap-analysis` L91/101）。
- `origin/Carla_Road_Processing_Fixes` = `6f3dbcf...`、`origin/InterpretabilityStudy` = `c919c72...`：**仓内无文档，内容不可核验**（需 git diff）。
- 值得并入基线（据文档）：9_bernhard 的噪声注入（ADR-0004 起点）→ 解锁 01/07。Carla 路网处理脚本若属 log→map，解锁 03（未核验）。

## 6. WOSAC 工具链 — 成色好，缺口在"接线+验榜"
`../PufferDrive/pufferlib/ocean/evaluation_utils/wosac/`：
- **已实现**：`metrics.py` 的 kinematic + interaction + map 三组特征全部真实实现（`compute_interaction_features`/`compute_map_features` 均为 torch 向量化，`_METRIC_FIELD_NAMES` 9 字段）；`estimators.py` histogram NLL（`independent_timesteps` 支持）；`wosac.ini` 用 **WOSAC 2025 官方直方图区间+metametric 权重**（collision/offroad 各 0.25）；`evaluator.py` 完整 rollout→score 流程。
- **README 已过时**（说"仅 kinematics"）；实际 interaction/map 已补上。
- **缺口**：(a) 未接入 `puffer benchmark` 目录（README 自述）；(b) 依赖 driver 的 `get_ground_truth_trajectories()/get_global_agent_state()`（`docs/html/...架构分析.html` L301 确认存在，但需跑通验证）；(c) 官方场景 split 与 leaderboard 提交管线缺失，分数未与官方对数。`weights/mimolette/config.yaml` 已有 `control_wosac/256 agents/32 rollouts/init_steps 10` 配置。解锁 05/06/09（sim-agent 验收 ~80% 就位）。

## 对比表

| 资产 | license | 维护状态 | 集成成本 | 解锁票 | 判定 |
|---|---|---|---|---|---|
| PufferDrive evaluation_utils + 18 指标 + 种子回放 | MIT | 活跃（本基线） | 零（已入树） | 05/09 | ✅ 直接可用 |
| PufferDrive wosac/ | MIT | 本基线，未接线 | 低（接 catalog + driver 验证） | 05/06/09 | ✅ 成色好 |
| behavior-bench SMART 实现+权重 | **AGPL-3.0** | 活跃（同生态） | 中（动作空间对齐） | 06 | ⚠️ 只能读/重实现，不可并入 MIT |
| behavior-bench nuplan_integration（val14 闭环 runner + val14_split + nuplan→bin） | **AGPL-3.0** | 活跃 | 中 | 03/05/09 | ⚠️ 同上 |
| InterPlan（场景集+metrics runner） | 未知 | 未核验（未克隆） | 中（Hydra 挂点已有） | 05/09 | ❓ 待克隆 |
| scenarionet（nuplan→vector 转换） | Apache-2.0（RAP 仓库） | 未核验 | 中 | 03 | ⚠️ 借转换器，无 hazard 逻辑 |
| CAT-K / CommonRoad / KB3D | 未知 | 未知 | — | 06/07 | ❌ 未找到公开仓库 |

## Top 借件推荐
1. **仓内 evaluation_utils + wosac**（MIT，零成本，sim-agent 验收地基）→ 09/05。
2. **behavior-bench nuplan_integration**（val14 闭环 runner + val14_split + SMART）→ 09/03/06；**但按 AGPL 纪律：只读算法重实现，不进 MIT 基线**。
3. **9_bernhard_dev 噪声注入**（pose/brownian，文档佐证）→ 01/07；需 git diff 摘取。

## 最大缺口
- **InterPlan 场景集/metrics runner 未克隆且无法联网核验** → 09 的"拉取就位"被阻塞，需 shell 克隆+pin SHA。
- **CAT-K 无公开实现**；**hazard 放置逻辑全生态无处可借** → 07 只能自建 C 内核（~600-1000 行）。
- **AGPL-3.0 污染风险**：最肥的借件（SMART+nuPlan 闭环）不能直接进 MIT 基线。

## 升级证据路径（后续会话可执行）
1. `git -C ../PufferDrive diff 3.0..origin/9_bernhard_dev --stat`（补 §2/§5 核验）→ 并入 01 票证据。
2. `git clone github.com/mh0797/interPlan` 后读 LICENSE + scenario_filter（补 §1）→ 并入 09 票。
3. `pip show scenarionet` / 其 LICENSE（补 §4）→ 并入 03 票。
