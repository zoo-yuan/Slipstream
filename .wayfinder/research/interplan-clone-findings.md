# InterPlan 克隆核验（票 09 外部支 · A）

> 执行：Slipstream 实现子代理，2026-11-05。克隆命令按票面：
> `git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/mh0797/interPlan external/interplan`（代理 127.0.0.1:7980 已配，一次成功，无重试）。
> 上游情报源：`../behavior-bench/NUPLAN.md` §interPlan Evaluation；04 票 `ecosystem-findings.md` §1。

## 1. 就位状态

| 项 | 值 |
|---|---|
| 位置 | `external/interplan`（已在 `.gitignore` 的 "External local clones" 段，不入库） |
| HEAD SHA | `a85024a48935c92ec0d7c2a8cc2e583d01514469`（2024-08-29 "Fix typo"，即上游最新） |
| 体积 | 1.7 MB（纯代码+配置，无数据） |
| **LICENSE** | **Apache-2.0** ✓（根目录 LICENSE 全文为 Apache License 2.0） |
| 维护状态 | 最后一提交 2024-08-29；News 区最后条目同日（换掉无解 token `f8684ea2cf1c512b`→`9e7581ef72155c9f` 并重评全部模型） |

**License 判定：可安全用于验收与外部榜单复算**。Apache-2.0 允许复制/修改/商用，仅要求保留 NOTICE 与 license 副本；与 MIT 基线无冲突，理论上可并入，但按 05 票纪律仍保持 external/ 只读参考、不并入代码树。

## 2. 场景集构成（定义位置已核）

- **80 官方 split**：`interplan/planning/script/config/common/scenario_filter/interplan10.yaml`，显式 `scenario_tokens` 列表 **80 条**（实测 grep 计数 = 80）= 8 类 × 每类 10：low/medium/high traffic density（变道类）、approaching construction zone、avoiding crashed cars、jaywalking pedestrian、nudge/overtake parked vehicle。`num_scenarios_per_type: 10`。
- **335 全量**：同目录 `benchmark_scenarios.yaml`，显式 token 列表 **335 条**（实测计数 = 335）。README 自述 "335 diverse scenarios"，两数一致。
- **场景修改字典**：`.../scenario_filter/modifications/interPlan_modifications.yaml`——按原始 nuPlan token 组织：导航目标（left/right/straight 坐标）、`extra_agents_to_spawn`、`roadblock_locations_to_spawn_agents`、`special_scenario`（stopped_vehicle/stopped_bus/cones/pedestrian 起终点）。token 后缀编码：`dl/dm/dh`=低中高密度、`gl/gr/gs`=目标左/右/直、`oa/om/oc`=对手 assertive/mixed/cautious；`-sN`=特殊场景索引。
- **场景构建器**：`NuPlanModifiedScenarioBuilder`（`scenario_builder/nuplan_db/modified_nuplan_scenario_builder.py`）——在 nuPlan 原始场景上动态应用修改；`include_cameras: false`。

## 3. Metrics runner 入口与依赖

- **入口**：`interplan/planning/script/run_simulation.py`（Hydra main，基于 nuPlan devkit 的 `default_simulation` 配置，`NUPLAN_SIMULATION_ALLOW_ANY_BUILDER=1`）。调用示例（官方 `scripts/sim_pdm_closed.sh`）：
  `+simulation=default_interplan_benchmark planner=<name> scenario_filter=interplan10|benchmark_scenarios` + `hydra.searchpath` 含 interplan/nuplan/tuplan_garage 包。
- **默认实验配置** `default_interplan_benchmark.yaml`：observation=`idm_modified_agents_observation`（**背景交通 = IDM 反应式**，behavior 支持 ASSERTIVE/CAUTIOUS/MIXED/STANDARD，radius 100 m）、ego_controller=`two_stage_controller`（nuPlan 标准 LQR+跟踪）、planner=`simple_planner`（须覆盖）。
- **指标**：`simulation_benchmark.yaml` = 低层 9 项（lane change/jerk/lat-lon acc/yaw/progress/lane_changes_to_goal）+ 高层 7 项（drivable area、at-fault collisions、TTC、speed limit、comfort、making progress、driving direction）+ 5 类场景特有 `ego_sorts_*`。
- **汇总**：`benchmark_weighted_average.yaml`——加权平均（progress 5.0、TTC 5.0、lane_changes_to_goal 4.0、speed_limit 4.0、comfort 2.0、其余 1.0）**再乘以门禁项**（collisions / drivable area / making progress / direction + 场景特有 sorts）；8 个场景类型等权 1。
- **依赖**：`requirements.txt` = 仅 `nuplan-devkit`；`setup.py` install_requires 空，Python≥3.9.12。另需 hydra/omegaconf/pytorch_lightning（随 devkit 带入）。
- **是否需要 nuPlan 数据集本体：是**。`scenario_builder/interplan.yaml` 用 `NUPLAN_DATA_ROOT/nuplan-v1.1/trainval`（.db 文件）+ `NUPLAN_MAPS_ROOT`（nuplan-maps-v1.0）；`db_files` 可本地指定绕开 data_root。**sensor blobs 不需要**（仿真不用传感器数据）。

## 4. 与 PufferDrive 的对接点（只看，不并——behavior-bench 为 AGPL）

- behavior-bench 的挂点 = `pufferlib/nuplan_integration/config/simulation/planner/pufferdrive_planner.yaml` → `PufferDrivePlanner(AbstractPlanner)`（`planner.py`，549 行）：nuPlan 状态→PufferDrive 1120 维 obs（经 C binding `compute_obs_external`，1:1 训练口径）→ PPO(LSTM) 离散动作（91=7 accel×13 steer）→ Python 移植的 `move_dynamics` bicycle 模型推 1 步 + 恒速外推 79 点 → `InterpolatedTrajectory`。运行时把该 planner yaml 拷进 nuplan-devkit 的 planner 配置目录即可，interPlan runner 经 `hydra.searchpath` 发现它。
- **注意执行语义差异**：TerraZero/02 票口径是「jerk 动作→nuPlan 标准 LQR 执行」；behavior-bench 的 pufferdrive_planner 实际是「动作在 planner 内部用 bicycle 模型执行，nuPlan LQR 只近完美跟踪轨迹」。若复算 interPlan 榜（其 baseline 用 IDM/PDM/LLM 系 planner，走 nuPlan 标准 two_stage 控制），PufferDrive 系 planner 与榜上数字存在执行模型差异，报告中须标注。

## 5. 缺口清单（按阻塞度排序）

1. **nuPlan 数据集本体未就位**（trainval .db + maps v1.0；需从 nuscenes.org 下载并受其 license 约束——非商用条款，验收前须确认可接受）。这是跑 interPlan 80/335 任何一条场景的硬前提。
2. **ego policy 未定**：interPlan 榜比的是 nuPlan planner 口径；用 PufferDrive/Nova 系 ego 需 behavior-bench 的 AGPL planner 插件思路（只读重实现）或本仓 cosim planner 改造（见 val14-harness-options.md 方案 D 的复用讨论）。
3. **runner 未在本机跑通过**（无 nuPlan 数据；未验证 nuplan-devkit 在本机 Python/g++ 环境可安装——兄弟仓 STATE 记录 g++ 9.4 不支持 C++20，会卡 `pufferlib._C`，devkit 本体不受此影响但 torch 依赖需核）。
4. **上游停更**（2024-08 后无提交）；80 split 中 `7bc92573f75d5a4c-*` 组 token 全部来自换 token 前的旧新闻口径，需在跑通时对照 News 校验（仓库内 yaml 已是修正版，风险低）。
5. **外部榜单可比性**：interPlan 官方数字（IDM 31 / PDM-Closed 42 / HybridLLM 53）来自 nuPlan two_stage 执行 + IDM 反应式背景；任何自研 ego 直接同表需声明执行模型与背景策略差异。

## 6. 结论

资产就位判定：**仓库克隆完成、license 干净（Apache-2.0）、场景/指标定义全部本地可审计**。剩余缺口全部集中在 nuPlan 数据集获取与 ego 接入形态，不阻塞票 09 的「记录在案」完成标准，但阻塞「实际跑分」。
