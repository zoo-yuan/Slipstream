# val14/InterPlan 外部标定：数据可达性侦察（量级提案路径① 实况核查）

日期：2026-11-12 夜 · 触发：提案①「数小时纯评测」成本估计的实地验证（owner 已授权夜间执行）
结论先行：**路径① 不是数小时纯评测，是数据工程日**。三缺口未闭环前不可点火。

## 1. 仓内资产核验（在位的）

| 资产 | 位置 | 状态 |
|---|---|---|
| nuPlan cosim 桥（PufferDrivePlanner + 双 challenge 预设 + 全套 eval hack 旋钮） | `pufferlib/ocean/cosim/nuplan/` | ✅ 完整（上游 2026-09-03 跑过 val14） |
| run_nuplan_planner.sh（val14_split 全量 ~900 场景；上游用 slurm 集群跑） | 同上 | ✅ 脚本在，集群绑定需剥离 |
| InterPlan 插件（nuPlan 生态 benchmark：metrics/scenario_builder/simulation） | `external/interplan/` | ✅ 源码 1.7MB |
| nuplan-devkit 源码 | lab42 `/home/zoo/rino_we/nuplan-devkit` | ✅ clone 在，环境未建（py3.9 老依赖栈） |

## 2. 三缺口（全部未闭环）

1. **val split 数据**：lab42 `/media/pc/data2/rino_we/nuplan` 只有 train 9 城 1.5TB（ModelScope `PengfeiGuo/NuPlan`，且 singapore/vegas_1 两城 server 端有断块）；**镜像全库 12864 文件无任何 val 路径**。
2. **nuPlan 地图（gpkg）**：`NUPLAN_MAPS_ROOT` 上游指向集群路径，本地/远端均无；仓内仅 1 个单地图 bin（00018a38…，非 val14 城市集）。
3. **carl_nuplan 包**：hydra searchpath 依赖 `pkg://carl_nuplan…`（含 `val14_split` scenario_filter = 14 log 名单的持有者）；GitHub 匿名 clone 未定位到正确 org/repo（HF/ModelScope/openxlab 均无）。
4. AWS 官方桶 `s3://nuplan` 匿名 AccessDenied——**官方数据需 motional/AWS OpenData 注册凭证（owner 侧动作）**。

## 3. 通盘成本（若数据到位后）

- 独立 devkit 环境（py3.9 + hydra 老 pin + sqlalchemy）：1-3h 依赖地狱风险
- 桥接调通：上游自述「只有 worker=sequential 端到端验证过（单场景）」；ray_distributed 128 workers 曾 OOM 160G——lab42 192 核可承载但要实测
- 全量 900 场景 ×2 challenge：sequential 慢、集群规模才快——单机过夜量级
- **改造提示**：run_nuplan_planner.sh 头部 apptainer/srun 段需条件剥离（lab42 无 slurm）；PY/PD/三个 ROOT 环境变量已参数化 ✓

## 4. 给 owner 的行动项（提案决策点 1 的实底）

1. 若要外部标定：注册 nuPlan AWS OpenData → 下载 val14 14 log db + 2 城 gpkg 地图（~40GB，磁盘 67G 够）+ 定位/获取 carl_nuplan 源（上游集群 `/scratch/yw4142/venvs/pufferdrive` 的 site-packages 直接打包搬运是最快路径）
2. 替代口径（零新数据）：把「外部」降级为上游同款 CARLA benchmark 扩大样本（1600+ 场景 GPU eval，明晚 GPU 空档 1-2h）——统计收紧而非外部战场
3. val14 分数对标系（README 在案）：PDM-Closed 92 / IDM 77 / TerraZero 94.19 (#1)——本仓策略首个外部锚点将直接进入该表

## 5. 夜间执行实况（2026-11-12 21:00-22:00）

- 已验证：ModelScope tree API 全量枚举（无 val）、AWS s3 匿名拒绝、HF/openxlab 无镜像、GitHub carl_nuplan org 定位失败
- 本地 CPU eval 100 场景标定：13.6 min/场景（ETA 22.5h）——CPU 大样本 eval 不可行，GPU 空档再补
