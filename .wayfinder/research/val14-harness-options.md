# val14 闭环 harness 选型备忘（票 09 外部支 · B）

> 只读调研，零代码拷贝。执行：Slipstream 实现子代理，2026-11-05。
> 调研对象：`../behavior-bench`（**AGPL-3.0**，只读纪律：只看不并）官方 val14 闭环栈；对照本仓与 `../PufferDrive`（MIT）已有资产。
> 结论先行：**推荐方案 D = 用本仓已有的 MIT cosim/nuplan 栈作为官方闭环 harness**；AGPL 栈（方案 A/C）降为对照参考，方案 B（replay 口径）保留为无 nuPlan 数据时的内部粗口径。

## 0. 先修正一个情报前提

04 票 `ecosystem-findings.md` §2 判定「官方 val14 闭环 runner + val14_split 在 behavior-bench」——只对了一半。核验后发现：

- **behavior-bench 的 val14 栈只是"planner 插件 + 数据转换"**，闭环仿真引擎/指标/runner 全部来自 nuplan-devkit 本体（`run_simulation.py` + `closed_loop_*_agents` + devkit metrics）；
- **本仓（上游基线 2c1ca3e4）已自带一套完整的 MIT nuPlan 闭环接入栈**：`pufferlib/ocean/cosim/nuplan/`（planner.py 652 行 + nuplan_bridge.py 678 行 + Hydra 配置 + run 脚本 + README），设计目标同样是「nuPlan 未修改的 run_simulation.py 闭环 + 官方指标」，且自带 val14 运行记录（README 提及 2026-09-03 val14 run 的 lon-jerk 失败统计）。这套栈上游已提交（`847ce800 Add cosim evals`、`2c1ca3e4`），Slipstream 继承自基线 tag，**无需任何 AGPL 代码即可获得官方闭环口径**。

## 1. behavior-bench 官方 val14 栈构成（AGPL，只读盘点）

| 文件 | 行数 | 作用 |
|---|---|---|
| `nuplan_integration/planner.py` | 549 | `PufferDrivePlanner(AbstractPlanner)`：nuPlan 状态→1120 维 obs（C binding 1:1）→ PPO(LSTM) → 91 离散动作（7 accel×13 steer） |
| `nuplan_integration/observation_builder.py` | 492 | nuPlan DetectionsTracks/map → PufferDrive obs 各槽位 |
| `nuplan_integration/bicycle_dynamics.py` | 114 | drive.h `move_dynamics` 的 Python 移植（CLASSIC bicycle，dt=0.1） |
| `nuplan_integration/trajectory_filler.py` | 108 | 动作→80 点 InterpolatedTrajectory（1 步 bicycle + 79 步恒速） |
| `nuplan_integration/hybrid_controller.py` | 77 | PPO 完美跟踪 / PDM 两段式(LQR) 切换 ego controller |
| `nuplan_integration/config/.../val14_scenarios.yaml` | — | 自带 val14 filter：**15 类 × 1**（含 `on_pickup_dropoff`；注意 ≠ tuplan_garage 官方 val14 = 14 场景） |
| `nuplan_integration/nuplan_to_binary.py` | 445 | nuPlan 场景→PufferDrive .bin（replay/训练用） |
| 核心合计 | ~1263 | 闭环插件本体（不含 nuplan_to_binary） |

**运行方式**（NUPLAN.md）：装 nuplan-devkit + 下载数据（.db + maps，无需 sensor blobs）→ planner yaml 拷进 devkit → `run_simulation.py +simulation=closed_loop_nonreactive_agents|closed_loop_reactive_agents planner=pufferdrive_planner scenario_filter=val14_split`（主口径用 devkit 内置 `val14_split`，非自带 15 类 yaml）。
**执行语义**：动作在 planner 内部经 bicycle 模型执行，nuPlan LQR 只跟踪轨迹（与 TerraZero/02 票「jerk→LQR」口径不同，对比数字时必须标注）。

## 2. 本仓已有资产盘点（MIT，可直接用）

- `pufferlib/ocean/cosim/nuplan/planner.py`：`PufferDrivePlanner(AbstractPlanner)`——影子 Drive env（1 个策略 ego + 不生成的 partner 槽位），每步把 nuPlan `DetectionsTracks`/红绿灯流进影子 env、跑策略、`env.step` 由 **PufferDrive C 内核积分 ego**，回报积分位姿；配置钉死 `perfect_tracking_controller`，位姿发散即报错。goal_source 支持 external/roadblock/roadblock_lane（PDM 式路网 route→目标窗口）。
- `pufferlib/ocean/cosim/nuplan_bridge.py`：nuPlan↔PufferDrive 地图/状态转换；分城市 map-only bin（含 lane graph，来自 123Drive 转换器；`write_drive_bin` 现写空 lane graph 是已知 TODO）。
- 配置预设 `closed_loop_{nonreactive,reactive}_agents_pufferdrive`：**log-replay 背景 vs IDM 反应式背景**两口径都有。
- 运行入口 `run_nuplan_planner.sh`：默认 `SPLIT=val14_split`（carl_nuplan 口径，~900 场景全量；tuplan_garage 口 val14=14 场景，两套 val14 名同实异，**报告须写明用的是哪套 split**），依赖 CaRL 的 `carl_nuplan` 框架；纯 devkit 路径未在本机验证。
- **前置缺口（本机）**：① nuPlan 数据集本体（.db + maps）未下载，license 须先核（非商用条款）；② 分城市 map-only bin（含 lane graph）未生成——123Drive 转换器不在本机，`write_drive_bin` 有 TODO；③ 本机 g++ 9.4 编不了 `pufferlib._C`（C++20），drive binding 可 gcc 直编，nuPlan 栈依赖 torch 侧需核。

## 3. 方案对比

### 方案 A：clean-room 重实现 behavior-bench 栈
- **工作量估计**：核心 ~1263 行 AGPL 代码的等价重写，其中 planner.py（策略装载/LSTM/动作解码）约 300 行可对照自写、observation_builder.py ~492 行是纯格式转换（每槽位字段对齐，最费人力，估 3–5 人日含对拍）、bicycle+trajectory_filler ~220 行可从 drive.h 语义独立移植（1 人日，本仓已有 C 真值可数值对拍）。合计 **≈1–2 人周**（含与训练 env 的 obs 对拍验收）。
- **优点**：MIT 干净；完全贴合 behavior-bench 口径（可与外部报告同表）。
- **缺点**：**重复造轮子**——本仓 cosim planner 已覆盖同一目标且更完整（影子 env、goal 窗口、红绿灯匹配、观测 HTML 可视化、已有 val14 运行记录）；执行语义上 behavior-bench 版（bicycle 内执行）并不比本仓版（C 内核积分）更"官方"，两者都不是 nuPlan 原生 LQR 执行。

### 方案 B：仓内 benchmark.yaml replay 口径 + 标注差异
- `pufferlib/config/evaluation/benchmark.yaml` `nuplan_multi/nuplan_single`：replay 模式读预转 .bin（1000 场景、seed 42、dt 0.1、200 步），18 指标齐（`constants.h` COLLISION..MULTI_LANE_SCORE 共 18 项，已核）+ `eval_replay.py` 种子化回放。本机 binaries/nuplan 目前只有 1 个 fixture bin（2.5 MB），全量需自转。
- **与官方闭环的差异（须标注）**：① 背景交通 = 日志回放，**零反应性**（官方 reactive 口径是 IDM）；② 无红绿灯/停止牌违规判定（`disable_*_infractions: true`）；③ 指标是 PufferDrive 18 项不是 nuPlan 官方 score；④ 场景集不是 val14（是转好的 nuplan bin 批）。
- **工作量**：最小（转 bin 即用），**定位 = 无 nuPlan devkit 数据时的内部粗口径 + 回归基线**，不能当"官方 val14"报数。

### 方案 C：进程隔离借用 behavior-bench
- 形态：AGPL 栈以独立进程运行（本仓只喂 checkpoint、收 JSON 分数），代码不进 MIT 仓。
- **License 风险评估**：AGPL-3.0 的网络条款对本场景影响有限，但**"进程隔离不构成衍生作品"并无稳定法律结论**；只要发布结果数字/对比表引用 AGPL 程序输出，属使用而非分发，风险集中在"是否把 adapter 脚本与其打包分发"。验收内部使用=风险最低，**公开发榜数字=必须同时公开 AGPL 栈源码或放弃**。另外工程上要维护双环境（behavior-bench 依赖本仓 binding 版本），长期成本高于方案 D。
- **判定：仅在"需要与 behavior-bench 论文数字严格同口径对拍"时启用**，且默认不并入任何分发产物。

### 方案 D（推荐）：用本仓 cosim/nuplan 栈做官方闭环 harness
- **理由**：
  1. **已在仓内、MIT、上游验证过**（2026-09-03 val14 运行记录在案），零 AGPL 暴露；
  2. 双口径现成：nonreactive（log-replay 背景）+ reactive（IDM 背景），正是 val14 的两种官方挑战；
  3. 执行保真最强：ego 由 PufferDrive C 内核积分（`ACTION_TYPE` 语义可换成兄弟仓已落地的 `control` 模式，为 E1 的 Nova bridge 复用同一执行路径）；
  4. 升级路径自然：map-only bin 缺口可先用 `write_drive_bin`（无 lane graph，GPS goal 特征退化）起步，后换 123Drive 转换器补齐；
  5. 与 E1a/E1b 内部支共享基础设施（见 e1-dependencies.md），一份前置投入两处收益。
- **残留差异须在报告标注**：本仓栈经 CaRL `carl_nuplan` 预设运行（val14_split≈900 场景），与 tuplan_garage/interPlan 论文的 14 场景 val14 **不是同一 split**；ego 执行模型是 PufferDrive 积分器而非 nuPlan LQR。

## 4. 推荐

**主选 D，B 作内部回归粗口径，A 仅在 D 跑不通且确需 behavior-bench 同口径时降级启用，C 不推荐（license 灰区 + 双环境维护成本 > 收益）。**

前置动作顺序（解锁 D）：① 核 nuPlan 数据 license 并下载 .db+maps；② 生成/获取分城市 map-only bin（起步可用无 lane graph 版）；③ 本机用 `CKPT=dummy` 走通 run_nuplan_planner.sh 的 wiring 测试；④ 再换真 checkpoint 跑 val14_split。
