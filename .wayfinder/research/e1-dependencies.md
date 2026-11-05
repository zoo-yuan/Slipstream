# E1a/E1b 内部依赖清单（票 09 内部支 · C）

> 定义（05 票）：**E1a** = ego=Nova（19 输入 / 4 输出契约，ADR-0006）、背景 GT 回放；**E1b** = 背景 IDM / RL（反应式混合背景）。
> 证据源：`../PufferDrive/docs/adr/0006`、`docs/adr/0007`、`docs/integration/2026-10-09-pufferdrive3-gap-analysis...md`（G1–G9）、`../PufferDrive/docs/STATE.md`（Next/Open rulings）、`git -C ../PufferDrive diff`（空）、nova 仓 `~/labs-x/agent_group/nova` 实地核验。
> **状态更正**：Slipstream STATE L15 称「`ACTION_TYPE_CONTROL` WIP 未提交」——已过时。`git -C ../PufferDrive diff` 与 `--cached` 均为空，该线已提交至 `dev/zoo_v3.0` HEAD `4875b53f`（"bridge acc wash-in in production code + Apollo lane matching"）。

## 0. 公共前置（E1a/E1b 共用）

### 已具备
| 项 | 位置 | 状态 |
|---|---|---|
| C 内核直连控制 `ACTION_TYPE_CONTROL` + `INTEGRATION_SCHEME_MODEL_EULER` + `skip_actuator_limits` | PufferDrive `constants.h/drive.h/drive.py`（`action_type="control"` → Box(2)） | ✅ 已落地，数值对拍通过（60 步/6 s，max\|Δxy\| 1.7e-5–3.4e-5 m，`test_control_mode_parity.py`） |
| Nova acc wash-in 桥（`apply_acc_wash_in`/`build_control_actions`，0.2/0.4/0.6/0.8 前 4 样本插值） | PufferDrive `pufferlib/ocean/drive/control_bridge.py` | ✅ 4875b53f；洗过 6 s 偏差 3.0e-5 m，不洗 0.39–0.92 m |
| 执行限幅标定（STEERING_LIMIT 0.60564、转角速率 0.42336、ACCEL {-6,3}、wheelbase 2.20、底盘上限 16.7） | PufferDrive `drive.h:4241/4384` + `docs/vehicle_profile.md` §3/§4 | ✅ |
| 观测包络（`agent_range_m=70`、`observation_delay_s=0.78` + `_apply_observation_delay` ring buffer） | PufferDrive（2026-11-05 落地，6 例对拍测试） | ✅ |
| Nova 契约一手定义 | nova 仓 `docs/dataset/数据规范V1.1.3_0911修正版.md` + nova ADR-0003/0004（本机可达） | ✅ |
| 18 指标 + 种子化失败回放 | Slipstream `constants.h`（COLLISION..MULTI_LANE_SCORE 18 项）+ `evaluation_utils/eval_replay.py` + `benchmark.yaml` | ✅ |
| per-agent 行为原语 `CONTROLLER_{STATIC,POLICY,REPLAY,IDM}`（E1b 原语） | Slipstream C 内核（idm.h 866 行，`move_idm` 含跟驰/route 投影/灯停判断） | ✅ 基线自带 |

### 缺失
1. **G8 观测桥本体**（Drive 世界态 → 19 路输入）：agents 16 帧历史、map 15 维、TL 关联、ego-centric RFU 重抽——两侧都未实现。PufferDrive STATE 仍列为 Next-P0「最小 Nova 契约 bridge，先在公开 WOMD 场景跑通」。
2. **G9 静态障碍四路**：scene 有 40 static objects（`map_data.h num_objects`），Python 侧完整暴露给 bridge 需确认；四路定长（points/types/heights/mask，≤32 polygon×32 点，cap 有损截断须上报丢弃数）未实现。
3. **G3 参考线**：route → 3 槽几何序（slot0=真实 ego 几何；三槽全零帧跳过推理）未实现。
4. **比例混合旋钮**（E1b）：`mix_traffic/ppo_fraction/idm_fraction/expert_fraction/idm_target_velocity/idm_random_velocity` 只在 behavior-bench（AGPL）——需只读重实现（C/Python 各一小块，基线已有 per-agent 原语，缺的只是编排层旋钮）。

### 阻塞项
1. **`wt-apollo/record_exporter/frame_builder.h` 本机不可达**：`ls ../PufferDrive/wt-apollo` 不存在；STATE/gap 分析引用的生产侧 14ch 重抽逻辑无实物可对照（外部机器路径）。M1 复用录制帧特征前必须先解决。
2. **本机 `pufferlib._C` 编不过**（g++ 9.4 无 C++20；torch intrusive_ptr 需 `<compare>`）——影响推理/训练栈；drive binding 可 gcc 直编不受影响。装 g++-11/13 前无法在本机跑完整 eval loop。
3. **nuPlan 数据集未下载**（若 E1 场景种子取自 nuPlan/WOMD 录制场景；程序化生成线不受此卡）。

## 1. E1a：ego=Nova、背景 GT 回放

### 已具备
- 背景回放栈：`SIMULATION_MODE_REPLAY` + `benchmark.yaml nuplan_multi/nuplan_single`（读预转 .bin、seed 42）+ 18 指标 + `eval_replay.py` 种子化失败回放——「背景 GT 回放」这半边是现成的（bin 需自转，本机 binaries/nuplan 仅 1 个 fixture）。
- ego 执行半边：`ACTION_TYPE_CONTROL` + wash-in 桥 + 限幅 = Nova 输出 (acc,κ) 60 帧@10Hz 进内核的全部执行语义（见 §0）。
- Nova ONNX 工具链：`export_onnx.py`（opset 17、GT-free 路径、受限常量折叠）、`dump_onnx_io.py`（golden 输入输出 JSON）、`check_onnx_vs_pytorch.py`（ORT CPU 一致性）、`gen_random_init_ckpt.py`（30.86M 参数随机初始化，确定性 seed）。
- 契约规格冻结基础：PufferDrive `spec/canonical_v0.yaml`（73 字段/8 组，三源审计可复跑）。

### 缺失
1. **§0 缺失 1–3（G8/G9/G3 桥三件套）**——E1a 的主体工作，估为该线最大单项。
2. **Nova 训练权重**：nova 仓全树 find 无 `.ckpt/.pt/.pth`（仅文档化随机初始化工具）。真实 Nova 不可得时只能用随机初始化 ckpt 占位跑通管线（结构等价、行为无意义）——验收语义须明确降级为"管线通了"，不是"模型强"。
3. **推理运行时接线**：ONNX Runtime 进 eval loop + 60 帧（6 s）输出与 Drive 步长（replay dt=0.1 = 10Hz 恰好整除；benchmark dt=0.0667=15Hz 需重采样口径）的 chunked 执行 + 桥侧 16 帧历史缓冲维护。
4. **E1a 场景种子集**：种子化场景清单（hazard demo M0–M3 已起步；若要 nuPlan/WOMD 录制帧则撞 §0 阻塞 3）。

### 阻塞项
- **Nova 权重获取**（外部依赖，本地无法解决；须向 owner 升级）。
- **数据湖/录包就位**：若按 ADR-0007「真正裁判必须跑在录制场景上」，6 条 Apollo bag + `base_map.bin` 在 PufferDrive 生态侧（3 条 bag 缺 bin，车道类约束不出判定——已在那边 STATE 记录）。

## 2. E1b：背景 IDM / RL

### 已具备
- IDM 背景：C 内核 `idm.h`（含 lateral/heading 限制、灯停响应）+ cosim 预设 `closed_loop_reactive_agents_pufferdrive`（IDM 反应式口径现成）；IDM 可行性裁判已真跑（constraints_v0：3 控制器 × 40 步 ≈ 2 s，18 可解 / 2 IDM 不足 / 0 不可解）。
- RL 背景：`SIMULATION_MODE_GIGAFLOW` + `CONTROLLER_POLICY`（自博弈全车共策）已在基线；05 票已立项零演示 self-play 配方；`9_bernhard_dev`（`09ca44c`）为权重/噪声挖矿源。
- 行为异常注入（对手鲁棒性轴）：`partner_blindness_*`/`phantom_braking_*` 基线已有。
- IDM 参数旋钮：`idm_target_velocity/idm_random_velocity` 语义已有参照（behavior-bench），基线 `idm.h` 支持参数化。

### 缺失
1. **比例混合 Σ 编排层**（§0 缺失 4）：背景 = x% replay + y% IDM + z% RL policy 的 fraction 旋钮与 registry（behavior-bench 思路只读重实现；PufferDrive STATE P1 已登记但未落地）。
2. **RL 对手 checkpoint**：`9_bernhard_dev` 的 val14 最强模型 + 噪声训练权重未 cherry-pick（git diff 未核，属 01 票范围）。
3. **E1b 下背景 agent 的 Nova 口径覆盖**：若 E1b 与 E1a 同世界（ego=Nova 观测桥开着），IDM/RL 背景的轨迹也要进 19 路输入的历史帧与 map/agent 编码——桥三件套须按"背景行为源无关"设计（这也反向要求 §0 缺失 1 的桥接口把 agent 来源抽象掉）。
4. **IDM comfort 自触发的口径核查**：constraints_v0 遗留「IDM 自身触发 comfort 阈值待查」——E1b 作为背景时需先解决，否则把背景物理误记为 ego 违规。

### 阻塞项
- **E1b 依赖 E1a 的桥**：两支共享 G8/G9/G3，桥不通则 E1b 只能退化为本仓 self-play（无 Nova）口径——这与「E1=Nova 闭环」的验收定义不符，须在票图上标 E1b blocked-by E1a 桥。

## 3. 汇总计数

| 支线 | 已具备 | 缺失 | 阻塞 |
|---|---|---|---|
| 公共 | 7 | 4 | 3 |
| E1a | 4 | 4 | 2（Nova 权重、录包/数据湖） |
| E1b | 4 | 4 | 1（blocked-by E1a 桥） |

**关键路径**：G8/G9/G3 桥三件套（E1a 主体）→ Nova 权重到位 → E1a 跑通 → Σ 混合层 → E1b。桥三件套可在无 Nova 权重期间用随机初始化 ckpt 先行开发对拍（golden 工具链齐全）。
