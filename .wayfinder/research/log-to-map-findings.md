# 异源 log→地图管线调研（票 03）

> Wayfinder 03 票调研产物。执行：council-glm 子代理（只读核查，无写文件工具，成果由会话代为落盘）。
> 日期 2026-11-03 · 证据全部来自 `../PufferDrive` 与 `~/labs-x/agent_group/nova` 只读核查。

## 1. `.bin` 必需字段（反推自 `pufferlib/ocean/drive/map_data.h:327-731` `load_map_binary()`）

**Header（定长 4×i32）**：`num_total_agents, num_roads, num_traffic, num_objects`。计数≤0 时对应数组保持 NULL，**允许缺席**。

| 段 | 字段（顺序即字节序，均 little-endian） | 必需性 |
|---|---|---|
| Agents ×N | `id i32`(=idx 强校验)、`type i32`(0/1车/2人/3骑)、`trajectory_size T i32`、9×`T×f32`(x,y,z,heading,vx,vy,len,wid,hgt)、`T×i32 valid`、`route_len i32`+lane序列、`route_gt_len i32`、`gt_goal x/y/z f32`、`mark_as_expert i32` | **gigaflow 下整段可缺席（N=0）** |
| Roads ×M | `id i32`(=idx 强校验)、`type i32`（十进制分类 `datatypes.h:5-31`：0-9 lane〔1 freeway/2 surface/3 bike/4 bus〕、10-19 line、20-29 edge、31 crosswalk/32 bump）、`segment_size S i32`、`x/y/z` 3×`S f32`、`headings S f32`；**仅 lane 型**追加：`num_entries`+数组、`num_exits`+数组、`speed_limit f32`、`length f32`、`cum_lengths S f32` | **必需**；`is_drivable_road_lane` 只认 type 1/2（`datatypes.h:9-11`），无之则 gigaflow 无法 spawn（`drive.h:2502-2510` 从 drivable 网格采样，`rng_below(0)` 会炸） |
| Traffic ×K | `id i32`(=idx 强校验)、`type i32`(1 light/2 stop/3 yield)、`stop_line` 6×f32（两端点）、`heading f32`、`state_size i32`+states 数组、`num_controlled_lanes`+lane 序列 | 几何+受控 lane **必需**；states 可空（gigaflow 兜底，见 §2） |
| Objects | 加载器 `fseek` 跳过但**字节必须在**（`map_data.h:624-640`）：每 obj `id/type/traj_len i32`+9×T f32+T i32 | 置 num_objects=0 即可 |
| Lane graph | `n_lanes_graph i32`（=0 可，int 本身必须有）、lane_ids、`n² f32` distances | 可选；仅 `obs_goal_lane_distance` 用（`drive.h:3941-3956`，NULL 安全） |
| 元数据 | `scenario_id char[128]`、`dataset_name char[32]`、`log_length i32`、`log_dt f32`、`num_objects_of_interest`+数组、`num_tracks_to_predict`+数组 | **必需**（定长读写，无哨兵）；计数可 0 |

Python 侧镜像读写器已存在：`data_utils/mirror_map_bin.py`（读写/镜像 bin，即格式的独立实现，可直接当 writer 模板）。

## 2. gigaflow 下"纯地图 bin"成立

- `set_active_agents()` gigaflow 分支 `calloc` 全新 agents 并 `spawn_agent` 从 drivable 网格采样位姿+随机游走 route（`drive.h:2459, 2752-2794`）；reset 时同样重 spawn（`drive.h:4518-4548`）。bin 里的 log agents 被**整体替换**（顺带是内存泄漏）。
- 默认配置即纯地图友好：`drive.py:65,88` `simulation_mode="gigaflow", goal_source="route"`。
- ⚠️ 陷阱：gigaflow + `goal_source="gt"` 会读 NULL log 轨迹崩溃（`drive.h:3064-3081` 无 gigaflow guard）——纯地图 bin 必须配 route/map goal。

## 3. 交通控制配时：bin 内置 states vs 程序化

- **gigaflow：完全程序化，bin states 无效**。`init()` 把 light 的 states 强制 resize 到 `scenario_length` 并调 `generate_traffic_light_states`（`drive.h:3009-3032`）；生成器**无条件覆写**：训练模式随机时长+20% 全关/常绿概率（`drive.h:2302-2405`），eval 用默认 10s 绿/3s 黄/2s 红（`constants.h:222-224`）。reset 再生（`drive.h:4511`）。
- **replay：bin states 即真值**，运行时按 `states[timestep]` 索引（`drive.h:1414,4147`、`idm.h:197`）。
- 内置配时现状：WOMD bin 带 lane_states（ScenarioMax 转换）；nuPlan bin（123Drive 0.3.2）部分有；CARLA opendrive bin 只有信号几何。→ "log 降级为地图"路线配时**只需几何（stop_line+controlled_lanes+heading），不需要真值配时**；要真值配时须走 replay 或改 gigaflow 让 states 保真（小改动：init 里加 skip 标志）。

## 4. 现有转换器盘点

| 源 | 现状 | 证据 |
|---|---|---|
| WOMD | ✅ 外部转换（ScenarioMax），bin 走 GCP/HF 分发，**仓内无代码** | `data_utils/download_database_gcp.sh:9`、`download_womd_data.py`、gap 文档 §3 |
| nuPlan | ✅ 外部转换（py123d+123Drive 0.3.2），S3 公开（全量 ~475GB/170k bins） | `data_utils/datasets.yaml:24-52` |
| CARLA | ⚠️ 8 个 opendrive Town bin 已入库 + 镜像工具；**转换脚本不在当前树**（2026-09-04 文档提过 `data_utils/carla/`，现已无）。`origin/Carla_Road_Processing_Fixes` 远端 ref 存在（tip `6f3dbcf0`，packed-refs:14），另有 `carla_dev/carla_data_utils/aditya/carla_big_maps/pragnay/z_coord_carla` 等 5 条相关分支；**diff 内容未核查（无 git 工具）** | `binaries/carla/`、`mirror_map_bin.py`、`docs/2026-09-04…md:53` |
| Apollo HD map | ❌ 无代码。规划级共识 = 仅"几何/可行驶/交通控制/route"最小转换，语义走 sidecar；优先级 P1 | `docs/integration/CONTEXT.md` Relationships、gap 文档 G1-G3+§4 |
| rino 自有 | ⚠️ 输入侧已起步：`data_utils/apollo_record.py`（cyber record 解析+localization/perception/planning/control 解码，"bag→scene 管线入口"）+ 车辆参数门禁；**无 bin writer** | `apollo_record.py:1-35`、`docs/vehicle_profile.md:264` |
| nova 仓 | 只读可访问。其 DataFeature 是 **ego-centric RFU 逐帧裁剪**（[256 lane×64seg 等]，`data_feature.proto:175-186`），非全局地图、无 lane 拓扑；raw `apollo.hdmap.Map` 在 wt-apollo（不在 nova）。**结论：nova 规范可当下游消费契约（sidecar/19入4出/96帧窗口），不能当 Apollo→bin 的输入契约**；但 `数据规范V1.1.3` 对 map_lane.proto/overlap 灯-车道绑定的描述可当转换器的领域参考 | nova `data_feature.proto`、`docs/dataset/数据规范V1.1.3`、`adfm/simulation/map_adapter.py`（仅合成单 lane） |

## 5. 五类"log 当地图"缺口排序（工作量小→大）

1. **CARLA/类 OpenDRIVE**：差一个从路网到 bin 的转换脚本找回/重写（历史分支已有先例）＋复核 Carla_Road_Processing_Fixes diff。最小。
2. **WOMD**：管线已通（外部转换+现成 bin）；"当地图"只差**去 agents 化工具**（可基于 `mirror_map_bin.py` 的读写器写一个 strip-agents/重打包脚本，小时级）。
3. **nuPlan**：同 WOMD（bin 已有、license CC BY-NC-SA）；同样只差 strip/选图工具。
4. **rino log**：有 record 解码与车辆真值源，缺 (a) Apollo HD map→bin 几何转换器（与第 5 项共享）(b) 感知轨迹→agents 段 + scenario_id/provenance 编码。中等偏大，但 G1 转换器复用后主要剩轨迹侧。
5. **Apollo HD map**：最大缺口=G1/G2/G3 转换器本体（Lane/Boundary/Signal/Overlap→road/traffic elements、坐标投影、routing→route）；无仓内代码，上游 proto 在 wt-apollo。是第 4 项的前置共享件。

**关键裁决输入**：纯地图 bin（0 agents）技术上成立且是 gigaflow 默认形态；交通配时无真值需求（程序化兜底已在内核）。真正要新建的只有"一个几何转换器 + 一个 bin 重打包工具"，五类源共享。

## 执行者附加立场（council-glm）

- 此事可大幅简化——不要为五类源建五条管线；建"源→中间几何表示→单 writer"一层即可，writer 直接复制 `mirror_map_bin.py` 的格式实现。
- 最大反方论点：gigaflow 程序化配时与 log 真值分布差异可能伤及"自有 ODD 逼真度"，若验收口径要求真实配时则需加 states 保真开关（小改 `drive.h:3009` 块）。
- 置信度：高（除 Carla 分支 diff 未核，需一次 `git diff` 收尾）。
