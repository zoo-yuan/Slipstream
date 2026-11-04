---
id: "03"
title: 异源 log→地图管线：TerraZero 的"任何日志当地图"如何落地
labels: [wayfinder:research]
status: closed
assignee: council-glm（子代理）
blocked-by: []
---
## Question

用户认定 TerraZero 段位差的关键之一是"能把其他输入的自动驾驶 log 应用起来"（log 降级为地图来源）。要回答：在 PufferDrive 上做等价能力，管线的真实形状是什么？

1. **输入需求反推**：`../PufferDrive` 的 `pufferlib/ocean/drive/map_data.h` + `load_map_binary()` 要求 .bin 提供哪些字段（lane 几何/边界/交通控制含配时/静态物/scenario_id/agents）？gigaflow 模式下 agents 轨迹是否可以缺席（即"纯地图 bin"是否成立）？
2. **交通控制配时来源**：log 降级为几何后，红绿灯配时是程序化生成（drive.h:2342 已有随机化）还是需要 log 真值？TerraZero 口径是什么（对照 02 票产出）？
3. **现有转换器盘点**：本仓 WOMD→bin 工具链在哪、覆盖什么；`origin/Carla_Road_Processing_Fixes` 分支的 Carla 路网处理脚本成色如何；nuPlan→bin 是否存在；`../PufferDrive/docs/integration/CONTEXT.md` 里 Apollo HD map→bin 的"最小转换"（仅几何+交通控制+route）规划到什么程度；`~/labs-x/agent_group/nova` 的数据规范能否充当 Apollo 侧输入契约。
4. **缺口清单**：对"任何 log（WOMD/nuPlan/Carla/Apollo/自有 rino）当地图"各自还缺什么，按工作量排序。

产出：`.wayfinder/research/log-to-map-findings.md`（含字段级表格与缺口排序）。

## Resolution

已解决（2026-11-03，council-glm 只读核查，成果见 [log-to-map-findings](../research/log-to-map-findings.md)）：

- **纯地图 bin（num_total_agents=0）在 gigaflow 下成立且是默认形态**；唯一禁忌 gigaflow+`goal_source="gt"`（读 NULL 崩）；bin 内 log agents 被 gigaflow 整体替换（顺带内存泄漏，可修）。
- **配时无需真值**：gigaflow 无条件程序化覆写（训练随机化/eval 10-3-2s），bin states 仅 replay 生效 → log 降级只要信号几何+受控 lane；若验收要真 ODD 配时，需加 states 保真开关（小改 drive.h:3009 块，已入地图雾区）。
- **「任何 log 当地图」= 一个几何转换器 + 一个 bin 重打包工具**（writer 模板 `data_utils/mirror_map_bin.py` 已在仓），五源共享；缺口排序：CARLA 脚本找回 < WOMD/nuPlan strip-agents（小时级）< rino 轨迹侧 < **Apollo HD map 几何转换器（G1-G3，最大且是 rino 前置）**。
- nova 规范 = 下游消费契约（非 Apollo 输入契约）；rino 已有 `apollo_record.py` 解码入口、无 writer。
- 未核项：`origin/Carla_Road_Processing_Fixes` diff（需一次 git diff，已归入 01 票证据）。
