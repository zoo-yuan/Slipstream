---
id: "01"
title: 基底裁决：复用 PufferDrive 3.0 家族还是别路，Slipstream 如何承接
labels: [wayfinder:grilling]
status: closed
assignee: zoo（已裁决）
blocked-by: []
---
## Question

复刻工作的代码基底与承接方式选哪条？选项（证据见地图 Notes 的事实基座）：

- **A. Slipstream 为主仓（用户倾向方向）**：把 PufferDrive 3.0 家族作为基线导入 Slipstream 并提交（squash import / vendored copy / submodule 三选一），复刻演进全在 Slipstream；`../PufferDrive` 只读跟随上游、保留 Nova bridge 线。
- **B. PufferDrive 仓内演进**：继续在 `dev/zoo_v3.0` 干，Slipstream 仅沙盒/笔记。
- **C. 双仓分工**：内核改动（hazard spawner 等）回 PufferDrive 仓；生成层/集成/实验在 Slipstream。
- **D. 从零自研内核**：11-03 文档 §7 已给排除论证（4704 行已验证同血统 C 内核，重写零增量）；选此需明确推翻理由。

裁决需一并回答：

1. 基线分支选哪个：`3.0`（干净上游）vs `dev/zoo_v3.0`（=3.0+docs）vs `origin/9_bernhard_dev`（+pose/brownian 噪声，5280 行，val14 90 CLS-R 最强模型所在；但偏离上游）。
2. `../PufferDrive` 工作树里**未提交的 `ACTION_TYPE_CONTROL` Nova bridge WIP**（253 行 + 测试）归属哪条线、由谁收尾提交。
3. 若选 A：导入方式（保留上游历史便于 rebase 跟进，还是 squash 干净起步）。

**补充证据（04 票调研，待本票裁决时执行）**：`git -C ../PufferDrive diff 3.0..origin/9_bernhard_dev --stat` 摘资产清单（文档佐证 pose/brownian 噪声+nuPlan 10Hz cosim+val14 最强模型，diff 未核）；另复核 `origin/Carla_Road_Processing_Fixes`（03 票未核项）。

## Resolution

已裁决（zoo，2026-11-03；证据：上述 diff 已核——197 文件 +62796 行，大头为实验产物 episode_metrics.csv 40001 行 / final_model.pt 30MB / parquet，代码改动与 rerun 交织）：

1. **A：Slipstream 为主仓**。PufferDrive 仓只读跟随上游 + 保留 Nova bridge 线（`ACTION_TYPE_CONTROL` WIP 归那边收尾提交）。
2. **基线 = `3.0` 干净上游**（80a681b4）。`9_bernhard_dev` 定位为**挖矿源**：噪声块/最强模型权重按需 cherry-pick（与 PufferDrive 仓 STATE.md 现行 P0 一致），不进基线历史。
3. **导入 = 保留历史**：3.0 的 commit 历史直接作 Slipstream master 起点，后续 Emerge-Lab 3.0 线 bugfix 可 git merge 跟进。
4. 文档（Terra 系列分析/ADR/strata）留在 PufferDrive 仓，Slipstream 按需引用，不随基线导入。
5. 执行落地 = 「基线导入执行」票（新建，见 tickets/11）。
