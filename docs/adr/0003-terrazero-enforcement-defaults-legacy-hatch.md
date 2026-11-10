# ADR-0003: 执法语义以 TerraZero 口径为默认，legacy 逃生门永续

Status: accepted 2026-11-09（K3 owner 验收 + K4a 切默认；一手依据 = TerraZero 报告 §3.3 + rule-enforcement-checklist）

红灯 = OBB 全身扫越停止线（footprint_sweep 默认 **on**）；停牌 = 采样驻停 dwell **[0.5,1.5]**（首版标定，TerraZero 分布未披露待再标定）；行人豁免路口违规（为 sim-agent 线 active 行人铺路）；truth_timing 默认 off（仅 ODD 保真场景，CARLA 无 states bin 上 no-op）。**每个语义都有 legacy 显式旋钮可逐位复现旧行为**（K4a 已验证）——历史锚点永远可复算。裁决链教训：11-03 checklist 的停牌 0.5s 条目测自兄弟仓旧线，本仓基线已被上游重写为 RunStopSign2 单 tick——外部核对文档必须标注其测量基线，过期即修（`72b51cb`）。K3 实证：footprint 开 → 红灯 latch 9.5×（擦边跨线被抓），dwell 采样致 RNG 流位移属设计内。
