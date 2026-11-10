# ADR-0002: 训练配方 = 零演示纯 RL 主轴 + 混合对手保留为配置能力

Status: accepted 2026-11-05（owner wayfinder「训练配方立场」票裁决）；progress 区间 2026-11-10 T2 扫描定案

主轴 = 纯 RL self-play（复现 TerraZero 始终如一，零 behavior clone）；replay/IDM/policy 混合对手**保留为必备旋钮但非默认**——owner 战略框架：「RL 是确定性路径，replay/IDM/policy 增大真实世界可部署 SOTA 的概率」。SPACeR 锚定不采纳但留门（轻量 sim-agent 线同源）。progress shaping 补 TerraZero 之短（其 reward 无进度项、InterPlan 进度分最低）：T2 三臂各 2.217e8 步定案 **C 臂 [2.5e-3, 2.5e-2] 最强**（goals 0.0101<0.0112<0.0118 单调，return 最优，代价红灯 0.0045→0.0060）→ 配方 v1 默认（`eed4b55`，[0,0] 逃生门保留流门逐位一致）。判读纪律：不调参刷曲线，1M 步不定案、2e8 步才定案。
