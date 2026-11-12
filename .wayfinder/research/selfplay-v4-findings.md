# 自博弈 v4 侦察：naive self-play 的真实缺口（票：量级提案②）

日期：2026-11-12 夜 · 触发：`docs/2026-11-12-量级改进路径提案.md` 路径②待 owner 裁决
证据源：冠军训练 config 实地核验 + `pufferlib/ocean/drive/drive.py` + `pufferlib/config/puffer_drive.yaml` + `docs/2026-11-03-Terra系列全景与PufferDrive落地路线.md` §5

## 1. 提案②表述修正（重要）

提案②原文「把 IDM 伙伴替换/混入训练快照群体」**表述有误**。实地核验：

- `artifacts/t3-R4x/ckpt/R_ent5_off23/config.yaml`：`non_sdc_controller: policy`（**不是 IDM**）
- 上游默认（yaml L182）：`non_sdc_controller: policy` —— gigaflow 模式**本来就是同策略 self-play**
- `drive.py` controller_values：`static/policy/replay/idm` 四选一，内核原生支持 policy 伙伴
- IDM 只在 cosim/eval 语境出现（`non_sdc_controller_str == "idm"` → non-vehicle 用 replay 的 auto 规则、nuplan reactive agents）

**因此 v4 的起点不是「替换 IDM」，而是「打破同策略群体的同质化」。**

## 2. 真实缺口：naive 同策略 self-play 的塌缩面

当前训练 = 单策略 π 同时驱动全部 128 agents/env（+44 cosim 静态槽做 hazard 面）。已知风险（Terra 落地文档 §5 #3 在案）：

- 对手分布 = 自己 → 博弈压力不足，群体趋向「互相礼让」的共谋均衡（保守吸引子族，ADR-0006/0007 的病根之一）
- 策略迭代时对手分布同步漂移 → 非平稳（无固定对手混合可 exploited）
- **Terra 落地文档已裁决方向**：「采纳 SPACeR 第三路线：reference-anchored self-play，替代 fraction / policy-pool 之争」——伙伴锚定冻结参考模型，防塌缩且免 policy-pool 管理负担

## 3. 工程量核验（1-2 天估计成立，但落点在 Python 层）

| 改造点 | 层 | 内容 | 规模 |
|---|---|---|---|
| 多策略 batch 前向 | `pufferl.py` 训练循环 | learner π 前向 ego 子集；池内快照 π₁..πₖ 各前向伙伴子集（no-grad） | 核心，~半天 |
| 快照池 | 训练循环外围 | epoch 间隔存 flat state_dict（已有 best_models 机制可复用）；采样分配策略→agent 槽位 | ~2h |
| 参考锚定（SPACeR） | 同上 | 池中固定一个冻结参考（如当前冠军 ckpt），伙伴按 p_ref/p_learner/p_snapshots 混合 | 配置项 |
| 内核 | **无需改动** | C 内核只收 action 数组（CONTROLLER_POLICY 语义不变）；热路径零 malloc 边界不动 | 0 ✓ |
| 场景分布共进化 | 独立轴 | 属「更难任务」分支，与伙伴轴解耦，v4 不混 | — |

风险：多策略前向切碎 batch → GPU 利用率降（估 SPS -20~40%）；缓解 = 伙伴子集按快照分组聚合前向（每个快照一个大 batch）。

## 4. 验收设计（v4 立项时的双门）

1. **对手门**：v4 策略 vs v3 冠军 ckpt 对打（ego=v4，伙伴池=v3）——净 return ≥ v3 对 v3
2. **原面门**：carla_fast/mix 双面 eval 不劣于 v3 冠军 -5%（防「只会对付活人、不会开标准面」）
3. 上限探针：hazard mix 面碰撞率应显著降（博弈压力真实化后保守-激进谱系拉开）

## 5. 给 owner 的决策修正

提案②的机理判断（「策略在 IDM 对手分布上过拟合」）应更正为：「策略在**自身镜像**分布上过拟合」。这不削弱路径②的价值判断——同质化缺口比 IDM 缺口更根本，reference-anchored population 是唯一相变候选轴的结论不变，且工程量核验后仍为 1-2 天。
