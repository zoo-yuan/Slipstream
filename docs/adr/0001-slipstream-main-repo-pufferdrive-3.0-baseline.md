# ADR-0001: Slipstream 为主仓，基线 = PufferDrive 3.0，从零实现排除

Status: accepted 2026-11-03（owner 结构化问卷 + wayfinder「基底裁决」票一手裁决）

Slipstream（空壳新仓）承接 TerraZero 复刻主线；基线 = 上游 Emerge-Lab `3.0@2c1ca3e4` **保留全历史**导入（tag `baseline/pufferdrive-3.0-2c1ca3e4`），后续上游 bugfix 可 git merge 跟进。从零自研被排除：4704 行同血统已验证 C 内核重写零增量（11-03 分析 §7），owner 认同风险确定性优先。`origin/9_bernhard_dev` = 挖矿源（pose/brownian 噪声、val14 90 CLS-R 最强模型），只 cherry-pick 不并基线（+6.3 万行实验产物）。兄弟仓 `../PufferDrive` 保留 Nova bridge 线，本仓对它只读。理由不可从代码推导：这是 owner 的风险偏好裁决。
