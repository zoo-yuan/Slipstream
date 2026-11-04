# AGENTS.md — Slipstream

## 仓库定位

Slipstream = **TerraZero 复刻主仓**。基线 = 上游 PufferDrive `3.0`（`2c1ca3e4`，tag `baseline/pufferdrive-3.0-2c1ca3e4`），保留完整上游历史。

- `upstream` remote = `Emerge-Lab/PufferDrive`，**只读**：只 fetch/merge 跟进上游，**永不 push**。
- `origin` remote = `zoo-yuan/Slipstream`（本仓远端）。
- 兄弟仓 `../PufferDrive` = Nova bridge 线（`ACTION_TYPE_CONTROL` WIP）+ strata 文档语境；本仓对其**只读引用**，不在那边提交复刻代码。`origin/9_bernhard_dev` = 挖矿源（噪声块/权重按需 cherry-pick，见 wayfinder 地图「基底裁决」票）。

## 放置规则

- `.wayfinder/` — TerraZero 复刻 effort 的 wayfinder 地图与票（约定见 `.wayfinder/TRACKER.md`；调研产物在 `.wayfinder/research/`）。
- `docs/` — strata 层：`docs/STATE.md` 为 grounding 真值（as-of 指针须与最新 commit 同步）；日期前缀文档 = 冻结叙事，新写入须在 STATE.md 台账登记。
- 仿真内核改动（`pufferlib/ocean/drive/`）遵循上游 PufferDrive 信任边界：外部配置 init 前校验；热路径（c_step/obs/collision）零 malloc。

## 环境

- 本机网络走代理 `127.0.0.1:7980`；git 已配 repo-local `http.version=HTTP/1.1`（HTTP/2 过代理握手失败）。大包传输（fetch/push 全历史）可能超时，优先从本地兄弟仓 fetch。
- 工作语言：中文。
