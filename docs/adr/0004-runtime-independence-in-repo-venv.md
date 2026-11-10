# ADR-0004: 运行时零依赖兄弟仓——本仓自持 .venv 与地图 bin

Status: accepted 2026-11-10（owner 指出命令行仍挂 ../PufferDrive 伤信任，当场裁决并落地 `1901519`）

Slipstream 自持 `.venv/`（torch 2.14.1+cu126，从兄弟仓 venv 整体拷贝，独立性断言验证）与 `pufferlib/resources/drive/binaries/carla/`；一切脚本/文档/AGENTS 的运行时引用清除。`../PufferDrive` 仅存两种合法身份：git 上游/挖矿源（只读）与 strata 文档语境。历史原因：基线导入时 torch 2GB 下载过不了代理而借用兄弟 venv，属 bootstrap 捷径，欠债两周被 owner 抓出——教训：**bootstrap 借用的外部依赖必须显式记债并限期本地化**，否则伤信任。附带陷阱记录：venv python 是指向 conda 的符号链接，`Path.resolve()` 会剥离 venv 上下文（gymnasium 消失），python_bin 类函数必须原样返回符号路径。
