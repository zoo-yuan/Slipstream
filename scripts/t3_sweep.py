#!/usr/bin/env python3
"""T2-v2 两臂短扫：mix 面重标配方扫（2 臂 × ~5e7 步）+ 臂毕自动 eval 定 v2。

v2 sweep runner (owner 批准的三臂规格，经自测惰性键发现后 supervisor 裁决收缩为两臂：
3 臂 × 5e7 步 → 2 臂 × 5e7 步，扫完自动 eval 定 v2；offroad 权重方向 pending owner)。
引擎复用 T3 ignition 的 machinery（scripts/t2_haztrain.py fork）：面 FACES 钉 8x128
（probe 只测 SPS + peak VRAM 稳定性，全部在 hazard mix 面上测，sizing 用 mix SPS）->
honest sizing -> 两串行臂 -> 收割。mix 面（hazard 四开 + cosim_partner_slots 提到
hazard_slot_capacity）不是臂维度；臂维度 = 奖励重标。训练奖励现值：offroad 1.5 /
collision 1.5 / lane_center 0.001 / progress v1 [2.5e-3, 2.5e-2]；eval 基准 offroad 3.0。
根因假设 = 借出路换机动性（offroad 换 progress）：

  A_v1ctl:     {}（空覆盖）= 纯 v1 区间 mix 同视界对照（补 T3 v1 缺 eval 的缺口）
  B_proghalf:  reward_progress_lo/hi 减半 [1.25e-3, 1.25e-2]  # progress 减半

progress 采样独立于 reward_randomization（T1 特例，真实生效）；reward_randomization=true
下 env.reward_offroad 训练惰性（drive.h generate_reward_coefs 从静态 REWARD_BOUNDS 采样），
故原规格的 offroad 臂（A_offroad3/C_both）已删，offroad 覆盖机制保留泛化，owner 批内核
offroad 区间旋钮后追加第三臂即刻生效（调研备查：.scratch/workers/task5_v2sweep.md）。未
覆盖键沿用 config 默认 v1 区间（kernel 校验 finite + lo <= hi）。对照 = A 臂即 v1 同视界
（另可对照 T3 v1 曲线 artifacts/t2-haztrain/metrics.csv）。

臂毕自动 eval（本 runner 的核心增量）：每臂训练 done 后，串行 subprocess 调
scripts/t3_eval_ckpt.py 两次——carla_fast 干净面与 --mix-face 各一次（默认
--eval-scenarios 200 场景），--out artifacts/t3-sweep/eval-{clean,mix}-<arm>。eval
失败不炸主流程：醒目警告 + 该臂该面 eval 记 failed，训练数据照常收割。收割器把各臂
两面的 comparison（episode_return / offroad_rate / collision_rate / num_goals_reached /
dnf_rate / making_progress_rate 六指标）并排进 summary.json 顶层 eval 块与 index.html
eval 对照表。

Kernel continuity gate (t2_hazsmoke discipline): the drive binding sha256 must match the
T2 main-train anchor (artifacts/t2-maintrain/summary.json); on mismatch the run refuses
to start unless --allow-kernel-drift is passed (owner opt-in). The gate result lands in
state.json and, after harvest, in summary.json's top-level t2_binding_sha256_match.

Engine: face probe -> honest sizing (steps floored to epoch boundaries) -> two serial
arms, each with per-epoch CSV appends + ~10%-progress checkpoints (pufferl auto-resume
via trainer_state.pt) + auto-eval on done. Expected wall ~1.3h/arm at the default
budgets (~5e7 步/臂 @47K SPS，eval 墙钟另计). All run state lives in state.json so a
relaunch skips completed work: a crashed arm resumes from its checkpoint (or restarts
cleanly if it died before the first checkpoint), a dead arm is recorded, completed
eval faces are skipped. The orchestrator is meant to be launched detached (setsid +
nohup) and survives the launching session.

Pipeline reuse note: worker subprocess + python_bin() (venv-symlink trap: return the
interpreter AS-IS, no .resolve()) all follow scripts/t2_sweep.py / scripts/k4b_smoke.py.

Repro (the exact detached launch for the real run):
  cd <repo root>
  setsid nohup env PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/t3_sweep.py --seed 42 --arm-budget-s 6300 --total-budget-s 9000 </dev/null >> artifacts/t3-sweep-console.log 2>&1 &

Progress view:
  tail -f artifacts/t3-sweep/orchestrator.log
  tail -2 artifacts/t3-sweep/ckpt/B_proghalf/metrics.csv

Harvest (safe to run mid-run, uses whatever rows/evals exist):
  PYTHONPATH=$PWD .venv/bin/python scripts/t3_sweep.py --collect-only
  -> artifacts/t3-sweep/{metrics.csv(合并), curves.png, summary.json, index.html};
  summary.json 顶层 eval 块原样嵌入各臂 eval-{clean,mix}-<arm>/summary.json 的
  comparison；index.html 每次收割由活数据重建（「本页由收割器重建」）。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts" / "t3-sweep"
# 臂毕自动 eval 的入口脚本（独立评测 CLI；--out 产物落在 ART_DIR 下 eval-<face>-<arm>/）。
EVAL_SCRIPT = REPO_ROOT / "scripts" / "t3_eval_ckpt.py"
EVAL_BENCHMARK = "carla_fast"
EVAL_SCENARIOS_DEFAULT = 200
# 内核连续性门锚点（t2_hazsmoke 同款）：仍指向 T2 主训的 summary，不随本脚本产物目录变。
T2_ANCHOR_SUMMARY = REPO_ROOT / "artifacts" / "t2-maintrain" / "summary.json"
CKPT_DIR = ART_DIR / "ckpt"
PROBE_DIR = ART_DIR / "probe"
STATE_PATH = ART_DIR / "state.json"
DECISION_PATH = ART_DIR / "decision.json"
ORCH_LOG = ART_DIR / "orchestrator.log"

# Page repro block (render_index): the docstring's detached launch + harvest command,
# one line each, so index.html and this file stay in sync.
DETACHED_LAUNCH = (
    "setsid nohup env PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    ".venv/bin/python scripts/t3_sweep.py --seed 42 --arm-budget-s 6300 --total-budget-s 9000 "
    "</dev/null >> artifacts/t3-sweep-console.log 2>&1 &"
)
HARVEST_CMD = "PYTHONPATH=$PWD .venv/bin/python scripts/t3_sweep.py --collect-only"

# Ordered smallest -> largest; probing stops at the first unstable face (OOM risk only
# grows with face size, so larger faces after a failure are classified unstable-by-risk).
# The v2 sweep holds the T2 main-train face (8x128) fixed for reward-recipe
# comparability; larger faces are out of scope for this run (probe-stable != hour-stable
# under fragmentation).
FACES = [(8, 128)]

# Minimum epochs an arm must be able to sustain at the arm budget (size_arms guard);
# choose_face uses the same bar when picking the face (fallback to a smaller stable face
# if the largest one cannot sustain it -- only fires on tiny smoke budgets, never at
# the ignition budgets).
MIN_ARM_EPOCHS = 10

# v2 全量（单臂 = 探针胜者 C 配方：proghalf + offroad 钉 [3,3]，预宣判据 eval return 主指标；
# D 膝点探臂双指标全败 2026-11-11，详见 artifacts/t2-v2sweep/）。下述历史注释保留备查。
# v2 两臂（supervisor 裁决 2026-11-10，原三臂规格收缩）：mix 面不是臂维度——probe 与正式臂
# 都带 hazard-on（sizing 必须用 mix SPS）。臂维度 = 奖励重标（训练奖励现值 offroad 1.5 /
# progress v1 [2.5e-3, 2.5e-2]；eval 基准 offroad 3.0）。
# ⚠ 惰性键发现（2026-11-10 自测，已升级 owner 裁决中）：reward_randomization=true 下
#   env.reward_offroad 训练中不被读（drive.h generate_reward_coefs 随机化分支从静态
#   REWARD_BOUNDS U(0,3) 逐 agent 采样 REWARD_COEF_OFFROAD，env 键只在 randomization=false
#   分支生效），故原规格 A_offroad3/C_both 的 offroad 覆盖是 no-op（自测 B≡C 位级全同），
#   已从 ARMS 删除。offroad 覆盖机制保留泛化——owner 批内核 offroad 区间旋钮后追加第三臂
#   即刻生效（调研备查：.scratch/workers/task5_v2sweep.md）。
#   A_v1ctl    = 空覆盖（config 默认 v1），纯 v1 mix 同视界对照，补 T3 v1 缺 eval 的缺口；
#   B_proghalf = progress 区间减半（T1 特例：progress 采样独立于 reward_randomization，真实生效）。
# 未覆盖键沿用 config 默认（kernel 校验 finite + lo <= hi）。对照 = A 臂即 v1 同视界
#（另可对照 T3 v1 曲线 artifacts/t2-haztrain/metrics.csv）。
ARMS: dict[str, dict[str, float]] = {
    # v3 熵控探针（ADR-0006 病灶=熵衰减→保守吸引子）：基座 = B 配方（proghalf，无安全钉价），
    # 臂维度 = train.ent_coef 倍率（默认 0.01）；全量直跑（探针读瞬态，禁外推），步数对齐 B@2.43e8。
    # 键约定："train." 前缀 → cfg["train"]，其余 → cfg["env"]。
    "F_ent2": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.02},
    "G_ent3": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.03},
    # H = 形状变体：0.02 起步、末 25% 退火到 0.005（高探索起步 + 软暮年下限，对抗常数高熵的晚期噪声）
    "H_ent_anneal": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2,
                      "train.ent_coef": 0.02, "train.ent_coef_anneal": True, "train.ent_coef_final": 0.005},
    # 轮 2 保险臂（在轮 1 结果前即有价值）：I = 速度侧直接杠杆（训练 velocity 奖 3×，
    # 对准 owner 贴限速需求）；J = 熵剂量上界（×4，与 G×3 括出剂量响应）
    "I_vel3": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "reward_velocity": 0.15},
    "J_ent4": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.04},
    # K = velocity 剂量上探（6×）：与 I(3×)、B 基线(1×) 括出速度侧剂量响应
    "K_vel6": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "reward_velocity": 0.30},
    # 轮 3 组合假设：高熵破瘫（G 已证 dnf 0.75→0.06）+ T6/T7 安全钉价剪风险（低熵下致瘫的 E/D
    # 结论是在已瘫动力学下取得，高熵下重验）；N = 退火到更高下限（括 G 与 H 之间）
    "L_ent3_off23": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.03,
                      "reward_offroad_lo": 2.0, "reward_offroad_hi": 3.0},
    "M_ent3_coll23": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.03,
                      "reward_collision_lo": 2.0, "reward_collision_hi": 3.0},
    "N_ent3_anneal_hi": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2,
                          "train.ent_coef": 0.03, "train.ent_coef_anneal": True, "train.ent_coef_final": 0.01},
    # 轮 4：L 已证「高熵+offroad 钉价」= 安全+效率双拿（return 0.993/碰撞 0.0099）。
    # P = J 剂量(×4)+L 安全剪；Q = 双钉价（offroad+collision 同时剪两风险通道）
    "P_ent4_off23": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.04,
                      "reward_offroad_lo": 2.0, "reward_offroad_hi": 3.0},
    "Q_ent3_both23": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.03,
                       "reward_offroad_lo": 2.0, "reward_offroad_hi": 3.0,
                       "reward_collision_lo": 2.0, "reward_collision_hi": 3.0},
    # 轮 5：P(ent×4+off[2,3]) 新冠军 1.163/0.865 全指标支配 L。R = 剂量上探 ×5；
    # P@seed43 复现走同配方不同 seed（GPU1，冠军可信度门）
    "R_ent5_off23": {"reward_progress_lo": 1.25e-3, "reward_progress_hi": 1.25e-2, "train.ent_coef": 0.05,
                      "reward_offroad_lo": 2.0, "reward_offroad_hi": 3.0},
}
ARM_COLORS = {"F_ent2": "tab:cyan", "G_ent3": "tab:purple", "H_ent_anneal": "tab:olive", "I_vel3": "tab:red", "J_ent4": "tab:brown", "K_vel6": "tab:pink", "L_ent3_off23": "tab:orange", "M_ent3_coll23": "tab:blue", "N_ent3_anneal_hi": "tab:gray", "P_ent4_off23": "tab:green", "Q_ent3_both23": "tab:red", "R_ent5_off23": "tab:brown"}

# progress 未覆盖臂沿用的 config 默认 v1 区间（puffer_drive.yaml 现值，逐字喂 env）。
PROGRESS_LO_DEFAULT = 2.5e-3
PROGRESS_HI_DEFAULT = 2.5e-2

INTERPRETER_CANDIDATES = [
    REPO_ROOT / ".venv" / "bin" / "python",  # 本仓自行 venv（T2a 后本地化，不再依赖兄弟仓）
    REPO_ROOT.parent / "PufferDrive" / ".venv" / "bin" / "python",  # 仅应急回退，正常不应触达
    Path(sys.executable),
]
# Guard so scripts/ imports (hazard_demo provenance helpers) resolve regardless of cwd.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------- utilities --


def log(msg: str) -> None:
    """Stamp progress to stdout AND the orchestrator log (every line hits disk, so a
    detached orchestrator is readable line-by-line while it runs)."""
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


def python_bin() -> Path:
    """Interpreter that owns torch; the sibling venv is borrowed for Python only.
    NOTE: return the candidate AS-IS (no .resolve()): the venv python is a symlink
    into miniconda, and resolving it strips the venv context (site-packages/pyvenv.cfg
    association follows the invoked path, not the physical binary). Same fix as k4b."""
    for cand in INTERPRETER_CANDIDATES:
        if cand.exists():
            return cand
    raise RuntimeError(f"no usable interpreter among {INTERPRETER_CANDIDATES}")


def verify_in_repo_kernel() -> dict:
    """Kernel pinning discipline (same rule as K-series/T1): pufferlib and the drive
    binding .so must resolve to THIS repo; provenance block goes into every summary."""
    sys.path.insert(0, str(REPO_ROOT))
    import hazard_demo as hd  # repo scripts dir; provides verify/provenance helpers

    hd.verify_in_repo_kernel()
    prov = hd.kernel_provenance()
    c_so = REPO_ROOT / "pufferlib" / "_C.cpython-311-x86_64-linux-gnu.so"
    if not c_so.exists():
        raise RuntimeError(
            f"in-repo advantage kernel required: {c_so} missing; build with "
            "NO_OCEAN=1 CC=<gcc13> CXX=<g++13> python setup.py build_torch --inplace --force"
        )
    import hashlib

    import torch

    prov["advantage_so"] = str(c_so)
    prov["advantage_sha256"] = hashlib.sha256(c_so.read_bytes()).hexdigest()
    prov["torch"] = torch.__version__
    prov["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return prov


def gate_t2_kernel_binding(provenance: dict, allow_drift: bool) -> bool:
    """内核连续性门（「锚点无回归」的实现，原样拷自 t2_hazsmoke）：本长训的 drive binding
    sha256 必须与 T2 主训锚点（artifacts/t2-maintrain/summary.json）一致。不一致 → 醒目警告 +
    state/summary 记 t2_binding_sha256_match=false，且默认拒跑（对齐仓库内核钉死纪律：
    非预期内核不得静默运行）；仅 --allow-kernel-drift（owner 明示）可放行。"""
    if not T2_ANCHOR_SUMMARY.exists():
        raise RuntimeError(f"T2 主训锚点缺失: {T2_ANCHOR_SUMMARY} 不存在，无法比对内核连续性")
    anchor_sha = json.loads(T2_ANCHOR_SUMMARY.read_text())["kernel"]["binding_sha256"]
    local_sha = provenance["binding_sha256"]
    if local_sha != anchor_sha:
        log(
            f"!!! [KERNEL DRIFT] 本仓 binding sha256={local_sha[:12]}… != T2 主训锚点 "
            f"{anchor_sha[:12]}…（锚点文件: {T2_ANCHOR_SUMMARY}）"
        )
        if not allow_drift:
            raise RuntimeError(
                f"kernel drift vs T2 maintrain anchor: local={local_sha[:12]}… "
                f"anchor={anchor_sha[:12]}…; 非预期内核不得静默运行 — 如 owner 明示放行，请加 --allow-kernel-drift"
            )
        log("!!! [KERNEL DRIFT] --allow-kernel-drift 明示放行：继续运行，但结果与 T2 主训不可直接比读")
        return False
    log(f"kernel continuity gate OK: binding sha256 与 T2 主训锚点一致 ({local_sha[:12]}…)")
    return True


def hazard_slot_capacity(env_cfg: dict) -> int:
    """Mirror of Drive.hazard_slot_capacity（原样拷自 t2_hazsmoke，勿跨脚本 import）:
    planned-count highs of the hazard face, computed from the *final* env config so the
    slot budget always fits the knobs."""
    if not env_cfg["hazard_enable"]:
        return 0
    cap = 0
    if env_cfg["hazard_parked_enable"]:
        cap += env_cfg["hazard_parked_n_hi"]
    if env_cfg["hazard_clusters_enable"]:
        cap += 2 * env_cfg["hazard_clusters_n_hi"]
    if env_cfg["hazard_cones_enable"]:
        cap += env_cfg["hazard_cones_zones_hi"] * max(
            env_cfg["hazard_cones_taper_cones"],
            env_cfg["hazard_cones_block_rows"] * env_cfg["hazard_cones_block_cols"],
        )
    if env_cfg["hazard_walkers_enable"]:
        cap += env_cfg["hazard_walkers_n_hi"]
    return cap


def save_state(state: dict) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    os.rename(tmp, STATE_PATH)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log(f"state.json unreadable ({exc}); starting fresh state (completed-arm "
                "detection falls back to metrics.csv/trainer_state.pt checks)")
    return {}


def repair_csv_tail(path: Path) -> bool:
    """Drop a trailing partial row (crash mid-write). Checkpointed rows before it stay."""
    if not path.exists():
        return False
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        idx = data.rfind(b"\n")
        path.write_bytes(data[: idx + 1] if idx >= 0 else b"")
        return True
    return False


# ------------------------------------------------------------- training CSV ---

# Fixed CSV columns (subset of the epoch logs dict; full dict goes to JSONL sidecar).
CSV_COLUMNS = [
    "step",
    "epoch",
    "sps",
    "uptime_s",
    "learning_rate",
    "episode_return",
    "score",
    "collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "stop_sign_violation_rate",
    "episode_length",
    "episodes_completed",
    "num_goals_reached",
    "dnf_rate",
    "avg_speed_per_agent",
    "reward_progress_component",
    "policy_loss",
    "value_loss",
    "entropy",
]

# The full logs dict uses environment/ and losses/ prefixes; map CSV column -> key.
COLUMN_KEYS = {
    "step": "agent_steps",
    "epoch": "epoch",
    "sps": "SPS",
    "uptime_s": "uptime",
    "learning_rate": "learning_rate",
    "episode_return": "environment/episode_return",
    "score": "environment/score",
    "collision_rate": "environment/collision_rate",
    "offroad_rate": "environment/offroad_rate",
    "red_light_violation_rate": "environment/red_light_violation_rate",
    "stop_sign_violation_rate": "environment/stop_sign_violation_rate",
    "episode_length": "environment/episode_length",
    "episodes_completed": "environment/n",
    "num_goals_reached": "environment/num_goals_reached",
    "dnf_rate": "environment/dnf_rate",
    "avg_speed_per_agent": "environment/avg_speed_per_agent",
    "reward_progress_component": "environment/reward_components/progress",
    "policy_loss": "losses/policy_loss",
    "value_loss": "losses/value_loss",
    "entropy": "losses/entropy",
}


class EpochCsvLogger:
    """Drop-in pufferl logger: appends one row per PPO epoch, flushing both files so
    partial runs stay on disk (edge-training-log discipline, as k4b/T1).
    Resume-aware: an existing non-empty CSV is APPENDED to (checkpoint resume), a fresh
    run truncates. save_checkpoint() keys the run dir off run_id."""

    def __init__(self, csv_path: Path, jsonl_path: Path, run_label: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.run_label = run_label
        self.run_id = run_label  # pufferl save_checkpoint() keys the run dir off this
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        resuming = csv_path.exists() and csv_path.stat().st_size > 0
        self._csv = open(csv_path, "a" if resuming else "w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=CSV_COLUMNS)
        if not resuming:
            self._writer.writeheader()
        self._jsonl = open(jsonl_path, "a")

    def log(self, logs: dict, step) -> None:
        row = {}
        for col in CSV_COLUMNS:
            v = logs.get(COLUMN_KEYS[col])
            row[col] = float(v) if v is not None else ""
        self._writer.writerow(row)
        self._csv.flush()
        self._jsonl.write(json.dumps({"run": self.run_label, **logs}, default=str) + "\n")
        self._jsonl.flush()
        # Human-readable progression line straight into the tee'd stdout log.
        def fnum(v, spec=".0f"):
            return format(v, spec) if isinstance(v, (int, float)) else "-"

        print(
            f"[{self.run_label}] step={fnum(row['step'])} epoch={fnum(row['epoch'])} "
            f"sps={fnum(row['sps'])} ret={fnum(row['episode_return'], '.3f')} "
            f"coll={fnum(row['collision_rate'], '.3f')} red={fnum(row['red_light_violation_rate'], '.3f')} "
            f"stop={fnum(row['stop_sign_violation_rate'], '.3f')} "
            f"prog={fnum(row['reward_progress_component'], '.5f')}",
            flush=True,
        )

    def close(self, model_path, early_stop) -> None:
        self._csv.close()
        self._jsonl.close()

    def upload_model(self, model_path) -> None:  # interface parity, nothing to upload
        pass


def read_epoch_rows(csv_path: Path) -> list[dict]:
    """Parsed real epoch rows (drops the terminal done_training log row, whose env stats
    are empty -- same filter as k4b read_run_metrics)."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("step")]
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                r[k] = None
    return [r for r in rows if r.get("episode_return") is not None or r.get("collision_rate") is not None]


# ------------------------------------------------------------------ worker ---


def run_worker(args) -> None:
    """One training run in this process. Invoked by the orchestrator as a subprocess
    with the absolute venv python path (PYTHONPATH=repo root).
    Resume semantics come from pufferl itself: an existing trainer_state.pt in data_dir
    auto-resumes (weights + optimizer + step counters + RNG); its absence restarts, and
    the orchestrator rotates stale metrics aside in that case."""
    import pufferlib.pufferl as pufferl
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    import hazard_demo as hd

    hd.verify_in_repo_kernel()  # pinning inside the worker process as well

    seed = args.seed
    if args.worker == "probe":
        # SPS/VRAM probe: face knobs matter, reward coefs are SPS-neutral -> config
        # defaults（v1 区间，不按臂覆盖）。Probe runs hazard-on too (mix SPS sizing).
        arm_overrides: dict[str, float] = {}
        run_label = f"t2v2_probe_{args.num_envs}x{args.num_agents}_s{seed}"
        run_dir = PROBE_DIR / f"{args.num_envs}x{args.num_agents}"
        # Short probe needs no resumability: skip mid-run checkpoint saves entirely.
        checkpoint_interval = 10**9
    else:
        arm_overrides = dict(ARMS[args.worker])
        run_label = f"t2v2_{args.worker}_s{seed}"
        run_dir = CKPT_DIR / args.worker
        epochs_total = args.steps // (args.num_envs * args.num_agents * 128)
        checkpoint_interval = max(1, epochs_total // 10)  # ~10% progress + endpoint

    # load_config treats sys.argv[1:] as Hydra overrides; clear them and mutate directly.
    sys.argv = [sys.argv[0]]
    cfg = pufferl.load_config("puffer_drive")
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["tb"] = False
    cfg["run_name"] = run_label
    cfg["train"]["seed"] = seed
    cfg["train"]["total_timesteps"] = args.steps
    cfg["train"]["data_dir"] = str(run_dir)
    cfg["train"]["checkpoint_interval"] = checkpoint_interval
    cfg["vec"]["seed"] = seed
    cfg["vec"]["num_envs"] = args.num_envs
    # num_workers auto -> num_envs, and pufferl raises APIUsageError when workers
    # exceed hardware threads (this box: 12) or don't divide num_envs -- which made
    # the assigned 16-env faces fail at spawn before any SPS/VRAM measurement. Idle
    # workers block instead of spinning (the guard's own documented mode), so pick
    # the largest divisor of num_envs within hw threads; face.json records it.
    hw_threads = psutil.cpu_count(logical=True) or 1
    cfg["vec"]["num_workers"] = max(
        d for d in range(1, min(args.num_envs, hw_threads) + 1) if args.num_envs % d == 0
    )
    cfg["env"]["num_agents"] = args.num_agents
    # T3 hazard-on mix 面（写法照抄 t2_hazsmoke run_worker 的 hazard_on 分支）：
    # parked/clusters/cones 默认开，walkers 显式开（四开）；静态 slot 预算按
    # planned-count highs 放大（kernel fail-fasts if short）。probe 与正式臂同走此路径。
    cfg["env"]["hazard_enable"] = True
    cfg["env"]["hazard_walkers_enable"] = True
    cfg["env"]["cosim_partner_slots"] = max(cfg["env"]["cosim_partner_slots"], hazard_slot_capacity(cfg["env"]))
    # 奖励重标按臂取值：progress 未覆盖键沿用 config 默认 v1 区间（kernel 校验 finite +
    # lo <= hi），覆盖键（reward_offroad / progress lo/hi）逐字喂 env。
    cfg["env"]["reward_progress_lo"] = arm_overrides.get("reward_progress_lo", PROGRESS_LO_DEFAULT)
    cfg["env"]["reward_progress_hi"] = arm_overrides.get("reward_progress_hi", PROGRESS_HI_DEFAULT)
    for key, value in arm_overrides.items():
        if key.startswith("train."):
            cfg["train"][key.split(".", 1)[1]] = value  # v3: trainer-level knobs (ent_coef …)
        else:
            cfg["env"][key] = value

    face = {
        "num_envs": cfg["vec"]["num_envs"],
        "num_workers": cfg["vec"]["num_workers"],
        "num_agents": cfg["env"]["num_agents"],
        "bptt_horizon": cfg["train"]["bptt_horizon"],
        "batch_size": cfg["vec"]["num_envs"] * cfg["env"]["num_agents"] * cfg["train"]["bptt_horizon"],
        "minibatch_size": cfg["train"]["minibatch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "total_timesteps": cfg["train"]["total_timesteps"],
        "checkpoint_interval": checkpoint_interval,
        "seed_train": cfg["train"]["seed"],
        "seed_vec": cfg["vec"]["seed"],
        "simulation_mode": cfg["env"]["simulation_mode"],
        "map_dir": cfg["env"]["map_dir"],
        "num_maps": cfg["env"]["num_maps"],
        "hazard_enable": cfg["env"]["hazard_enable"],
        "hazard_parked_enable": cfg["env"]["hazard_parked_enable"],
        "hazard_clusters_enable": cfg["env"]["hazard_clusters_enable"],
        "hazard_cones_enable": cfg["env"]["hazard_cones_enable"],
        "hazard_walkers_enable": cfg["env"]["hazard_walkers_enable"],
        "arm_env_overrides": arm_overrides,
        "reward_progress_lo": cfg["env"]["reward_progress_lo"],
        "reward_progress_hi": cfg["env"]["reward_progress_hi"],
        "reward_offroad": cfg["env"]["reward_offroad"],
        "stop_sign_dwell_range": cfg["env"]["stop_sign_dwell_range"],
        "traffic_light_footprint_sweep": cfg["env"]["traffic_light_footprint_sweep"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "face.json").write_text(json.dumps(face, indent=2))
    print(f"[worker:{args.worker}] face: {json.dumps(face)}", flush=True)

    logger = EpochCsvLogger(run_dir / "metrics.csv", run_dir / "metrics_full.jsonl", run_label)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        pufferl.train("puffer_drive", args=cfg, logger=logger)
    finally:
        logger.close(None, False)
        if torch.cuda.is_available():
            mem = {
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            }
            (run_dir / "memory.json").write_text(json.dumps(mem, indent=2))
            print(f"PEAK_CUDA {json.dumps(mem)}", flush=True)


# ------------------------------------------------------------- orchestrator ---


def spawn_worker(name: str, args, stdout_path: Path) -> tuple[int, float]:
    """Run one worker subprocess; returns (returncode, wall_seconds). No timeout set on
    the training process (edge discipline: never kill a training run)."""
    cmd = [
        str(python_bin()),
        str(REPO_ROOT / "scripts" / "t3_sweep.py"),
        "--worker",
        name,
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--num-envs",
        str(args.num_envs),
        "--num-agents",
        str(args.num_agents),
    ]
    if getattr(args, "art_dir", None):
        cmd += ["--art-dir", str(args.art_dir)]  # 子进程必须落同一产物树（默认目录陷阱）
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log(f"spawn worker {name} (face {args.num_envs}x{args.num_agents}, steps {args.steps}): {stdout_path.name}")
    t0 = time.time()
    with open(stdout_path, "w") as out:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    log(f"worker {name} finished rc={proc.returncode} wall={wall:.1f}s")
    return proc.returncode, wall


def run_probe(face: tuple[int, int], state: dict, args) -> dict:
    """Probe one face: 2 PPO epochs, measure steady SPS + fixed overhead + peak VRAM.
    Stability = clean exit + >= 2 metric rows + finite losses + memory stats captured.
    Two attempts, wiping the probe dir between them (probes are not resumable)."""
    envs, agents = face
    face_key = f"{envs}x{agents}"
    batch = envs * agents * 128
    run_dir = PROBE_DIR / face_key
    stdout_path = ART_DIR / f"probe_{face_key}_stdout.log"
    args.steps, args.num_envs, args.num_agents = 2 * batch, envs, agents
    result: dict = {"face": face_key, "probe_steps": 2 * batch, "stable": False, "attempts": 0}
    for attempt in range(1, 3):
        result["attempts"] = attempt
        if run_dir.exists():
            shutil.rmtree(run_dir)
        rc, wall = spawn_worker("probe", args, stdout_path)
        result["wall_s"] = round(wall, 1)
        rows = read_epoch_rows(run_dir / "metrics.csv")
        result["probe_rows"] = len(rows)
        mem = {}
        mem_path = run_dir / "memory.json"
        if mem_path.exists():
            mem = json.loads(mem_path.read_text())
        result["peak_mem"] = mem
        if rc == 0 and len(rows) >= 2:
            last = rows[-1]
            losses_finite = all(
                last.get(k) is not None and np.isfinite(last[k])
                for k in ("policy_loss", "value_loss", "entropy")
            )
            if losses_finite:
                result.update(
                    stable=True,
                    sps=round(last["sps"] or 0.0, 1),
                    active_s=round(last["uptime_s"] or wall, 1),
                    overhead_s=round(max(wall - (last["uptime_s"] or wall), 0.0), 1),
                )
                break
        result["error"] = f"attempt {attempt}: rc={rc} rows={len(rows)} mem={mem} (see {stdout_path.name})"
        log(f"probe {face_key} attempt {attempt} unstable: {result['error']}")
    state.setdefault("probes", {})[face_key] = result
    save_state(state)
    log(f"probe {face_key}: {'STABLE' if result['stable'] else 'UNSTABLE'} "
        f"sps={result.get('sps')} overhead_s={result.get('overhead_s')} peak_mem={mem}")
    return result


def probe_phase(state: dict, args) -> None:
    """Probe faces smallest -> largest; stop at the first unstable face (larger faces
    are classified unstable-by-risk). Skipped faces already probed stable on a relaunch."""
    state.setdefault("probes", {})
    for face in FACES:
        envs, agents = face
        face_key = f"{envs}x{agents}"
        prev = state["probes"].get(face_key)
        if prev and prev.get("stable"):
            log(f"probe {face_key} reused from state (stable, sps={prev.get('sps')})")
            continue
        if prev and not prev.get("stable") and prev.get("face") == face_key and face != FACES[0]:
            # A larger face after an unstable smaller one: unstable by ordering.
            log(f"probe {face_key} skipped (unstable-by-risk after {FACES[FACES.index(face)-1][0]}x"
                f"{FACES[FACES.index(face)-1][1]} failed earlier)")
            state["probes"][face_key] = {"face": face_key, "stable": False, "skipped_after_failure": True}
            save_state(state)
            continue
        run_probe(face, state, args)
        if not state["probes"][face_key]["stable"]:
            log(f"face {face_key} unstable; stopping the probe ladder (larger faces not attempted)")
            return


def choose_face(state: dict, arm_budget_s: int) -> tuple[tuple[int, int], dict]:
    """Largest stable face by total agents that the arm budget can sustain (>=
    MIN_ARM_EPOCHS epochs at that face's own probe SPS/overhead). At ignition budgets
    this is still simply the largest stable face; the feasibility fallback (the
    epochs<10 guard's own remedy "shrink FACES and rerun", done automatically here
    instead of by hand) only fires on tiny smoke budgets. size_arms' guard remains
    as the backstop."""
    stable = [
        (f, state["probes"][f"{f[0]}x{f[1]}"]) for f in FACES if state["probes"].get(f"{f[0]}x{f[1]}", {}).get("stable")
    ]
    if not stable:
        raise RuntimeError("no stable face found in probes; cannot size the sweep")
    stable.sort(key=lambda item: item[0][0] * item[0][1], reverse=True)
    face, probe = stable[0]
    rejected_for_budget: dict[str, int] = {}
    for f, p in stable:
        batch = f[0] * f[1] * 128
        epochs = int(max(arm_budget_s - p["overhead_s"], 0.0) * p["sps"]) // batch
        if epochs >= MIN_ARM_EPOCHS:
            face, probe = f, p
            break
        rejected_for_budget[f"{f[0]}x{f[1]}"] = epochs
    else:
        raise RuntimeError(
            f"no stable face sustains >= {MIN_ARM_EPOCHS} epochs at arm budget {arm_budget_s}s "
            f"(epochs per face: {rejected_for_budget}); budget too small or faces too heavy"
        )
    decision = {
        "chosen": f"{face[0]}x{face[1]}",
        "sps": probe["sps"],
        "overhead_s": probe["overhead_s"],
        "peak_mem": probe.get("peak_mem", {}),
        "probe_ladder": {f"{f[0]}x{f[1]}": (state["probes"].get(f"{f[0]}x{f[1]}") or {}).get("stable") for f in FACES},
        "rejected_for_budget": rejected_for_budget or None,
        "rule": "largest stable face (probe: 2 epochs, clean exit + finite losses + VRAM captured) sustaining "
                f">= {MIN_ARM_EPOCHS} epochs at the arm budget; unstable = OOM/crash; larger faces after a "
                "failure are unstable-by-risk",
    }
    log(f"face decision: {json.dumps(decision)}")
    return face, decision


def size_arms(face: tuple[int, int], probe: dict, state: dict, args) -> dict:
    """Honest sizing: ~arm-budget per arm, total sweep <= total-budget (hard cap).
    steps = floor-to-epoch((budget - overhead) * sps); shrink proportionally if the
    n-arm total would breach the cap (n = len(ARMS) = 2 here; 臂毕 auto-eval 墙钟
    不计入本预算——eval 是收割增量，不挤压训练步数)."""
    envs, agents = face
    batch = envs * agents * 128
    sps = max(probe["sps"], 1.0)
    overhead = probe["overhead_s"]
    probe_wall = sum(
        (state["probes"].get(f"{f[0]}x{f[1]}") or {}).get("wall_s", 0.0) for f in FACES
    )
    per_arm_step_s = args.arm_budget_s - overhead
    epochs = int(per_arm_step_s * sps) // batch
    steps = epochs * batch
    n_arms = 1 if getattr(args, "only_arm", None) else len(ARMS)
    if getattr(args, "arm_steps", 0):
        # v3: 步数对齐模式（跨机可比性）：目标步数直接钉死（floor-to-epoch），预算仅作上限校验。
        steps = (args.arm_steps // batch) * batch
        epochs = steps // batch
        est_arm_s = overhead + steps / sps
        if est_arm_s > args.arm_budget_s and probe_wall + est_arm_s > args.total_budget_s:
            shrink_note = f"arm-steps {steps} est {est_arm_s:.0f}s 超总预算，按总帽收缩"
            avail_step_s = (args.total_budget_s - probe_wall) / n_arms - overhead
            epochs = int(avail_step_s * sps) // batch
            steps = epochs * batch
        else:
            shrink_note = f"arm-steps pinned {steps} (B-parity), est {est_arm_s:.0f}s"
    else:
        total_est = probe_wall + n_arms * (overhead + steps / sps)
        shrink_note = None
        if total_est > args.total_budget_s:
            avail_step_s = (args.total_budget_s - probe_wall) / n_arms - overhead
            epochs = int(avail_step_s * sps) // batch
            steps = epochs * batch
            shrink_note = f"shrunk to fit {args.total_budget_s}s total cap (probe_wall {probe_wall:.0f}s)"
    if epochs < MIN_ARM_EPOCHS:
        raise RuntimeError(
            f"sizing collapsed to {epochs} epochs on face {envs}x{agents}; "
            "face too heavy for the budget on this GPU -- shrink FACES and rerun"
        )
    decision = {
        "hazard": "mix",
        "face": f"{envs}x{agents}",
        "batch": batch,
        "sps_used": sps,
        "overhead_s": overhead,
        "budget_per_arm_s": args.arm_budget_s,
        "total_cap_s": args.total_budget_s,
        "probe_wall_s": round(probe_wall, 1),
        "steps_per_arm": steps,
        "epochs_per_arm": epochs,
        "checkpoint_interval_epochs": max(1, epochs // 10),
        "est_per_arm_s": round(overhead + steps / sps, 1),
        "est_total_s": round(probe_wall + n_arms * (overhead + steps / sps), 1),
        "rule": "steps = floor-to-epoch((arm_budget - overhead) * sps); "
                f"shrunk proportionally if {n_arms}-arm total breached the cap",
    }
    if shrink_note:
        decision["note"] = shrink_note
    log(f"sizing: {json.dumps(decision)}")
    return decision


def prepare_arm_run(run_dir: Path) -> str:
    """Resume bookkeeping BEFORE spawning an arm worker:
    - trainer_state.pt present -> pufferl will auto-resume; keep metrics.csv (repair a
      possible trailing partial row from a mid-write crash).
    - no trainer_state.pt but metrics.csv non-empty -> the dead attempt died before its
      first checkpoint; rotate stale metrics aside so the fresh run doesn't duplicate
      rows from step 0."""
    trimmed = repair_csv_tail(run_dir / "metrics.csv")
    if (run_dir / "trainer_state.pt").exists():
        return f"resume from checkpoint (metrics.csv kept, tail_repaired={trimmed})"
    if (run_dir / "metrics.csv").exists() and (run_dir / "metrics.csv").stat().st_size > 0:
        attempts = len(list(run_dir.glob("metrics_attempt*.csv"))) + 1
        os.replace(run_dir / "metrics.csv", run_dir / f"metrics_attempt{attempts}.csv")
        if (run_dir / "metrics_full.jsonl").exists():
            os.replace(run_dir / "metrics_full.jsonl", run_dir / f"metrics_full_attempt{attempts}.jsonl")
        return f"fresh start (attempt died pre-checkpoint; stale metrics -> metrics_attempt{attempts}.csv)"
    return "fresh start"


def checkpoint_step(run_dir: Path) -> int | None:
    """global_step recorded in trainer_state.pt (exact completion evidence; -1 on any
    load failure so callers fall back to the CSV tail)."""
    import torch

    path = run_dir / "trainer_state.pt"
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return int(state.get("global_step", -1))
    except Exception as exc:  # noqa: BLE001 - any load failure falls back to CSV
        log(f"trainer_state.pt unreadable in {run_dir.name}: {exc}")
        return -1


def arm_progress(run_dir: Path) -> tuple[int, int | None]:
    rows = read_epoch_rows(run_dir / "metrics.csv")
    return len(rows), (int(rows[-1]["step"]) if rows else None)


def eta_line(now: float, remaining_s: float) -> str:
    return f"ETA all-arms finish ≈ {time.strftime('%m-%d %H:%M', time.localtime(now + remaining_s))} (remaining ~{remaining_s/3600:.1f}h)"


def fmt_overrides(o: dict) -> str:
    """Compact per-arm override display for panels/logs，例：A_v1ctl → 'config defaults (v1)'、
    B_proghalf → 'prog=[1.25e-3,1.25e-2]'。未覆盖的 progress 键按实际生效的 config 默认 v1
    区间显示。"""
    parts = []
    if "reward_offroad" in o:
        parts.append(f"offroad={_coef(o['reward_offroad'])}")
    if "reward_progress_lo" in o or "reward_progress_hi" in o:
        parts.append(
            f"prog=[{_coef(o.get('reward_progress_lo', PROGRESS_LO_DEFAULT))},"
            f"{_coef(o.get('reward_progress_hi', PROGRESS_HI_DEFAULT))}]"
        )
    return " · ".join(parts) if parts else "config defaults (v1)"


def eval_arm(arm: str, state: dict, num_scenarios: int) -> None:
    """臂毕自动 eval（本 runner 的核心增量）：该臂 ckpt/<arm>/models/ 下唯一 .pt
    （pufferl save_checkpoint 只保留最新一个 model_*.pt）逐面喂 scripts/t3_eval_ckpt.py
    subprocess：carla_fast 干净面 + --mix-face 各一次，--label <arm>，--out
    artifacts/t3-sweep/eval-{clean,mix}-<arm>（env 继承 PYTHONPATH + 显式
    PYTORCH_CUDA_ALLOC_CONF）。eval 失败不炸主流程：醒目警告 + 该面记 failed，训练
    数据照常收割。状态进 state.json：done 的面 relaunch 时跳过，failed 的面重试。"""
    st = state["arms"].setdefault(arm, {})
    eval_st = st.setdefault("eval", {})
    models = sorted((CKPT_DIR / arm / "models").glob("*.pt"))
    if not models:
        eval_st["clean"] = eval_st["mix"] = "failed"
        eval_st["error"] = f"no model .pt under ckpt/{arm}/models/"
        log(f"!!! [EVAL FAIL] arm {arm}: {eval_st['error']} —— eval 记 failed，训练数据照常收割")
        save_state(state)
        return
    model_path = models[-1]
    eval_st["model"] = str(model_path)
    faces = (
        ("clean", [], ART_DIR / f"eval-clean-{arm}"),
        ("mix", ["--mix-face"], ART_DIR / f"eval-mix-{arm}"),
    )
    for face, extra_flags, out_dir in faces:
        if eval_st.get(face) == "done":
            log(f"eval {arm}/{face} already done ({eval_st.get(face + '_n')} scenarios); skipping")
            continue
        cmd = [
            str(python_bin()),
            str(EVAL_SCRIPT),
            "--models", str(model_path),
            "--label", arm,
            "--benchmark", EVAL_BENCHMARK,
            "--num-scenarios", str(num_scenarios),
            "--out", str(out_dir),
        ] + extra_flags
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log_path = ART_DIR / f"eval_{arm}_{face}_stdout.log"
        log(f"=== arm {arm} auto-eval [{face}]: {EVAL_BENCHMARK} n={num_scenarios} model={model_path.name} "
            f"-> {out_dir.name} (log {log_path.name})")
        t0 = time.time()
        with open(log_path, "w") as out:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        ok = proc.returncode == 0 and (out_dir / "summary.json").exists()
        if ok:
            eval_st[face] = "done"
            eval_st[face + "_n"] = num_scenarios
            log(f"eval {arm}/{face} DONE rc=0 wall={(time.time() - t0) / 60:.1f}min -> {out_dir.name}/summary.json")
        else:
            eval_st[face] = "failed"
            eval_st[face + "_error"] = f"rc={proc.returncode} (log {log_path.name})"
            log(f"!!! [EVAL FAIL] arm {arm} face={face}: rc={proc.returncode} ≠0 或 summary.json 缺失 "
                f"（log: {log_path.name}）—— eval 记 failed，训练数据照常收割，主流程继续")
        save_state(state)


def run_arms(state: dict, args, sizing: dict) -> None:
    """Serial arms with resume: completed skip, crashed resume from checkpoint, dead
    recorded and the sweep continues. Two in-place attempts per arm (no timeout).
    每臂 done 后自动 eval（eval_arm）；relaunch 时 done 臂也补漏 eval（failed/缺失面重试）。"""
    envs, agents = [int(x) for x in sizing["face"].split("x")]
    steps_target = sizing["steps_per_arm"]
    batch = sizing["batch"]
    est_per_arm = sizing["est_per_arm_s"]
    for arm, overrides in ARMS.items():
        if getattr(args, "only_arm", None) and arm != args.only_arm:
            continue  # v3 双卡并行：本进程只跑指定臂
        st = state["arms"].setdefault(arm, {"status": "pending", "attempts": 0})
        if st["status"] == "done":
            log(f"=== arm {arm} already DONE (last_step={st.get('last_step')}); skipping training")
            eval_arm(arm, state, args.eval_scenarios)
            continue
        if st["status"] == "dead":
            log(f"=== arm {arm} DEAD from a previous launch ({st.get('error')}); skipping "
                "(delete its ckpt dir + state entry to re-arm it)")
            continue
        run_dir = CKPT_DIR / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        note = prepare_arm_run(run_dir)
        st.update(status="running", overrides=dict(overrides), face=sizing["face"], steps_target=steps_target)
        save_state(state)
        remaining_before = sum(
            est_per_arm
            for a2, s2 in ARMS.items()
            if (state["arms"].get(a2) or {}).get("status") not in ("done", "dead")
        )
        log(f"=== arm {arm} start: overrides={fmt_overrides(overrides)} steps={steps_target} "
            f"epochs={sizing['epochs_per_arm']} (ckpt every {sizing['checkpoint_interval_epochs']} epochs) "
            f"est {est_per_arm/3600:.2f}h; {eta_line(time.time(), remaining_before)}; {note}")
        done = False
        error = None
        for attempt in range(1, 3):
            st["attempts"] = attempt
            save_state(state)
            stdout_path = ART_DIR / f"arm_{arm}_attempt{attempt}_stdout.log"
            rc, wall = spawn_worker(arm, args_for_arm(args, envs, agents, steps_target), stdout_path)
            ckpt_step = checkpoint_step(run_dir)
            n_rows, last_csv_step = arm_progress(run_dir)
            st.update(wall_s=round(wall, 1), csv_rows=n_rows, last_csv_step=last_csv_step)
            ckpt_done = ckpt_step is not None and ckpt_step >= steps_target
            csv_done = bool(last_csv_step) and last_csv_step >= steps_target - batch
            if rc == 0 and (ckpt_done or csv_done):
                st.update(status="done", last_step=max(ckpt_step or 0, last_csv_step or 0))
                done = True
                log(f"arm {arm} DONE: trainer_state.global_step={ckpt_step} csv_rows={n_rows} "
                    f"wall={wall/3600:.2f}h")
                break
            error = f"attempt {attempt}: rc={rc} ckpt_step={ckpt_step} last_csv_step={last_csv_step} (see {stdout_path.name})"
            log(f"arm {arm} attempt {attempt} incomplete: {error}; in-place retry (checkpoint resume)")
        if done:
            eval_arm(arm, state, args.eval_scenarios)
        else:
            st.update(status="dead", error=error)
            log(f"arm {arm} DEAD after {st['attempts']} attempts: {error}; moving on (eval skipped)")
        save_state(state)


def args_for_arm(args, envs: int, agents: int, steps: int):
    """Namespace for spawn_worker (same fields as the orchestrator's, arm-sized)."""
    import copy

    a = copy.copy(args)
    a.num_envs, a.num_agents, a.steps = envs, agents, steps
    return a


def view_and_harvest_block(state: dict, sizing: dict | None) -> list[str]:
    """The standing 'how to watch / how to harvest' block for orchestrator.log."""
    lines = [
        "---- 进度查看命令 ----",
        "  tail -f artifacts/t3-sweep/orchestrator.log",
        "  wc -l artifacts/t3-sweep/ckpt/*/metrics.csv   # 各臂 epoch 行数（含表头）",
        "  tail -2 artifacts/t3-sweep/ckpt/B_proghalf/metrics.csv",
        "  tail -3 artifacts/t3-sweep/eval_B_proghalf_mix_stdout.log   # 最新臂 eval 日志尾",
        "  ps -p $(cat artifacts/t3-sweep/state.json | python3 -c 'import json,sys;print(json.load(sys.stdin).get(\"pid\",0))') && nvidia-smi",
        "---- v2 sweep 收割入口（臂 done/dead 后均可跑，用现存行/现存 eval 收割）----",
        "  PYTHONPATH=$PWD .venv/bin/python scripts/t3_sweep.py --collect-only",
        "  -> artifacts/t3-sweep/{metrics.csv(合并), curves.png, summary.json(含顶层 eval 块), index.html(含两张 eval 对照表)}",
        "     + eval-{clean,mix}-<arm>/{summary.json, index.html}（臂毕自动 eval 产物）",
    ]
    if sizing:
        lines.append(
            f"  （sizing: face {sizing['face']}, {sizing['steps_per_arm']} steps/arm "
            f"= {sizing['epochs_per_arm']} epochs, est_total {sizing['est_total_s']/3600:.1f}h）"
        )
    return lines


def final_summary(state: dict, sizing: dict | None, args=None) -> None:
    def fnum(v):
        return f"{v:.3f}" if isinstance(v, float) else "-"

    log("==== T3 SWEEP FINAL SUMMARY ====")
    for arm in ((getattr(args, "only_arm", None) and [args.only_arm]) or list(ARMS)):
        st = state["arms"].get(arm) or {}
        eval_st = st.get("eval") or {}
        rows = read_epoch_rows(CKPT_DIR / arm / "metrics.csv")
        if rows:
            last = rows[-1]
            log(
                f"arm {arm}: status={st.get('status')} overrides={fmt_overrides(ARMS[arm])} rows={len(rows)} "
                f"last_step={int(last['step'])} ret={fnum(last['episode_return'])} coll={fnum(last['collision_rate'])} "
                f"red={fnum(last['red_light_violation_rate'])} stop={fnum(last['stop_sign_violation_rate'])} "
                f"eval[clean={eval_st.get('clean', 'n/a')} mix={eval_st.get('mix', 'n/a')}]"
            )
        else:
            log(f"arm {arm}: status={st.get('status')} rows=0 eval[clean={eval_st.get('clean', 'n/a')} "
                f"mix={eval_st.get('mix', 'n/a')}]")
    state["phase"] = "done"
    save_state(state)
    for line in view_and_harvest_block(state, sizing):
        log(line)
    log("==== sweep complete ====")


def orchestrate(args) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    provenance = verify_in_repo_kernel()
    # 内核连续性门：必须在 probe（及任何训练）之前过门（t2_hazsmoke 同款）。
    binding_match = gate_t2_kernel_binding(provenance, args.allow_kernel_drift)
    state = load_state()
    state.setdefault("probes", {})
    state.setdefault("arms", {})
    state.setdefault("sizing", None)
    state["pid"] = os.getpid()
    state["kernel"] = provenance
    state["hazard"] = "mix"  # v2 sweep hazard-on mix 面（render_index 文案读此键）
    state["t2_binding_sha256_match"] = binding_match  # collect 顶层转记进 summary.json
    if "started_at" not in state:
        state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    log(f"T2-v2 sweep orchestrator pid={state['pid']} kernel pinned: "
        f"binding sha256={provenance['binding_sha256'][:12]}… _C sha256={provenance['advantage_sha256'][:12]}… "
        f"gpu={provenance['gpu']} git={provenance['git_sha'][:12]}")

    sizing = state.get("sizing")
    if sizing:
        log(f"sizing reused from state: face={sizing['face']} steps/arm={sizing['steps_per_arm']} "
            f"epochs={sizing['epochs_per_arm']}")
        face = tuple(int(x) for x in sizing["face"].split("x"))
    else:
        probe_phase(state, args)
        face, face_decision = choose_face(state, args.arm_budget_s)
        probe = state["probes"][face_decision["chosen"]]
        sizing = size_arms(face, probe, state, args)
        sizing["face_decision"] = face_decision
        state["sizing"] = sizing
        save_state(state)
        DECISION_PATH.write_text(json.dumps(
            {"kernel": provenance, "probes": state["probes"], "face_decision": face_decision,
             "sizing": sizing, "arms": {a: {"overrides": dict(o)} for a, o in ARMS.items()}},
            indent=2, ensure_ascii=False))
        log(f"decision snapshot written: {DECISION_PATH.name}")

    # Ignition block: ETA + how-to-watch + harvest entry, into the orchestrator log.
    arms_left = [a for a in ARMS if (state["arms"].get(a) or {}).get("status") != "done"]
    log("---- T2-v2 sweep ignition: " + f"{len(arms_left)} arm(s) to run ({', '.join(arms_left) or 'none'}); "
        + eta_line(time.time(), len(arms_left) * sizing["est_per_arm_s"]) + " ----")
    for line in view_and_harvest_block(state, sizing):
        log(line)

    run_arms(state, args, sizing)
    final_summary(state, sizing, args)


# ------------------------------------------------------------ collect (v2 harvest) ---


def seg_stats(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "epochs": len(rows),
        "episode_return": mean("episode_return"),
        "collision_rate": mean("collision_rate"),
        "offroad_rate": mean("offroad_rate"),
        "red_light_violation_rate": mean("red_light_violation_rate"),
        "stop_sign_violation_rate": mean("stop_sign_violation_rate"),
        "num_goals_reached": mean("num_goals_reached"),
        "sps": mean("sps"),
        "reward_progress_component": mean("reward_progress_component"),
    }


def _sci(x: float) -> str:
    """Compact scientific form, manual-page style: 221773824 -> '2.218e8' (plain .3g
    would print 2.22e+08 and drop the 4th digit the frozen manual page shows)."""
    mant, exp = f"{x:.3e}".split("e")
    return f"{mant}e{int(exp)}"


def _coef(x: float) -> str:
    """Override/coefficient display: 3.0 -> '3', 1.25e-3 -> '1.25e-3', 2.5e-2 -> '2.5e-2'."""
    if x == int(x):
        return str(int(x))
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def _fmt4(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _window_mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _mid_last_windows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Manual-page window semantics (digits frozen in index.manual.html): mid =
    rows[int(0.45n):int(0.55n)] -- exclusive end reproduces the manual page
    digit-for-digit (a +1 here shifts 7 cells by 1e-4; owner ruling 2026-11-10),
    last = rows[int(0.90n):]. Tiny-n guard: mid = first half, last = final
    min(3, n) rows, both never empty."""
    n = len(rows)
    mid = rows[int(0.45 * n):int(0.55 * n)]
    last = rows[int(0.90 * n):]
    if not mid or not last:
        mid = rows[:max(1, n // 2)]
        last = rows[-min(3, n):]
    return mid, last


_INDEX_CSS = """body{font-family:'Noto Sans CJK SC',sans-serif;margin:24px;background:#fafafa;color:#222;max-width:1280px}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-left:4px solid #2b6cb0;padding-left:8px}
table{border-collapse:collapse;font-size:13px;margin:10px 0}
th,td{border:1px solid #ccc;padding:5px 9px;text-align:right}
th{background:#e8eef7} td:first-child,th:first-child{text-align:left}
.s{font-weight:400;font-size:10px;color:#777}
img{max-width:100%;border:1px solid #ddd;background:#fff}
.box{background:#fff;border:1px solid #ddd;padding:10px 14px;font-size:13px;margin:8px 0}
.win{background:#eafbea;border-color:#9ad89a} code{background:#eee;padding:1px 5px;font-size:12px}"""

# Quant-table metric columns in display order; labels match the frozen manual page.
_INDEX_METRICS = [
    ("episode_return", "return"),
    ("collision_rate", "碰撞率"),
    ("num_goals_reached", "goals/agent"),
    ("offroad_rate", "出路率"),
    ("red_light_violation_rate", "红灯违规"),
    ("stop_sign_violation_rate", "停牌违规"),
]

# eval 对照表六指标（判读核心：return / 出路率 / 碰撞 / goals / DNF / 有无进展）。
_EVAL_METRICS = [
    "episode_return",
    "offroad_rate",
    "collision_rate",
    "num_goals_reached",
    "dnf_rate",
    "making_progress_rate",
]


def load_eval_block(state: dict, runs: dict) -> dict:
    """Per-face per-arm eval comparison 块：逐臂读 eval-{clean,mix}-<arm>/summary.json，
    comparison 原样嵌入（{metric: {label: value}}）；缺失的臂/面记入 failed（附 state 里的
    eval 状态），num_scenarios 取自首个读到的 eval summary。收割安全：eval 未跑/失败不炸。"""
    block: dict = {"clean": {}, "mix": {}, "failed": {}}
    scen: dict = {"clean": None, "mix": None}
    for arm in runs:
        for face, prefix in (("clean", "eval-clean-"), ("mix", "eval-mix-")):
            path = ART_DIR / f"{prefix}{arm}" / "summary.json"
            if path.exists():
                data = json.loads(path.read_text())
                block[face][arm] = data.get("comparison", {})
                scen[face] = data.get("num_scenarios", scen[face])
            else:
                st_eval = (state.get("arms", {}).get(arm) or {}).get("eval") or {}
                block["failed"][f"{arm}/{face}"] = st_eval.get(face, "summary.json 缺失（未跑或失败）")
    block["num_scenarios"] = scen
    return block


def render_index(runs: dict[str, list[dict]], state: dict, provenance: dict, eval_block: dict) -> None:
    """Rebuild ART_DIR/'index.html' (utf-8) from live runs/state/provenance -- the page
    itself promises 「本页由收割器重建」. CSS/structure mirror the hand-written page
    (index.manual.html); table windows + 4-decimal rounding reproduce it digit-for-digit."""
    sizing = state.get("sizing") or {}
    face = str(sizing.get("face", "?"))
    parts = face.split("x")
    face_txt = f"{parts[0]} env × {parts[1]} agents" if len(parts) == 2 else face
    steps_txt = _sci(sizing["steps_per_arm"]) if sizing.get("steps_per_arm") is not None else "?"
    sps = sizing.get("sps_used")
    sps_txt = f"~{sps / 1000:.0f}K SPS/臂" if sps else "SPS n/a"
    # hazard 文案读 state（键缺失时默认 off，文案与 T2 页完全一致）。
    hazard_txt = "hazard ON（mix 面）" if state.get("hazard") == "mix" else "hazard off"

    trs = []
    for arm, rows in runs.items():
        mid, last = _mid_last_windows(rows)
        cells = "".join(
            f"<td>{_fmt4(_window_mean(mid, key))} → <b>{_fmt4(_window_mean(last, key))}</b></td>"
            for key, _ in _INDEX_METRICS
        )
        trs.append(
            f"<tr><td>{arm} · {fmt_overrides(ARMS[arm])}</td>"
            f"<td>{_sci(rows[-1]['step'])}</td>{cells}</tr>"
        )

    ths = "<th>臂</th><th>步数</th>" + "".join(
        f"<th>{label}<br><span class='s'>中段 → 末10%</span></th>" for _, label in _INDEX_METRICS
    )
    trs_html = "".join(trs)
    prov_box = (
        f"pufferlib: <code>{html.escape(str(provenance.get('pufferlib_path', '?')))}</code><br>"
        f"binding .so sha256: <code>{str(provenance.get('binding_sha256', '?'))[:16]}…</code> · "
        f"_C sha256: <code>{str(provenance.get('advantage_sha256', '?'))[:16]}…</code><br>"
        f"torch {provenance.get('torch', '?')} · {provenance.get('gpu', '?')} · "
        f"git {str(provenance.get('git_sha', '?'))[:9]} · 构建 <code>{html.escape(str(provenance.get('built_by', '?')))}</code>"
    )
    def eval_table(face: str) -> str:
        """一张 eval 对照表：行 = 臂，列 = _EVAL_METRICS 六指标；数据来自
        eval-{face}-<arm>/summary.json 的 comparison（comparison 键序 = metric → label=臂）。"""
        comp_by_arm = (eval_block or {}).get(face) or {}  # {arm: {metric: {label: value}}}
        scen_n = (eval_block or {}).get("num_scenarios", {}).get(face)
        n_txt = f"{scen_n} 场景" if scen_n is not None else "场景数见各 eval summary"
        ths_e = "<th>臂</th>" + "".join(f"<th>{m}</th>" for m in _EVAL_METRICS)
        if comp_by_arm:
            rows_e = "".join(
                f"<tr><td>{arm}</td>"
                + "".join(f"<td>{_fmt4((comp_by_arm.get(arm) or {}).get(m, {}).get(arm))}</td>" for m in _EVAL_METRICS)
                + "</tr>"
                for arm in ARMS
            )
            note = f"来源 eval-{face}-<arm>/summary.json comparison（原样嵌入 summary.json 顶层 eval 块）"
        else:
            rows_e = "".join(
                f"<tr><td>{arm}</td>" + "<td>—</td>" * len(_EVAL_METRICS) + "</tr>" for arm in ARMS
            )
            note = "无该面 eval 数据（failed/未跑，见 summary.json 顶层 eval 块的 failed）"
        return f"<table><tr>{ths_e}</tr>{rows_e}</table><div class='s'>{n_txt} · {note}</div>"

    page = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>T2-v2 两臂短扫 · mix 面重标</title>
<style>
{_INDEX_CSS}
</style></head><body>
<h1>T2-v2 两臂短扫：mix 面重标（hazard mix 面）</h1>
<div class="box">face: {face_txt} · batch {sizing.get("batch", "?")} · 每臂 {steps_txt} 步（{sizing.get("epochs_per_arm", "?")} epochs）· seed 42 · {hazard_txt} · {provenance.get("gpu", "?")} · 实测 {sps_txt}<br>
<b style="color:#2f855a">判读：B 全量（progress 减半，无安全钉价）· 验收口径 = 安全+效率（速度贴限速，拒绝瘫痪型保守）· 对照 B@4.1e7（artifacts/t2-v2sweep/）与 C 全量（artifacts/t2-v2final/）</b><br>
<span class="s">两臂 = A_v1ctl（空覆盖 = v1 对照，补 v1@5e7 缺 eval 缺口）/ B_proghalf（progress 减半）；offroad 权重方向 pending owner（reward_randomization 下 env.reward_offroad 训练惰性）；根因假设 = 借出路换机动性。</span></div>

<h2>学习曲线（两臂）</h2>
<img src="curves.png" alt="T2-v2 curves">
<div class="s">上：episode return（每 epoch）；中：碰撞/出路率与 goals；SPS 与 losses。滑动窗口平滑。</div>

<h2>量化对照（中段 45–55% → 末 10% 窗口均值）</h2>
<table><tr>{ths}</tr>{trs_html}</table>

<h2>eval 对照 · carla_fast 干净面（臂毕自动 eval）</h2>
{eval_table("clean")}

<h2>eval 对照 · hazard mix 面（臂毕自动 eval）</h2>
{eval_table("mix")}

<h2>复现</h2>
<div class="box"><code>{html.escape(DETACHED_LAUNCH)}</code><br>
收割：<code>{HARVEST_CMD}</code>（本页由收割器重建）</div>

<h2>内核溯源</h2>
<div class="box">{prov_box}</div>
<div class="s">数字可由 metrics.csv 复算；本页由 --collect-only 自动重建。</div>
</body></html>"""
    (ART_DIR / "index.html").write_text(page, encoding="utf-8")


def collect(args) -> None:
    """v2 sweep 收割入口: merged metrics + 两臂 curves + summary（含顶层 eval 块）+
    index.html（含两张 eval 对照表）from whatever rows/evals exist (safe mid-run).
    Reuses state.json for face/sizing/kernel provenance and the kernel-gate result
    (t2_binding_sha256_match)."""
    provenance = verify_in_repo_kernel()
    state = load_state()
    runs = {}
    for arm in ARMS:
        rows = read_epoch_rows(CKPT_DIR / arm / "metrics.csv")
        if rows:
            runs[arm] = rows
    if not runs:
        raise RuntimeError("no arm metrics found under artifacts/t3-sweep/ckpt/*/metrics.csv")
    log(f"collect: arms with data: {', '.join(f'{a}({len(r)})' for a, r in runs.items())}")

    merged = ART_DIR / "metrics.csv"
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm"] + CSV_COLUMNS)
        w.writeheader()
        for arm, rows in runs.items():
            for r in rows:
                out = {"arm": arm}
                for col in CSV_COLUMNS:
                    v = r.get(col)
                    out[col] = "" if v is None else v
                w.writerow(out)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def slide(vals, window):
        v = np.asarray(vals, dtype=float)
        if len(v) < 2 or window <= 1:
            return v
        w = min(window, len(v))
        k = np.ones(w) / w
        sm = np.convolve(v, k, mode="valid")
        out = np.empty_like(v)
        out[: len(sm)] = sm
        out[len(sm):] = sm[-1]
        return out

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.25)
    axes = {
        "episode_return": fig.add_subplot(gs[0, :]),
        "collision_rate": fig.add_subplot(gs[1, 0]),
        "offroad_rate": fig.add_subplot(gs[1, 1]),
        "num_goals_reached": fig.add_subplot(gs[1, 2]),
        "violations": fig.add_subplot(gs[2, 0]),
        "sps": fig.add_subplot(gs[2, 1]),
        "losses": fig.add_subplot(gs[2, 2]),
    }
    for arm, rows in runs.items():
        steps = [r["step"] for r in rows]
        w = max(2, len(rows) // 8)
        c = ARM_COLORS[arm]
        for key, ax in (("episode_return", axes["episode_return"]), ("collision_rate", axes["collision_rate"]),
                        ("offroad_rate", axes["offroad_rate"]), ("num_goals_reached", axes["num_goals_reached"]),
                        ("sps", axes["sps"])):
            vals = [r[key] or float("nan") for r in rows]
            ax.plot(steps, vals, "o-", color=c, alpha=0.25, ms=2, lw=0.8)
            ax.plot(steps, slide(vals, w), "-", color=c, lw=2, label=arm)
        for key, ls in (("red_light_violation_rate", "-"), ("stop_sign_violation_rate", "--")):
            vals = [r[key] or float("nan") for r in rows]
            axes["violations"].plot(steps, slide(vals, w), ls, color=c, lw=2, label=f"{arm} {key.split('_')[0]}")
        for key, ls in (("policy_loss", "-"), ("value_loss", "--"), ("entropy", ":")):
            vals = [r[key] or float("nan") for r in rows]
            axes["losses"].plot(steps, slide(vals, w), ls, color=c, lw=1.8, label=f"{arm} {key}")
    titles = {
        "episode_return": "T2-v2 sweep: episode return (per-epoch, 2 arms, hazard mix face)",
        "collision_rate": "collision_rate", "offroad_rate": "offroad_rate",
        "num_goals_reached": "num_goals_reached", "violations": "violations (red -, stop - -)",
        "sps": "SPS", "losses": "losses (policy -, value - -, entropy :)",
    }
    for key, title in titles.items():
        ax = axes[key]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("agent steps", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(ART_DIR / "curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "kernel": provenance,
        "t2_binding_sha256_match": state.get("t2_binding_sha256_match"),
        "face": state.get("sizing"),
        "arms": {},
        # 顶层 eval 块：各臂两面的 comparison 原样嵌入（eval-{clean,mix}-<arm>/summary.json）。
        "eval": load_eval_block(state, runs),
    }
    for arm, rows in runs.items():
        n = len(rows)
        first = seg_stats(rows[: max(1, round(n * 0.25))])
        last = seg_stats(rows[-max(1, round(n * 0.25)):])
        summary["arms"][arm] = {
            "status": (state.get("arms", {}).get(arm) or {}).get("status", "unknown"),
            "epochs_logged": n,
            "steps_from": rows[0]["step"],
            "steps_to": rows[-1]["step"],
            "first_segment": first,
            "last_segment": last,
            "delta": {k: (None if last[k] is None or first[k] is None else round(last[k] - first[k], 4))
                      for k in first if k != "epochs"},
        }
    (ART_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    render_index(runs, state, provenance, summary["eval"])
    log(f"collect done: {merged.name}, curves.png, summary.json(含 eval 块), index.html(含两张 eval 表) under {ART_DIR}")


# --------------------------------------------------------------------- main ---


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arm-budget-s", type=int, default=6300, help="per-arm wall target (~1300s ≈ 5e7 步 @47K SPS)")
    p.add_argument("--total-budget-s", type=int, default=9000, help="hard cap on total training wall (v2 全量单臂)")
    p.add_argument("--eval-scenarios", type=int, default=EVAL_SCENARIOS_DEFAULT,
                   help=f"臂毕自动 eval 的场景数（每臂每面，默认 {EVAL_SCENARIOS_DEFAULT}；自测可调小）")
    p.add_argument("--collect-only", action="store_true", help="v2 sweep harvest: rebuild merged metrics/curves/summary(含 eval 块)/index")
    p.add_argument("--allow-kernel-drift", action="store_true", help="内核 sha 与 T2 主训不一致时仍继续，仅供 owner 明示")
    p.add_argument("--worker", choices=list(ARMS) + ["probe"], help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-envs", type=int, default=8, help=argparse.SUPPRESS)
    p.add_argument("--num-agents", type=int, default=128, help=argparse.SUPPRESS)
    p.add_argument("--only-arm", choices=list(ARMS), help="v3 双卡并行：本次只跑指定臂（另一卡另起进程）")
    p.add_argument("--arm-steps", type=int, default=0, help="v3 步数对齐模式：钉死每臂步数（floor-to-epoch），预算仅作上限（跨机可比性）")
    p.add_argument("--art-dir", default=None, help="v3 双卡隔离：覆盖产物目录（默认 artifacts/t3-sweep；双卡各用一目录避免 state 竞争）")
    args = p.parse_args()

    if args.art_dir:
        art = (REPO_ROOT / args.art_dir).resolve() if not str(args.art_dir).startswith("/") else Path(args.art_dir)
        global ART_DIR, CKPT_DIR, PROBE_DIR, STATE_PATH, DECISION_PATH, ORCH_LOG
        ART_DIR = art
        CKPT_DIR = art / "ckpt"
        PROBE_DIR = art / "probe"
        STATE_PATH = art / "state.json"
        DECISION_PATH = art / "decision.json"
        ORCH_LOG = art / "orchestrator.log"

    if args.worker:
        run_worker(args)
        return
    if args.collect_only:
        collect(args)
        return
    try:
        orchestrate(args)
    except Exception:
        # Fatal orchestrator errors must land line-by-line in orchestrator.log too
        # (the console log is the fallback, but orchestrator.log is the entry point).
        import traceback

        log("FATAL orchestrator error:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
