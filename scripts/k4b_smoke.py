#!/usr/bin/env python3
"""K4b PPO smoke: hazard on/off learning curves (K-series finale, zero-to-one first signal).

Two short PPO runs on an identical config (same seed, same batch/lr, same default carla
map face, gigaflow), differing ONLY in the hazard face:
  - hazard_off: env.hazard_enable=false (bit-identical to a hazard-free env)
  - hazard_on:  env.hazard_enable=true + parked/clusters/cones/walkers all on (mix face),
    with the static-slot budget cosim_partner_slots sized to the planned-count highs.

Success criterion (stated up front, judged honestly): reward curve trending up AND
collision/violation rates trending down = the learning signal exists. This smoke exists
to validate the pipeline and the signal, NOT to produce a score: no parameter tuning to
make the curves look good. Progress shaping is deliberately absent (ticket 06 main-train
item); the default reward is used as-is.

Pipeline: probe (measure SPS) -> honest sizing (<=12 min per run) -> run A -> run B
(serial) -> artifacts under artifacts/k4b-smoke/ (curves.png, metrics.csv, summary.json,
index.html). Training metrics are flushed to disk every epoch (CSV + JSONL sidecar), so a
crash still leaves partial data.

Repro:
  cd <repo root>
  PYTHONPATH=$PWD .venv/bin/python scripts/k4b_smoke.py --steps 1000000 --seed 42

Requires pufferlib._C (advantage kernel) built in-repo:
  NO_OCEAN=1 CC=<conda gcc13> CXX=<conda g++13> python setup.py build_torch --inplace --force
(see summary.json kernel block for the exact provenance of a given run).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts" / "k4b-smoke"
INTERPRETER_CANDIDATES = [
    REPO_ROOT / ".venv" / "bin" / "python",  # 本仓自持 venv（本地化后首选）
    REPO_ROOT.parent / "PufferDrive" / ".venv" / "bin" / "python",  # 仅应急回退
    Path(sys.executable),
]
RUN_MODES = ("hazard_off", "hazard_on")
# Guard so scripts/ imports (hazard_demo provenance helpers) resolve regardless of cwd.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------- utilities --


def log(msg: str) -> None:
    """Stamp progress to stdout AND the orchestrator log (every step hits disk)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_DIR / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def python_bin() -> Path:
    """Interpreter that owns torch; the sibling venv is borrowed for Python only.
    NOTE: return the candidate AS-IS (no .resolve()): the venv python is a symlink
    into miniconda, and resolving it strips the venv context (site-packages/pyvenv.cfg
    association follows the invoked path, not the physical binary)."""
    for cand in INTERPRETER_CANDIDATES:
        if cand.exists():
            return cand
    raise RuntimeError(f"no usable interpreter among {INTERPRETER_CANDIDATES}")


def verify_in_repo_kernel() -> dict:
    """Kernel pinning discipline (same rule as the K-series demos): pufferlib and the
    drive binding .so must resolve to THIS repo; summary carries the provenance block."""
    sys.path.insert(0, str(REPO_ROOT))
    import hazard_demo as hd  # repo scripts dir; provides verify/provenance helpers

    hd.verify_in_repo_kernel()
    prov = hd.kernel_provenance()
    # The advantage kernel (pufferlib._C) must also be the in-repo build: it is part of
    # the pufferlib package, so its path is implied, but we record its hash too.
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
    prov["advantage_built_by"] = (
        "NO_OCEAN=1 CC=/home/zoo/miniconda3/envs/pufdrive-gcc13/bin/"
        "x86_64-conda-linux-gnu-cc CXX=.../x86_64-conda-linux-gnu-g++ "
        "python setup.py build_torch --inplace --force"
    )
    prov["torch"] = torch.__version__
    prov["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return prov


def hazard_slot_capacity(env_cfg: dict) -> int:
    """Mirror of Drive.hazard_slot_capacity: planned-count highs of the hazard face,
    computed from the *final* env config so the slot budget always fits the knobs."""
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


# ------------------------------------------------------------------ worker ---

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
    "policy_loss": "losses/policy_loss",
    "value_loss": "losses/value_loss",
    "entropy": "losses/entropy",
}


class EpochCsvLogger:
    """Drop-in pufferl logger: appends one row per PPO epoch, flushing both files so
    partial runs stay on disk (edge-training-log discipline)."""

    def __init__(self, csv_path: Path, jsonl_path: Path, run_label: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.run_label = run_label
        self.run_id = run_label  # pufferl save_checkpoint() keys the run dir off this
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._jsonl = open(jsonl_path, "w")

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
            return format(v, spec) if isinstance(v, float) else "-"

        print(
            f"[{self.run_label}] step={fnum(row['step'])} epoch={fnum(row['epoch'])} "
            f"sps={fnum(row['sps'])} ret={fnum(row['episode_return'], '.3f')} "
            f"coll={fnum(row['collision_rate'], '.3f')} red={fnum(row['red_light_violation_rate'], '.3f')} "
            f"stop={fnum(row['stop_sign_violation_rate'], '.3f')}",
            flush=True,
        )

    def close(self, model_path, early_stop) -> None:
        self._csv.close()
        self._jsonl.close()

    def upload_model(self, model_path) -> None:  # interface parity, nothing to upload
        pass


def run_worker(args) -> None:
    """One training run in this process. Invoked by the orchestrator as a subprocess."""
    import pufferlib.pufferl as pufferl

    sys.path.insert(0, str(REPO_ROOT))
    import hazard_demo as hd

    hd.verify_in_repo_kernel()  # pinning inside the worker process as well

    mode = args.worker
    steps = args.steps
    seed = args.seed
    hazard_on = mode == "hazard_on"
    run_label = f"k4b_{mode}_s{seed}"
    run_dir = ART_DIR / "runs" / mode

    # load_config treats sys.argv[1:] as Hydra overrides; clear them and mutate directly.
    sys.argv = [sys.argv[0]]
    cfg = pufferl.load_config("puffer_drive")
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["tb"] = False
    cfg["run_name"] = run_label
    cfg["train"]["seed"] = seed
    cfg["train"]["total_timesteps"] = steps
    cfg["train"]["data_dir"] = str(run_dir)
    cfg["vec"]["seed"] = seed
    cfg["vec"]["num_envs"] = args.num_envs
    cfg["env"]["num_agents"] = args.num_agents
    cfg["env"]["hazard_enable"] = hazard_on
    if hazard_on:
        # Mix face: parked/clusters/cones are on by default; walkers opt in. The static
        # slot budget is sized from the planned-count highs (kernel fail-fasts if short).
        cfg["env"]["hazard_walkers_enable"] = True
        cfg["env"]["cosim_partner_slots"] = max(cfg["env"]["cosim_partner_slots"], hazard_slot_capacity(cfg["env"]))

    face = {
        "num_envs": cfg["vec"]["num_envs"],
        "num_agents": cfg["env"]["num_agents"],
        "bptt_horizon": cfg["train"]["bptt_horizon"],
        "batch_size": cfg["vec"]["num_envs"] * cfg["env"]["num_agents"] * cfg["train"]["bptt_horizon"],
        "minibatch_size": cfg["train"]["minibatch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "total_timesteps": cfg["train"]["total_timesteps"],
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
        "cosim_partner_slots": cfg["env"]["cosim_partner_slots"],
        "stop_sign_dwell_range": cfg["env"]["stop_sign_dwell_range"],
        "traffic_light_footprint_sweep": cfg["env"]["traffic_light_footprint_sweep"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "face.json").write_text(json.dumps(face, indent=2))
    print(f"[worker:{mode}] face: {json.dumps(face)}", flush=True)

    logger = EpochCsvLogger(run_dir / "metrics.csv", run_dir / "metrics_full.jsonl", run_label)
    try:
        pufferl.train("puffer_drive", args=cfg, logger=logger)
    finally:
        logger.close(None, False)


# ------------------------------------------------------------- orchestrator --


def batch_size_for(args) -> int:
    return args.num_envs * args.num_agents * 128  # bptt_horizon default (yaml)


def spawn_worker(mode: str, args, stdout_path: Path) -> tuple[int, float]:
    """Run one worker subprocess; returns (returncode, wall_seconds). No timeout set on
    the training process (edge discipline: never kill a short-timed training run)."""
    cmd = [
        str(python_bin()),
        str(REPO_ROOT / "scripts" / "k4b_smoke.py"),
        "--worker",
        mode,
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--num-envs",
        str(args.num_envs),
        "--num-agents",
        str(args.num_agents),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log(f"spawn worker {mode}: {' '.join(cmd[1:])}")
    t0 = time.time()
    with open(stdout_path, "w") as out:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    log(f"worker {mode} finished rc={proc.returncode} wall={wall:.1f}s")
    return proc.returncode, wall


def run_with_retry(mode: str, args, attempts: int = 2) -> float:
    """Single in-place retry on failure: wipe the run dir, rerun, same config."""
    run_dir = ART_DIR / "runs" / mode
    stdout_path = ART_DIR / f"{mode}_stdout.log"
    for attempt in range(1, attempts + 1):
        if run_dir.exists():
            shutil.rmtree(run_dir)
        rc, wall = spawn_worker(mode, args, stdout_path)
        if rc == 0 and (run_dir / "metrics.csv").exists():
            return wall
        log(f"worker {mode} attempt {attempt} FAILED rc={rc}; see {stdout_path}")
    raise RuntimeError(f"worker {mode} failed after {attempts} attempts; aborting (partial logs kept on disk)")


def read_run_metrics(mode: str) -> list[dict]:
    path = ART_DIR / "runs" / mode / "metrics.csv"
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("step")]
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                r[k] = None
    # Drop the terminal done_training log (step present, window env stats empty).
    return [r for r in rows if r.get("episode_return") is not None or r.get("collision_rate") is not None]


def probe(args) -> dict:
    """Short probe: measure steady-state SPS + fixed overhead for honest sizing."""
    probe_steps = max(args.probe_steps, 2 * batch_size_for(args))  # scheduler needs >=1 epoch
    if probe_steps != args.probe_steps:
        log(f"probe steps lifted to {probe_steps} (= 2 epochs of the smoke face; scheduler needs >= 1 epoch)")
    args.steps = probe_steps
    run_dir = ART_DIR / "runs" / "probe"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    t0 = time.time()
    run_with_retry("probe", args, attempts=2)
    wall = time.time() - t0
    rows = read_run_metrics("probe")
    if not rows:
        raise RuntimeError("probe produced no metrics rows; cannot size runs")
    sps_steady = rows[-1]["sps"] or 0.0
    active_span = rows[-1]["uptime_s"] or wall
    overhead = max(wall - active_span, 0.0)
    result = {
        "probe_steps": probe_steps,
        "probe_wall_s": round(wall, 1),
        "probe_active_s": round(active_span, 1),
        "sps_steady_last_window": round(sps_steady, 1),
        "fixed_overhead_s": round(overhead, 1),
        "note": "SPS = last logged window (steady state); overhead = env/map/policy/compile startup",
    }
    log(f"probe: {json.dumps(result)}")
    return result


def size_runs(args, probe_result: dict, budget_s: float = 720.0) -> dict:
    """Honest sizing: target <=12 min per run. 1e6 if SPS affords it, else 5e5/3e5."""
    sps = probe_result["sps_steady_last_window"]
    overhead = probe_result["fixed_overhead_s"]
    candidates = [1_000_000, 500_000, 300_000]
    min_steps = 2 * batch_size_for(args)  # below this the run has <2 PPO epochs: no signal
    chosen, est = None, None
    for c in candidates:
        est = overhead + c / sps if sps > 0 else float("inf")
        if est <= budget_s:
            chosen = c
            break
    if chosen is None:
        chosen = candidates[-1]
        est = overhead + chosen / sps if sps > 0 else float("inf")
    if chosen < min_steps:
        raise RuntimeError(
            f"even {chosen} steps is below 2 epochs of the smoke face (batch "
            f"{batch_size_for(args)}); shrink --num-envs/--num-agents and rerun"
        )
    decision = {
        "budget_per_run_s": budget_s,
        "sps_used": sps,
        "candidates_s": {str(c): round(overhead + c / sps, 1) for c in candidates},
        "chosen_steps": chosen,
        "estimated_time_s": round(est, 1),
        "epochs_per_run": chosen // batch_size_for(args),
        "rule": "largest of 1e6/5e5/3e5 fitting the 12-min budget; linear extrapolation of probe (incl. overhead)",
    }
    log(f"sizing: {json.dumps(decision)}")
    return decision


def first_segment(rows: list[dict], frac: float = 0.25) -> list[dict]:
    n = max(1, int(round(len(rows) * frac)))
    return rows[:n]


def last_segment(rows: list[dict], frac: float = 0.25) -> list[dict]:
    n = max(1, int(round(len(rows) * frac)))
    return rows[-n:]


def seg_stats(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "epochs": len(rows),
        "step_from": rows[0]["step"],
        "step_to": rows[-1]["step"],
        "episode_return": mean("episode_return"),
        "collision_rate": mean("collision_rate"),
        "red_light_violation_rate": mean("red_light_violation_rate"),
        "stop_sign_violation_rate": mean("stop_sign_violation_rate"),
        "offroad_rate": mean("offroad_rate"),
    }


def delta(last: dict, first: dict, key: str):
    a, b = last.get(key), first.get(key)
    if a is None or b is None:
        return None
    return {"first_seg": b, "last_seg": a, "abs_delta": round(a - b, 4)}


def sliding_mean(vals: list[float], window: int) -> np.ndarray:
    v = np.asarray(vals, dtype=float)
    if len(v) < 2 or window <= 1:
        return v
    w = min(window, len(v))
    kernel = np.ones(w) / w
    sm = np.convolve(v, kernel, mode="valid")
    out = np.empty_like(v)
    out[: len(sm)] = sm
    out[len(sm) :] = sm[-1]
    return out


def make_curves(runs: dict[str, list[dict]], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"hazard_off": "tab:blue", "hazard_on": "tab:red"}
    labels = {"hazard_off": "hazard OFF", "hazard_on": "hazard ON"}
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    ax_ret = fig.add_subplot(gs[0, :])
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        ret = [r["episode_return"] or float("nan") for r in rows]
        w = max(2, len(rows) // 4)
        ax_ret.plot(steps, ret, "o-", color=colors[mode], alpha=0.35, ms=4, lw=1)
        ax_ret.plot(steps, sliding_mean(ret, w), "-", color=colors[mode], lw=2.2, label=f"{labels[mode]} (slide~{w})")
    ax_ret.set_title("K4b PPO smoke: episode return (per-epoch mean, hazard on/off)")
    ax_ret.set_xlabel("agent steps")
    ax_ret.set_ylabel("episode_return")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend()

    ax_col = fig.add_subplot(gs[1, 0])
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        w = max(2, len(rows) // 4)
        col = [r["collision_rate"] or float("nan") for r in rows]
        ax_col.plot(steps, col, "o-", color=colors[mode], alpha=0.35, ms=4, lw=1)
        ax_col.plot(steps, sliding_mean(col, w), "-", color=colors[mode], lw=2.2, label=labels[mode])
    ax_col.set_title("collision_rate")
    ax_col.set_xlabel("agent steps")
    ax_col.grid(alpha=0.3)
    ax_col.legend()

    ax_vio = fig.add_subplot(gs[1, 1])
    styles = {"red_light_violation_rate": ("-", "red light"), "stop_sign_violation_rate": ("--", "stop sign")}
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        w = max(2, len(rows) // 4)
        for key, (ls, lab) in styles.items():
            vals = [r[key] or float("nan") for r in rows]
            ax_vio.plot(steps, vals, ls, color=colors[mode], alpha=0.35, lw=1.2)
            ax_vio.plot(steps, sliding_mean(vals, w), ls, color=colors[mode], lw=2.0,
                        label=f"{labels[mode]} · {lab} (slide~{w})")
    ax_vio.set_title("violation rates (red light / stop sign)")
    ax_vio.set_xlabel("agent steps")
    ax_vio.grid(alpha=0.3)
    ax_vio.legend()

    fig.suptitle("K4b smoke: learning-signal check (reward up / violations down = signal OK)", y=0.99)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_merged_metrics(runs: dict[str, list[dict]], out_csv: Path) -> None:
    fields = ["run"] + CSV_COLUMNS
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode, rows in runs.items():
            for r in rows:
                out = {"run": mode}
                for col in CSV_COLUMNS:
                    v = r.get(col)
                    out[col] = "" if v is None else v
                w.writerow(out)


def make_index(summary: dict, out_html: Path) -> None:
    def fmt(v):
        return "—" if v is None else f"{v}" if isinstance(v, str) else f"{v:,}" if isinstance(v, int) else f"{v:.4f}"

    rows_html = ""
    for mode in RUN_MODES:
        r = summary["runs"][mode]
        q = r["end_vs_start"]
        for key in ("episode_return", "collision_rate", "red_light_violation_rate", "stop_sign_violation_rate", "offroad_rate"):
            d = q.get(key) or {}
            rows_html += (
                f"<tr><td>{mode}</td><td>{key}</td>"
                f"<td>{fmt(d.get('first_seg'))}</td><td>{fmt(d.get('last_seg'))}</td>"
                f"<td>{fmt(d.get('abs_delta'))}</td></tr>\n"
            )
    face = summary["face"]
    sizing = summary["sizing"]
    cmd = summary["repro_command"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>K4b PPO smoke — hazard on/off learning curves</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem; color: #222; max-width: 1080px; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; margin: .6rem 0; font-size: .92rem; }}
td, th {{ border: 1px solid #ccc; padding: .3rem .6rem; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
code {{ background: #f4f4f4; padding: .5rem; display: block; white-space: pre-wrap; }}
.verdict {{ padding: .6rem 1rem; border-left: 4px solid #888; background: #fafafa; }}
</style></head><body>
<h1>K4b PPO smoke — hazard on/off learning curves</h1>
<p>同 seed 同配置双 PPO run，唯一差异 = hazard 面（off vs on 全开 mix）。
判读纪律：reward 上行 + 违规/碰撞下行 = 学习信号成立；本 smoke 只验证管线与信号，不调参刷分。</p>
<h2>curves.png</h2>
<img src="curves.png" alt="learning curves">
<h2>末段 vs 首段（前/后 25% epoch 均值）</h2>
<table><tr><th>run</th><th>metric</th><th>first segment</th><th>last segment</th><th>Δ</th></tr>
{rows_html}</table>
<h2>sizing / SPS</h2>
<table>
<tr><td>probe SPS (steady)</td><td>{fmt(sizing.get('sps_used'))}</td></tr>
<tr><td>chosen steps / run</td><td>{fmt(sizing.get('chosen_steps'))}</td></tr>
<tr><td>epochs / run (batch {face.get('batch_size')})</td><td>{fmt(sizing.get('epochs_per_run'))}</td></tr>
<tr><td>预算</td><td>{fmt(sizing.get('budget_per_run_s'))} s/run</td></tr>
</table>
<h2>verdict</h2>
<div class="verdict">{summary['signal']['verdict']}</div>
<h2>notes</h2>
<ul>{''.join(f'<li>{n}</li>' for n in summary.get('notes', []))}</ul>
<h2>repro（一条命令）</h2>
<code>{cmd}</code>
</body></html>
"""
    out_html.write_text(html)


def collect(args, probe_result, sizing, provenance) -> dict:
    runs = {mode: read_run_metrics(mode) for mode in RUN_MODES}
    for mode, rows in runs.items():
        if len(rows) < 3:
            raise RuntimeError(f"{mode}: only {len(rows)} epoch rows — not enough for signal judgement")

    make_curves(runs, ART_DIR / "curves.png")
    write_merged_metrics(runs, ART_DIR / "metrics.csv")

    # Faces as actually run (written by each worker) with a config-level fallback.
    faces = {}
    for mode in RUN_MODES:
        face_path = ART_DIR / "runs" / mode / "face.json"
        faces[mode] = json.loads(face_path.read_text()) if face_path.exists() else {"hazard_enable": mode == "hazard_on"}
    summary = {
        "kernel": provenance,
        "probe": probe_result,
        "sizing": sizing,
        "face": {
            "num_envs": args.num_envs,
            "num_agents": args.num_agents,
            "bptt_horizon": 128,
            "batch_size": batch_size_for(args),
            "minibatch_size": 65536,
            "learning_rate": 5e-4,
            "seed": args.seed,
            "simulation_mode": "gigaflow",
            "map_face": "default carla set (num_maps=8)",
            "reward": "default (progress shaping NOT included — ticket 06 main-train item)",
            "hazard_off": faces["hazard_off"],
            "hazard_on": faces["hazard_on"],
        },
        "runs": {},
    }
    for mode, rows in runs.items():
        stats_first = seg_stats(first_segment(rows, 0.25))
        stats_last = seg_stats(last_segment(rows, 0.25))
        sps_vals = [r["sps"] for r in rows if r.get("sps") is not None]
        summary["runs"][mode] = {
            "epochs_logged": len(rows),
            "steps_from": rows[0]["step"],
            "steps_to": rows[-1]["step"],
            "sps_mean": round(float(np.mean(sps_vals)), 1) if sps_vals else None,
            "first_segment": stats_first,
            "last_segment": stats_last,
            "end_vs_start": {
                k: delta(stats_last, stats_first, k)
                for k in ("episode_return", "collision_rate", "red_light_violation_rate",
                          "stop_sign_violation_rate", "offroad_rate")
            },
        }

    # Honest verdict: reward up AND violations/collisions not worse, per run.
    flags = {}
    for mode in RUN_MODES:
        d = summary["runs"][mode]["end_vs_start"]
        ret_up = (d["episode_return"] or {}).get("abs_delta", 0) > 0
        coll_down = (d["collision_rate"] or {}).get("abs_delta", 0) <= 0
        red_down = (d["red_light_violation_rate"] or {}).get("abs_delta", 0) <= 0
        stop_down = (d["stop_sign_violation_rate"] or {}).get("abs_delta", 0) <= 0
        flags[mode] = {
            "reward_up": ret_up,
            "collision_not_worse": coll_down,
            "red_light_not_worse": red_down,
            "stop_sign_not_worse": stop_down,
            "signal_ok": ret_up and coll_down and red_down and stop_down,
        }
    any_ok = any(f["signal_ok"] for f in flags.values())
    summary["signal"] = {
        "per_run": flags,
        "verdict": (
            "SIGNAL OK: at least one run shows reward up + collisions/violations not worse; "
            "read curves.png for the shape before over-trusting short-run noise."
            if any_ok
            else "NO CLEAR SIGNAL at this step budget: curves flat/down. Candidate causes: steps too few for "
                 "default reward (progress shaping absent — ticket 06 main-train item), lr/batch mismatch for "
                 "the small smoke face, or reward structure. Do NOT tune params to force a pretty curve; "
                 "record and hand to the main-train line."
        ),
    }
    summary["notes"] = [
        "smoke 用默认 reward；progress shaping 尚未加入（06 票主训项，不属本 smoke）",
        "artifacts 不入库（.gitignore artifacts/）；内核仅用本仓自编译 .so",
        "hazard_on 的 slot budget = planned-count highs（kernel fail-fast 校验）",
    ]
    summary["repro_command"] = (
        "PYTHONPATH=$PWD .venv/bin/python scripts/k4b_smoke.py "
        f"--steps {sizing['chosen_steps']} --seed {args.seed} --num-envs {args.num_envs} --num-agents {args.num_agents}"
    )
    (ART_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    make_index(summary, ART_DIR / "index.html")
    log(f"verdict: {summary['signal']['verdict']}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=1_000_000, help="target agent steps per run (sized down if SPS poor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-agents", type=int, default=128)
    p.add_argument("--probe-steps", type=int, default=50_000, help="probe length (lifted to 2 epochs minimum)")
    p.add_argument("--skip-probe", action="store_true", help="skip probe; reuse runs/probe metrics if present")
    p.add_argument("--collect-only", action="store_true", help="skip probe+runs, rebuild artifacts from run CSVs")
    p.add_argument("--worker", choices=RUN_MODES + ("probe",), help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        run_worker(args)
        return

    ART_DIR.mkdir(parents=True, exist_ok=True)
    provenance = verify_in_repo_kernel()
    log(f"kernel pinned: binding sha256={provenance['binding_sha256'][:12]}… _C sha256={provenance['advantage_sha256'][:12]}… gpu={provenance['gpu']}")

    if args.collect_only:
        probe_result, sizing = {}, {"chosen_steps": "n/a (collect-only)", "sps_used": None, "epochs_per_run": None, "budget_per_run_s": 720}
        summary = collect(args, probe_result, sizing, provenance)
        print(f"artifacts rebuilt under {ART_DIR}")
        return

    if args.skip_probe:
        prev = ART_DIR / "summary.json"
        if not prev.exists():
            raise RuntimeError("--skip-probe needs a previous artifacts/k4b-smoke/summary.json to reuse the probe result")
        probe_result = json.loads(prev.read_text())["probe"]
        log(f"probe reused from previous summary: {json.dumps(probe_result)}")
    else:
        probe_result = probe(args)
    sizing = size_runs(args, probe_result)
    args.steps = sizing["chosen_steps"]

    for mode in RUN_MODES:
        log(f"=== run {mode} (hazard {'ON (mix face)' if mode == 'hazard_on' else 'OFF'}) steps={args.steps} ===")
        run_with_retry(mode, args, attempts=2)

    summary = collect(args, probe_result, sizing, provenance)
    print(f"\nartifacts written under {ART_DIR}: curves.png metrics.csv summary.json index.html")


if __name__ == "__main__":
    main()
