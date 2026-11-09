#!/usr/bin/env python3
"""T1 route-progress shaping reward: A/B evidence runner + stream check (main-train first cut).

Ticket 06 main-train item: PufferDrive's default reward has no along-route progress term
(K4b plateau candidate cause). T1 adds a signed route-progress shaping reward:

  reward_progress = coef * ROUTE_PROGRESS  with  ROUTE_PROGRESS = v_signed * dt * cos(theta_lane) / REF_DIST

(forward along lane = positive, reverse = negative, normalized by ROUTE_PROGRESS_REF_DIST_M;
see pufferlib/ocean/drive/constants.h + drive.h). The per-agent coef is sampled from the
env knobs reward_progress_lo/hi; [0, 0] disables the term AND consumes no RNG draw, so it
keeps the pre-T1 stream bit-identical (escape hatch).

Two runs on the K4b face (4 envs x 128 agents, 1M steps, seed 42, gigaflow, carla maps,
hazard off), differing ONLY in the progress interval:
  - progress_off: reward_progress_lo/hi = [0, 0]   (= K4b reward, escape hatch face)
  - progress_on:  reward_progress_lo/hi = [--progress-lo, --progress-hi] (default calibrated)

NUM_REWARD_COEFS grows 18 -> 19, so the reward-conditioning observation (and the policy
input) gains one feature on BOTH arms: K4b checkpoints are incompatible (expected, both
arms are trained from scratch on the new obs size).

stream-check mode: deterministic fixed-length rollout under seeded random actions that
hashes rewards/terminals/truncations/observations. Use --save before changing the kernel
and --check after rebuilding: [0,0] must reproduce the pre-T1 stream bit-for-bit.

Repro:
  cd <repo root>
  PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/t1_progress_ab.py --steps 1000000 --seed 42
  PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/t1_progress_ab.py --mode stream-check --save <json>

Requires the in-repo advantage kernel (pufferlib._C) already built; the drive binding is
rebuilt by the kernel discipline below (NO_TRAIN=1 python setup.py build_ext --inplace).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts" / "t1-progress"
INTERPRETER_CANDIDATES = [
    REPO_ROOT.parent / "PufferDrive" / ".venv" / "bin" / "python",
    Path(sys.executable),
]
RUN_MODES = ("progress_off", "progress_on")
# Calibrated default sampling interval (see summary.json calibration block):
# per-step signal <= v_max*dt/ref = 20*0.1/10 = 0.2, so the reward per step stays
# <= 5e-4 = 2x the velocity anchor (2.5e-3 * dt * 1) and below lane_align's max.
PROGRESS_LO_DEFAULT = 2.5e-4
PROGRESS_HI_DEFAULT = 2.5e-3
STREAM_SEED = 7
STREAM_STEPS = 128  # 2 x scenario_length(64): every episode completes twice
STREAM_BPTT = 64
STREAM_AGENTS = 16
STREAM_NUM_ENVS = 2
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
    into miniconda, and resolving it strips the venv context (see k4b_smoke.py)."""
    for cand in INTERPRETER_CANDIDATES:
        if cand.exists():
            return cand
    raise RuntimeError(f"no usable interpreter among {INTERPRETER_CANDIDATES}")


def verify_in_repo_kernel() -> dict:
    """Kernel pinning discipline (same rule as the K-series demos): pufferlib and the
    drive binding .so must resolve to THIS repo; summary carries the provenance block."""
    sys.path.insert(0, str(REPO_ROOT))
    import hazard_demo as hd

    hd.verify_in_repo_kernel()
    prov = hd.kernel_provenance()
    c_so = REPO_ROOT / "pufferlib" / "_C.cpython-311-x86_64-linux-gnu.so"
    if not c_so.exists():
        raise RuntimeError(f"in-repo advantage kernel required: {c_so} missing")
    prov["advantage_so"] = str(c_so)
    prov["advantage_sha256"] = hashlib.sha256(c_so.read_bytes()).hexdigest()
    prov["advantage_built_by"] = "NO_OCEAN=1 CC=<gcc13> CXX=<g++13> python setup.py build_torch --inplace --force"
    import torch

    prov["torch"] = torch.__version__
    prov["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return prov


def binding_consts():
    sys.path.insert(0, str(REPO_ROOT))
    from pufferlib.ocean.drive import binding

    return binding


# ----------------------------------------------------------- stream check ----


def _stream_config(reward_conditioning: bool) -> dict:
    """Deterministic env-only face (no policy): Serial vec, seeded random actions."""
    from pufferlib.pufferl import load_config

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        cfg = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["tb"] = False
    cfg["vec"]["backend"] = "Serial"
    cfg["vec"]["num_envs"] = STREAM_NUM_ENVS
    cfg["vec"]["seed"] = STREAM_SEED
    updates = {
        "action_type": "discrete",  # seeded random discrete actions (same face as the rollout smoke test)
        "num_agents": STREAM_AGENTS,
        "min_agents_per_env": STREAM_AGENTS,
        "max_agents_per_env": STREAM_AGENTS,
        "num_maps": 2,
        "scenario_length": STREAM_BPTT,
        "stagger_first_episode": False,
        "reward_conditioning": bool(reward_conditioning),
        "hazard_enable": False,
    }
    # The progress knobs exist only on the T1 kernel; on the pre-T1 kernel absence IS the
    # [0,0] escape face, so the same script captures the baseline and the post-change check.
    from pufferlib.ocean.drive.drive import Drive

    if "reward_progress_lo" in inspect.signature(Drive.__init__).parameters:
        updates["reward_progress_lo"] = 0.0
        updates["reward_progress_hi"] = 0.0
    cfg["env"].update(updates)
    return cfg


def _coef_block_base(env, binding) -> int:
    """Feature index where the reward-coef conditioning block starts. write_reward_target_obs
    is called right after write_ego_obs (drive.h compute_observations): ego, then coefs+goals,
    then partners/lanes/boundaries/traffic/valid-counts."""
    return binding.EGO_FEATURES


def stream_rollout(reward_conditioning: bool) -> dict:
    """Roll STREAM_STEPS seeded random actions; hash the raw per-step buffers.

    Two hashes per face:
      rewards_sha256: rewards + terminals + truncations bytes (dynamics + reward stream)
      obs_sha256:     observation bytes. On the conditioning face the progress coef column
                      (the one feature T1 adds) is EXCLUDED, so pre/post hashes stay
                      comparable; the column itself is checked to be constant -1.0 (= the
                      [0,0] coef normalized), proving the extra slot carries no signal.
    """
    from pufferlib import unroll_nested_dict
    from pufferlib.pufferl import load_env

    binding = binding_consts()
    # On a TRUE pre-T1 binding the constant (and the column) does not exist: hash the
    # cond_on obs whole so a legacy baseline stays comparable to post-T1 spliced hashes.
    progress_idx = getattr(binding, "REWARD_COEF_PROGRESS", None)
    cfg = _stream_config(reward_conditioning)
    vecenv = load_env("puffer_drive", cfg)
    rng = np.random.default_rng(STREAM_SEED)
    h_rew = hashlib.sha256()
    h_obs = hashlib.sha256()
    n_coef = binding.NUM_REWARD_COEFS
    coef_base = None
    progress_col = None
    coef_col = []
    progress_component_sum = 0.0
    try:
        vecenv.reset(seed=STREAM_SEED)
        first_obs = None
        for step_i in range(STREAM_STEPS):
            actions = rng.integers(0, vecenv.single_action_space.n, size=vecenv.num_agents).astype(np.int32)
            obs, rewards, terminals, truncations, infos = vecenv.step(actions)
            obs = np.asarray(obs, dtype=np.float32)
            if first_obs is None:
                first_obs = obs.copy()
            rewards = np.asarray(rewards, dtype=np.float32)
            terminals = np.asarray(terminals)
            truncations = np.asarray(truncations)
            h_rew.update(rewards.tobytes())
            h_rew.update(terminals.tobytes())
            h_rew.update(truncations.tobytes())
            if reward_conditioning:
                if coef_base is None:
                    coef_base = _coef_block_base(vecenv.driver_env, binding)
                    progress_col = None if progress_idx is None else coef_base + progress_idx
                if progress_col is None:  # pre-T1 capture: no extra column to splice out
                    h_obs.update(obs.tobytes())
                else:
                    coef_col.append(obs[:, coef_base : coef_base + n_coef].copy())
                    # everything except the newly added progress coef column; np.delete
                    # yields the contiguous (rows, n-1) layout so the byte stream stays
                    # comparable with a pre-T1 full-row hash (slicing tobytes() would
                    # concatenate per-row segments and hash a different arrangement)
                    h_obs.update(np.delete(obs, progress_col, axis=1).tobytes())
            else:
                h_obs.update(obs.tobytes())
            for info in infos:
                for k, v in unroll_nested_dict(info):
                    if k == "environment/reward_components/progress" and isinstance(v, (int, float)):
                        progress_component_sum += float(v)
        last_obs = np.asarray(obs, dtype=np.float32).copy()
    finally:
        vecenv.close()
    out = {
        "reward_conditioning": bool(reward_conditioning),
        "num_reward_coefs": int(n_coef),
        "rewards_sha256": h_rew.hexdigest(),
        "obs_sha256": h_obs.hexdigest(),
        "steps": STREAM_STEPS,
        "obs_len": int(first_obs.shape[1]),
        "progress_component_sum_over_episode_infos": round(progress_component_sum, 6),
        "note": "hashes over per-step float32 buffers; conditioning face excludes the T1 coef "
        "column from obs_sha256 so pre/post are comparable (the column is checked separately)",
    }
    if reward_conditioning and coef_col:
        col = np.concatenate(coef_col, axis=0)
        out["progress_coef_col_min"] = float(col[:, binding.REWARD_COEF_PROGRESS].min())
        out["progress_coef_col_max"] = float(col[:, binding.REWARD_COEF_PROGRESS].max())
    return out


def run_stream_check(save: Path | None, check: Path | None) -> dict:
    binding = binding_consts()
    log(f"stream-check kernel: NUM_REWARD_COEFS={binding.NUM_REWARD_COEFS}")
    result = {
        "mode": "stream-check",
        "faces": {
            "cond_off": stream_rollout(reward_conditioning=False),
            "cond_on": stream_rollout(reward_conditioning=True),
        },
    }
    faces = result["faces"]
    invariants = {
        # conditioning adds obs features only; dynamics/rewards must be bit-identical
        "cond_on_rewards_match_cond_off": faces["cond_off"]["rewards_sha256"] == faces["cond_on"]["rewards_sha256"],
        # escape hatch: the progress component must contribute exactly zero
        "progress_component_zero_on_escape_face": faces["cond_off"]["progress_component_sum_over_episode_infos"] == 0.0
        and faces["cond_on"]["progress_component_sum_over_episode_infos"] == 0.0,
    }
    if "progress_coef_col_min" in faces["cond_on"]:
        invariants["extra_coef_column_is_constant_minus_one"] = (
            faces["cond_on"]["progress_coef_col_min"] == -1.0 and faces["cond_on"]["progress_coef_col_max"] == -1.0
        )
    result["invariants"] = invariants
    if check:
        base = json.loads(Path(check).read_text())
        base_faces = base["faces"]
        checks = {
            "cond_off_rewards_bit_identical": faces["cond_off"]["rewards_sha256"] == base_faces["cond_off"]["rewards_sha256"],
            "cond_off_obs_bit_identical": faces["cond_off"]["obs_sha256"] == base_faces["cond_off"]["obs_sha256"],
            "cond_on_rewards_bit_identical": faces["cond_on"]["rewards_sha256"] == base_faces["cond_on"]["rewards_sha256"],
            "cond_on_legacy_obs_bit_identical": faces["cond_on"]["obs_sha256"] == base_faces["cond_on"]["obs_sha256"],
            "obs_len_grew_by_exactly_one": faces["cond_on"]["obs_len"] == base_faces["cond_on"]["obs_len"] + 1,
        }
        result["checks"] = checks
        result["baseline_file"] = str(check)
        ok = all(checks.values()) and all(invariants.values())
        log(f"stream-check verdict: {'PASS — [0,0] stream bit-identical to baseline' if ok else 'FAIL — stream drifted'}")
        log(f"details: {json.dumps({**checks, **invariants})}")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(result, indent=2))
        log(f"stream baseline saved -> {save}")
    return result


# ------------------------------------------------------------- A/B runner ----


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
    partial runs stay on disk (edge-training-log discipline). Same as k4b_smoke."""

    def __init__(self, csv_path: Path, jsonl_path: Path, run_label: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.run_label = run_label
        self.run_id = run_label
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

        def fnum(v, spec=".0f"):
            return format(v, spec) if isinstance(v, float) else "-"

        print(
            f"[{self.run_label}] step={fnum(row['step'])} epoch={fnum(row['epoch'])} "
            f"sps={fnum(row['sps'])} ret={fnum(row['episode_return'], '.3f')} "
            f"coll={fnum(row['collision_rate'], '.3f')} goals={fnum(row['num_goals_reached'], '.4f')} "
            f"prog={fnum(row['reward_progress_component'], '.5f')}",
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

    hd.verify_in_repo_kernel()

    mode = args.worker
    progress_on = mode == "progress_on"
    lo, hi = (args.progress_lo, args.progress_hi) if progress_on else (0.0, 0.0)
    run_label = f"t1_{mode}_s{args.seed}"
    run_dir = ART_DIR / "runs" / mode

    # load_config treats sys.argv[1:] as Hydra overrides; clear them and mutate directly.
    sys.argv = [sys.argv[0]]
    cfg = pufferl.load_config("puffer_drive")
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["tb"] = False
    cfg["run_name"] = run_label
    cfg["train"]["seed"] = args.seed
    cfg["train"]["total_timesteps"] = args.steps
    cfg["train"]["data_dir"] = str(run_dir)
    cfg["vec"]["seed"] = args.seed
    cfg["vec"]["num_envs"] = args.num_envs
    cfg["env"]["num_agents"] = args.num_agents
    cfg["env"]["hazard_enable"] = False  # K4b hazard_off face for both arms
    cfg["env"]["reward_progress_lo"] = lo
    cfg["env"]["reward_progress_hi"] = hi

    binding = binding_consts()
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
        "reward_conditioning": cfg["env"]["reward_conditioning"],
        "reward_randomization": cfg["env"]["reward_randomization"],
        "reward_progress_lo": lo,
        "reward_progress_hi": hi,
        "num_reward_coefs": binding.NUM_REWARD_COEFS,
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


def batch_size_for(args) -> int:
    return args.num_envs * args.num_agents * 128  # bptt_horizon default (yaml)


def spawn_worker(mode: str, args, stdout_path: Path) -> tuple[int, float]:
    cmd = [
        str(python_bin()),
        str(REPO_ROOT / "scripts" / "t1_progress_ab.py"),
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
        "--progress-lo",
        str(args.progress_lo),
        "--progress-hi",
        str(args.progress_hi),
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
    """Single in-place retry on failure: wipe the run dir, rerun, same config.
    No timeout on the training process (edge discipline: never kill a short run)."""
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
    return [r for r in rows if r.get("episode_return") is not None or r.get("collision_rate") is not None]


# ------------------------------------------------------------ A/B artifacts --


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
        "num_goals_reached": mean("num_goals_reached"),
        "offroad_rate": mean("offroad_rate"),
        "reward_progress_component": mean("reward_progress_component"),
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

    colors = {"progress_off": "tab:blue", "progress_on": "tab:red"}
    labels = {"progress_off": "A progress [0,0] (=K4b)", "progress_on": "B progress on"}
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    ax_ret = fig.add_subplot(gs[0, :])
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        ret = [r["episode_return"] or float("nan") for r in rows]
        w = max(2, len(rows) // 4)
        ax_ret.plot(steps, ret, "o-", color=colors[mode], alpha=0.35, ms=4, lw=1)
        ax_ret.plot(steps, sliding_mean(ret, w), "-", color=colors[mode], lw=2.2, label=f"{labels[mode]} (slide~{w})")
    ax_ret.set_title("T1 A/B: episode return (per-epoch mean, same axis)")
    ax_ret.set_xlabel("agent steps")
    ax_ret.set_ylabel("episode_return")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend()

    ax_goals = fig.add_subplot(gs[1, 0])
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        w = max(2, len(rows) // 4)
        vals = [r["num_goals_reached"] or float("nan") for r in rows]
        ax_goals.plot(steps, vals, "o-", color=colors[mode], alpha=0.35, ms=4, lw=1)
        ax_goals.plot(steps, sliding_mean(vals, w), "-", color=colors[mode], lw=2.2, label=labels[mode])
    ax_goals.set_title("goals_reached")
    ax_goals.set_xlabel("agent steps")
    ax_goals.grid(alpha=0.3)
    ax_goals.legend()

    ax_col = fig.add_subplot(gs[1, 1])
    for mode, rows in runs.items():
        steps = [r["step"] for r in rows]
        w = max(2, len(rows) // 4)
        vals = [r["collision_rate"] or float("nan") for r in rows]
        ax_col.plot(steps, vals, "o-", color=colors[mode], alpha=0.35, ms=4, lw=1)
        ax_col.plot(steps, sliding_mean(vals, w), "-", color=colors[mode], lw=2.2, label=labels[mode])
    ax_col.set_title("collision_rate")
    ax_col.set_xlabel("agent steps")
    ax_col.grid(alpha=0.3)
    ax_col.legend()

    fig.suptitle("T1 route-progress shaping: A=[0,0] vs B=on (only the progress interval differs)", y=0.99)
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
        for key in ("episode_return", "collision_rate", "num_goals_reached", "offroad_rate", "reward_progress_component"):
            d = q.get(key) or {}
            rows_html += (
                f"<tr><td>{mode}</td><td>{key}</td>"
                f"<td>{fmt(d.get('first_seg'))}</td><td>{fmt(d.get('last_seg'))}</td>"
                f"<td>{fmt(d.get('abs_delta'))}</td></tr>\n"
            )
    cross = summary.get("cross_arm_compare", {})
    cross_html = ""
    for key, d in cross.items():
        cross_html += f"<tr><td>{key}</td><td>{fmt(d.get('progress_off'))}</td><td>{fmt(d.get('progress_on'))}</td><td>{fmt(d.get('delta_b_minus_a'))}</td></tr>\n"
    calib = summary.get("calibration", {})
    stream = summary.get("stream_check", {})
    face = summary["face"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>T1 route-progress shaping A/B</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem; color: #222; max-width: 1080px; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; margin: .6rem 0; font-size: .92rem; }}
td, th {{ border: 1px solid #ccc; padding: .3rem .6rem; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
code {{ background: #f4f4f4; padding: .5rem; display: block; white-space: pre-wrap; }}
.verdict {{ padding: .6rem 1rem; border-left: 4px solid #888; background: #fafafa; }}
</style></head><body>
<h1>T1 route-progress shaping A/B（progress [0,0] vs on）</h1>
<p>K4b 同 face（4 env × 128 agents、1M 步、seed 42、hazard off），两臂唯一差异 = reward_progress_lo/hi。
判读纪律：B 在 return/goals/collision 至少一项显著改善且其他不显著恶化才算正向；不调参刷图。</p>
<h2>curves.png</h2>
<img src="curves.png" alt="A/B curves">
<h2>区间标定</h2>
<table>
<tr><th>量</th><th>值</th></tr>
<tr><td>信号/步上限</td><td>{fmt(calib.get('signal_per_step_max'))} (= v_max·dt/ref = {fmt(calib.get('v_max_mps'))}·{fmt(calib.get('dt_s'))}/{fmt(calib.get('ref_dist_m'))})</td></tr>
<tr><td>采样区间 [lo, hi]</td><td>[{fmt(calib.get('progress_lo'))}, {fmt(calib.get('progress_hi'))}]</td></tr>
<tr><td>奖励/步上限（B 区间）</td><td>{fmt(calib.get('reward_per_step_max'))}（velocity 锚 = {fmt(calib.get('velocity_anchor_per_step'))}）</td></tr>
<tr><td>标定说明</td><td>{calib.get('rationale', '')}</td></tr>
</table>
<h2>末段 vs 首段（前/后 25% epoch 均值，分臂）</h2>
<table><tr><th>run</th><th>metric</th><th>first segment</th><th>last segment</th><th>Δ</th></tr>
{rows_html}</table>
<h2>跨臂对比（last segment / 全程均值，B − A）</h2>
<table><tr><th>metric</th><th>A progress_off</th><th>B progress_on</th><th>Δ (B−A)</th></tr>
{cross_html}</table>
<h2>stream check（[0,0] 逃生门 = pre-T1 流逐位一致）</h2>
<pre>{json.dumps(stream, indent=2, ensure_ascii=False)}</pre>
<h2>face / 观测维度</h2>
<table>
<tr><td>batch</td><td>{fmt(face.get('batch_size'))}</td></tr>
<tr><td>NUM_REWARD_COEFS</td><td>{fmt(summary.get('num_reward_coefs'))}（pre-T1 = 18；reward-conditioning 观测向量 +1 维，K4b checkpoint 不兼容，两臂均从头训练）</td></tr>
</table>
<h2>verdict</h2>
<div class="verdict">{summary['signal']['verdict']}</div>
<h2>notes</h2>
<ul>{''.join(f'<li>{n}</li>' for n in summary.get('notes', []))}</ul>
<h2>repro（一条命令）</h2>
<code>{summary['repro_command']}</code>
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)


def cross_arm_stats(runs: dict[str, list[dict]]) -> dict:
    """B-vs-A comparison on matched windows: last segment and whole-run means."""
    out = {}
    for key in ("episode_return", "collision_rate", "num_goals_reached", "offroad_rate"):
        whole = {}
        for mode, rows in runs.items():
            vals = [r[key] for r in rows if r.get(key) is not None]
            whole[mode] = round(float(np.mean(vals)), 4) if vals else None
        last = {}
        for mode, rows in runs.items():
            tail = last_segment(rows)
            vals = [r[key] for r in tail if r.get(key) is not None]
            last[mode] = round(float(np.mean(vals)), 4) if vals else None
        a_last, b_last = last.get("progress_off"), last.get("progress_on")
        out[key] = {
            "progress_off": whole.get("progress_off"),
            "progress_on": whole.get("progress_on"),
            "last_seg_off": a_last,
            "last_seg_on": b_last,
            "delta_b_minus_a": round(b_last - a_last, 4) if (a_last is not None and b_last is not None) else None,
        }
    return out


def collect(args, sizing, provenance, stream_result) -> dict:
    runs = {mode: read_run_metrics(mode) for mode in RUN_MODES}
    for mode, rows in runs.items():
        if len(rows) < 3:
            raise RuntimeError(f"{mode}: only {len(rows)} epoch rows — not enough for signal judgement")

    make_curves(runs, ART_DIR / "curves.png")
    write_merged_metrics(runs, ART_DIR / "metrics.csv")

    faces = {}
    for mode in RUN_MODES:
        face_path = ART_DIR / "runs" / mode / "face.json"
        faces[mode] = json.loads(face_path.read_text()) if face_path.exists() else {}

    binding = binding_consts()
    dt_s, v_max, ref = 0.1, 20.0, 10.0
    signal_max = v_max * dt_s / ref
    summary = {
        "kernel": provenance,
        "ticket": "T1 route progress shaping reward (06 main-train first cut, K4b plateau fix candidate)",
        "calibration": {
            "dt_s": dt_s,
            "v_max_mps": v_max,
            "ref_dist_m": ref,
            "signal_per_step_max": round(signal_max, 6),
            "progress_lo": args.progress_lo,
            "progress_hi": args.progress_hi,
            "reward_per_step_max": round(args.progress_hi * signal_max, 8),
            "velocity_anchor_per_step": 2.5e-3 * dt_s,
            "rationale": "signal = v_signed*dt*cos(theta)/ref_dist (10 m). hi*0.2 = "
            f"{args.progress_hi * signal_max:.2e} per step = 2x velocity anchor (2.5e-3*dt), "
            "below lane_align max (2.5e-3/step); per-episode (16 s, ~100 m route) "
            f"~ coef*10 ≈ [{args.progress_lo * 10:.1e}, {args.progress_hi * 10:.1e}] — dense and visible, "
            "cannot drown goal (±1) or collision (≤3) events.",
        },
        "stream_check": stream_result,
        "num_reward_coefs": binding.NUM_REWARD_COEFS,
        "obs_dim_note": "NUM_REWARD_COEFS 18→19: the reward-conditioning observation vector and "
        "policy input gain one feature on BOTH arms; K4b checkpoints are incompatible (expected).",
        "face": {
            "num_envs": args.num_envs,
            "num_agents": args.num_agents,
            "bptt_horizon": 128,
            "batch_size": batch_size_for(args),
            "minibatch_size": 65536,
            "learning_rate": 5e-4,
            "seed": args.seed,
            "simulation_mode": "gigaflow",
            "map_face": "default carla set (num_maps=8), hazard off (K4b hazard_off face)",
            "progress_off": faces.get("progress_off"),
            "progress_on": faces.get("progress_on"),
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
                for k in ("episode_return", "collision_rate", "num_goals_reached", "offroad_rate", "reward_progress_component")
            },
        }

    summary["cross_arm_compare"] = cross_arm_stats(runs)

    # Honest verdict (判读纪律): B better in >=1 of return/goals/collision AND others not worse.
    cross = summary["cross_arm_compare"]
    d = {k: v["delta_b_minus_a"] for k, v in cross.items()}
    improves_return = (d.get("episode_return") or 0) > 0
    improves_goals = (d.get("num_goals_reached") or 0) > 0
    improves_collision = (d.get("collision_rate") or 0) < 0
    not_worse_return = (d.get("episode_return") or 0) >= 0
    not_worse_goals = (d.get("num_goals_reached") or 0) >= 0
    not_worse_collision = (d.get("collision_rate") or 0) <= 0
    any_improve = improves_return or improves_goals or improves_collision
    none_worse = not_worse_return and not_worse_goals and not_worse_collision
    b_active = (summary["runs"]["progress_on"]["end_vs_start"]["reward_progress_component"] or {}).get(
        "abs_delta"
    )
    summary["signal"] = {
        "deltas_b_minus_a_last_seg": d,
        "b_reward_component_present": b_active is not None and b_active != 0,
        "verdict": (
            "SIGNAL OK: B improves >=1 of return/goals/collision with none worse."
            if any_improve and none_worse
            else "NO BETTER: B fails the improve-without-regression bar. Candidate causes: interval "
            "too small to register (淹没) / signal noise / 1M steps too few. Recorded as-is; do NOT "
            "tune params to force a pretty curve."
        ),
    }
    summary["notes"] = [
        "A 臂 = [0,0] 逃生门面（reward 流 = pre-T1，见 stream_check）；B 臂 = 标定默认区间",
        "artifacts 不入库（.gitignore artifacts/）；内核仅用本仓自编译 .so（provenance 见 kernel 块）",
        "NUM_REWARD_COEFS 18→19 → 观测/策略输入 +1 维（两臂同维，K4b checkpoint 不兼容属预期）",
    ]
    summary["repro_command"] = (
        "PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/t1_progress_ab.py "
        f"--steps {sizing['chosen_steps']} --seed {args.seed} --num-envs {args.num_envs} "
        f"--num-agents {args.num_agents} --progress-lo {args.progress_lo} --progress-hi {args.progress_hi}"
    )
    (ART_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    make_index(summary, ART_DIR / "index.html")
    log(f"verdict: {summary['signal']['verdict']}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("ab", "stream-check"), default="ab")
    p.add_argument("--steps", type=int, default=1_000_000, help="agent steps per arm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-agents", type=int, default=128)
    p.add_argument("--progress-lo", type=float, default=PROGRESS_LO_DEFAULT)
    p.add_argument("--progress-hi", type=float, default=PROGRESS_HI_DEFAULT)
    p.add_argument("--save", type=Path, help="stream-check: save baseline hashes to this json")
    p.add_argument("--check", type=Path, help="stream-check: compare against a saved baseline json")
    p.add_argument("--collect-only", action="store_true", help="skip runs, rebuild artifacts from run CSVs")
    p.add_argument("--worker", choices=RUN_MODES, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        run_worker(args)
        return

    ART_DIR.mkdir(parents=True, exist_ok=True)
    provenance = verify_in_repo_kernel()
    log(
        f"kernel pinned: binding sha256={provenance['binding_sha256'][:12]}… "
        f"_C sha256={provenance['advantage_sha256'][:12]}… gpu={provenance['gpu']}"
    )

    if args.mode == "stream-check":
        run_stream_check(save=args.save, check=args.check)
        return

    if args.collect_only:
        sizing = {"chosen_steps": "n/a (collect-only)"}
        stream_result = {}
        stream_path = ART_DIR / "stream_check.json"
        if stream_path.exists():
            stream_result = json.loads(stream_path.read_text())
        collect(args, sizing, provenance, stream_result)
        print(f"artifacts rebuilt under {ART_DIR}")
        return

    # 1) stream check: persist the post-change [0,0] hashes and, when the pre-T1 baseline
    #    exists, merge the bit-identical verdict into the summary's stream-gate block.
    stream_path = ART_DIR / "stream_check.json"
    baseline_path = ART_DIR / "stream_baseline.json"
    stream_result = run_stream_check(save=stream_path, check=baseline_path if baseline_path.exists() else None)

    # 2) A/B runs (fixed 1M-step face, no probe needed: K4b sizing known ~66s/run)
    sizing = {"chosen_steps": args.steps}
    for mode in RUN_MODES:
        lo_hi = f"[{args.progress_lo}, {args.progress_hi}]" if mode == "progress_on" else "[0, 0]"
        log(f"=== run {mode} (progress {lo_hi}) steps={args.steps} ===")
        run_with_retry(mode, args, attempts=2)

    summary = collect(args, sizing, provenance, stream_result)
    print(f"\nartifacts written under {ART_DIR}: curves.png metrics.csv summary.json index.html")


if __name__ == "__main__":
    main()
