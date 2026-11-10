#!/usr/bin/env python3
"""T2a main-train sweep: 3-arm reward_progress coefficient scan on a big training face.

Ticket 06 / STATE Next-2 main-train line. T1 (commit d5ce8ad) shipped the route-progress
shaping reward with the calibrated default [2.5e-4, 2.5e-3]; the K4b 1M-step smoke could
not settle whether the coefficient magnitude is right. This sweep runs THREE arms that
differ ONLY in the env knobs reward_progress_lo/hi, everything else identical (seed 42,
hazard off -- clean face first; hazard-on is a later round):

  A_progress0:  [0, 0]            control (pre-T1 reward stream, no RNG draw consumed)
  B_t1default:  [2.5e-4, 2.5e-3]  T1 calibrated default
  C_scale10x:   [2.5e-3, 2.5e-2]  10x magnitude probe

Engine: face probe (SPS + peak VRAM on 8x128 / 16x128 / 16x256) -> largest stable face ->
honest sizing (~2h per arm, <= 8h total hard cap, steps floored to epoch boundaries) ->
serial arms with per-epoch CSV appends + ~10%-progress checkpoints (pufferl auto-resume
via trainer_state.pt) -> final summary. All run state lives in state.json so a relaunch
skips completed work: completed arms are skipped, a crashed arm resumes from its
checkpoint (or restarts cleanly if it died before the first checkpoint), a dead arm is
recorded and the sweep continues to the next arm. The orchestrator is meant to be
launched detached (setsid + nohup) and survives the launching session.

Pipeline reuse note: worker subprocess + streaming logs + python_bin() (venv-symlink
trap: return the interpreter AS-IS, no .resolve()) all follow scripts/k4b_smoke.py.

Repro (the exact detached launch used for the overnight ignition):
  cd <repo root>
  setsid nohup env PYTHONPATH=$PWD .venv/bin/python \\
      scripts/t2_sweep.py --seed 42 --arm-budget-s 7200 --total-budget-s 28800 \\
      </dev/null >> artifacts/t2-maintrain/orchestrator_console.log 2>&1 &

Progress view:
  tail -f artifacts/t2-maintrain/orchestrator.log
  tail -2 artifacts/t2-maintrain/ckpt/A_progress0/metrics.csv

Harvest (T2b entry; safe to run mid-sweep, uses whatever rows exist):
  PYTHONPATH=$PWD .venv/bin/python scripts/t2_sweep.py --collect-only
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
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts" / "t2-maintrain"
CKPT_DIR = ART_DIR / "ckpt"
PROBE_DIR = ART_DIR / "probe"
STATE_PATH = ART_DIR / "state.json"
DECISION_PATH = ART_DIR / "decision.json"
ORCH_LOG = ART_DIR / "orchestrator.log"

# Ordered smallest -> largest; probing stops at the first unstable face (OOM risk only
# grows with face size, so larger faces after a failure are classified unstable-by-risk).
FACES = [(8, 128), (16, 128), (16, 256)]

# Arms in run order; lo/hi feed env.reward_progress_lo/hi verbatim (kernel validates
# finite + lo <= hi; [0, 0] disables the term AND consumes no RNG draw -> pre-T1 stream).
ARMS: dict[str, tuple[float, float]] = {
    "A_progress0": (0.0, 0.0),
    "B_t1default": (2.5e-4, 2.5e-3),
    "C_scale10x": (2.5e-3, 2.5e-2),
}
ARM_COLORS = {"A_progress0": "tab:blue", "B_t1default": "tab:orange", "C_scale10x": "tab:green"}

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
        # SPS/VRAM probe: face knobs matter, reward coefs are SPS-neutral -> arm B values.
        lo, hi = ARMS["B_t1default"]
        run_label = f"t2_probe_{args.num_envs}x{args.num_agents}_s{seed}"
        run_dir = PROBE_DIR / f"{args.num_envs}x{args.num_agents}"
        # Short probe needs no resumability: skip mid-run checkpoint saves entirely.
        checkpoint_interval = 10**9
    else:
        lo, hi = ARMS[args.worker]
        run_label = f"t2_{args.worker}_s{seed}"
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
    cfg["env"]["hazard_enable"] = False  # main-train line: clean face, hazard on later
    cfg["env"]["reward_progress_lo"] = lo
    cfg["env"]["reward_progress_hi"] = hi

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
        "reward_progress_lo": lo,
        "reward_progress_hi": hi,
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
        str(REPO_ROOT / "scripts" / "t2_sweep.py"),
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


def choose_face(state: dict) -> tuple[tuple[int, int], dict]:
    """Largest stable face by total agents (task contract: biggest face the GPU holds)."""
    stable = [
        (f, state["probes"][f"{f[0]}x{f[1]}"]) for f in FACES if state["probes"].get(f"{f[0]}x{f[1]}", {}).get("stable")
    ]
    if not stable:
        raise RuntimeError("no stable face found in probes; cannot size the sweep")
    face, probe = max(stable, key=lambda item: item[0][0] * item[0][1])
    decision = {
        "chosen": f"{face[0]}x{face[1]}",
        "sps": probe["sps"],
        "overhead_s": probe["overhead_s"],
        "peak_mem": probe.get("peak_mem", {}),
        "probe_ladder": {f"{f[0]}x{f[1]}": (state["probes"].get(f"{f[0]}x{f[1]}") or {}).get("stable") for f in FACES},
        "rule": "largest stable face (probe: 2 epochs, clean exit + finite losses + VRAM captured); "
                "unstable = OOM/crash; larger faces after a failure are unstable-by-risk",
    }
    log(f"face decision: {json.dumps(decision)}")
    return face, decision


def size_arms(face: tuple[int, int], probe: dict, state: dict, args) -> dict:
    """Honest sizing: ~arm-budget per arm, total sweep <= total-budget (hard cap).
    steps = floor-to-epoch((budget - overhead) * sps); shrink proportionally if the
    3-arm total would breach the cap."""
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
    total_est = probe_wall + 3 * (overhead + steps / sps)
    shrink_note = None
    if total_est > args.total_budget_s:
        avail_step_s = (args.total_budget_s - probe_wall) / 3 - overhead
        epochs = int(avail_step_s * sps) // batch
        steps = epochs * batch
        shrink_note = f"shrunk to fit {args.total_budget_s}s total cap (probe_wall {probe_wall:.0f}s)"
    if epochs < 10:
        raise RuntimeError(
            f"sizing collapsed to {epochs} epochs on face {envs}x{agents}; "
            "face too heavy for the budget on this GPU -- shrink FACES and rerun"
        )
    decision = {
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
        "est_total_s": round(probe_wall + 3 * (overhead + steps / sps), 1),
        "rule": "steps = floor-to-epoch((arm_budget - overhead) * sps); "
                "shrunk proportionally if 3-arm total breached the cap",
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


def run_arms(state: dict, args, sizing: dict) -> None:
    """Serial arms with resume: completed skip, crashed resume from checkpoint, dead
    recorded and the sweep continues. Two in-place attempts per arm (no timeout)."""
    envs, agents = [int(x) for x in sizing["face"].split("x")]
    steps_target = sizing["steps_per_arm"]
    batch = sizing["batch"]
    est_per_arm = sizing["est_per_arm_s"]
    for arm, (lo, hi) in ARMS.items():
        st = state["arms"].setdefault(arm, {"status": "pending", "attempts": 0})
        if st["status"] == "done":
            log(f"=== arm {arm} already DONE (last_step={st.get('last_step')}); skipping")
            continue
        if st["status"] == "dead":
            log(f"=== arm {arm} DEAD from a previous launch ({st.get('error')}); skipping "
                "(delete its ckpt dir + state entry to re-arm it)")
            continue
        run_dir = CKPT_DIR / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        note = prepare_arm_run(run_dir)
        st.update(status="running", coefs=[lo, hi], face=sizing["face"], steps_target=steps_target)
        save_state(state)
        remaining_before = sum(
            est_per_arm
            for a2, s2 in ARMS.items()
            if (state["arms"].get(a2) or {}).get("status") not in ("done", "dead")
        )
        log(f"=== arm {arm} start: progress_lo/hi=[{lo}, {hi}] steps={steps_target} "
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
        if not done:
            st.update(status="dead", error=error)
            log(f"arm {arm} DEAD after {st['attempts']} attempts: {error}; moving on")
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
        "  tail -f artifacts/t2-maintrain/orchestrator.log",
        "  wc -l artifacts/t2-maintrain/ckpt/*/metrics.csv   # 各臂 epoch 行数（含表头）",
        "  tail -2 artifacts/t2-maintrain/ckpt/A_progress0/metrics.csv",
        "  ps -p $(cat artifacts/t2-maintrain/state.json | python3 -c 'import json,sys;print(json.load(sys.stdin).get(\"pid\",0))') && nvidia-smi",
        "---- T2b 收割入口（全部臂 done/dead 后均可跑，用现存行收割）----",
        "  PYTHONPATH=$PWD .venv/bin/python scripts/t2_sweep.py --collect-only",
        "  -> artifacts/t2-maintrain/{metrics.csv(合并), curves.png, summary.json}",
    ]
    if sizing:
        lines.append(
            f"  （sizing: face {sizing['face']}, {sizing['steps_per_arm']} steps/arm "
            f"= {sizing['epochs_per_arm']} epochs, est_total {sizing['est_total_s']/3600:.1f}h）"
        )
    return lines


def final_summary(state: dict, sizing: dict | None) -> None:
    def fnum(v):
        return f"{v:.3f}" if isinstance(v, float) else "-"

    log("==== T2a FINAL SUMMARY ====")
    for arm in ARMS:
        st = state["arms"].get(arm) or {}
        rows = read_epoch_rows(CKPT_DIR / arm / "metrics.csv")
        if rows:
            last = rows[-1]
            log(
                f"arm {arm}: status={st.get('status')} rows={len(rows)} last_step={int(last['step'])} "
                f"ret={fnum(last['episode_return'])} coll={fnum(last['collision_rate'])} "
                f"red={fnum(last['red_light_violation_rate'])} stop={fnum(last['stop_sign_violation_rate'])}"
            )
        else:
            log(f"arm {arm}: status={st.get('status')} rows=0")
    state["phase"] = "done"
    save_state(state)
    for line in view_and_harvest_block(state, sizing):
        log(line)
    log("==== sweep complete ====")


def orchestrate(args) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    provenance = verify_in_repo_kernel()
    state = load_state()
    state.setdefault("probes", {})
    state.setdefault("arms", {})
    state.setdefault("sizing", None)
    state["pid"] = os.getpid()
    state["kernel"] = provenance
    if "started_at" not in state:
        state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    log(f"T2a orchestrator pid={state['pid']} kernel pinned: "
        f"binding sha256={provenance['binding_sha256'][:12]}… _C sha256={provenance['advantage_sha256'][:12]}… "
        f"gpu={provenance['gpu']} git={provenance['git_sha'][:12]}")

    sizing = state.get("sizing")
    if sizing:
        log(f"sizing reused from state: face={sizing['face']} steps/arm={sizing['steps_per_arm']} "
            f"epochs={sizing['epochs_per_arm']}")
        face = tuple(int(x) for x in sizing["face"].split("x"))
    else:
        probe_phase(state, args)
        face, face_decision = choose_face(state)
        probe = state["probes"][face_decision["chosen"]]
        sizing = size_arms(face, probe, state, args)
        sizing["face_decision"] = face_decision
        state["sizing"] = sizing
        save_state(state)
        DECISION_PATH.write_text(json.dumps(
            {"kernel": provenance, "probes": state["probes"], "face_decision": face_decision,
             "sizing": sizing, "arms": {a: {"coefs": list(c)} for a, c in ARMS.items()}},
            indent=2, ensure_ascii=False))
        log(f"decision snapshot written: {DECISION_PATH.name}")

    # Ignition block: ETA + how-to-watch + harvest entry, into the orchestrator log.
    arms_left = [a for a in ARMS if (state["arms"].get(a) or {}).get("status") != "done"]
    log("---- T2a ignition: " + f"{len(arms_left)} arm(s) to run ({', '.join(arms_left) or 'none'}); "
        + eta_line(time.time(), len(arms_left) * sizing["est_per_arm_s"]) + " ----")
    for line in view_and_harvest_block(state, sizing):
        log(line)

    run_arms(state, args, sizing)
    final_summary(state, sizing)


# ------------------------------------------------------------ collect (T2b) ---


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


def collect(args) -> None:
    """T2b harvest entry: merged metrics + 3-arm curves + summary from whatever rows
    exist (safe mid-sweep). Reuses state.json for face/sizing/kernel provenance."""
    provenance = verify_in_repo_kernel()
    state = load_state()
    runs = {}
    for arm in ARMS:
        rows = read_epoch_rows(CKPT_DIR / arm / "metrics.csv")
        if rows:
            runs[arm] = rows
    if not runs:
        raise RuntimeError("no arm metrics found under artifacts/t2-maintrain/ckpt/*/metrics.csv")
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
        "episode_return": "T2 sweep: episode return (per-epoch, progress coef scan)",
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
        "face": state.get("sizing"),
        "arms": {},
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
    log(f"collect done: {merged.name}, curves.png, summary.json under {ART_DIR}")


# --------------------------------------------------------------------- main ---


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arm-budget-s", type=int, default=7200, help="per-arm wall target (~2h)")
    p.add_argument("--total-budget-s", type=int, default=28800, help="hard cap on total sweep wall (8h)")
    p.add_argument("--collect-only", action="store_true", help="T2b harvest: rebuild merged metrics/curves/summary")
    p.add_argument("--worker", choices=list(ARMS) + ["probe"], help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-envs", type=int, default=8, help=argparse.SUPPRESS)
    p.add_argument("--num-agents", type=int, default=128, help=argparse.SUPPRESS)
    args = p.parse_args()

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
