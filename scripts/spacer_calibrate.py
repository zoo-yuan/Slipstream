#!/usr/bin/env python3
"""SPACeR α calibration (review 必改2, hard gate).

Measures the reference policy's −logπ_ref(a|s) distribution on its own rollout
(the checkpoint loaded as the acting policy, sampling exactly like training
does) on a short CPU rollout: Serial vectorization, small face, a few hundred
steps per agent. From the measured mean it recommends three α tiers such that
α·mean(−logπ_ref) lands inside the R-base progress band [1.25e-3, 1.25e-2] —
the proposal's α values assumed −logπ_ref ≈ 0.2-1.0 nats without evidence, and
a near-uniform ref (ln 12 ≈ 2.485 nats) would make those α dwarf the whole
progress band (review F1 + proposal self-assessment).

Also reports the fraction of transitions above the 3.0-nat floor (the F1
poisoning exposure) and the implied per-step anchor reward of the current
S-arm α values, so the S first run can be backfilled from data, not assumptions.

CPU ONLY (the training GPUs are busy); the acting policy never trains here.

Output: artifacts/t6-spacer/calibration.json

Usage:
  PYTHONPATH=$PWD .venv/bin/python scripts/spacer_calibrate.py \
      --ref artifacts/t3-R4x/ckpt/R_ent5_off23/models/model_puffer_drive_007424.pt

Note: the R champion checkpoint lives in the main checkout's artifacts/ tree
(artifacts are not committed); in a worktree the default resolves through the
main repo root automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(REPO_ROOT, "artifacts", "t6-spacer")
OUT_PATH = os.path.join(ART_DIR, "calibration.json")

DEFAULT_REF_REL = "artifacts/t3-R4x/ckpt/R_ent5_off23/models/model_puffer_drive_007424.pt"
# R champion base recipe progress band (proposal §3 / D6): per-step dominant reward.
PROGRESS_BAND = (1.25e-3, 1.25e-2)
# Current S-arm proposal α values (pending this calibration's backfill).
S_ARM_ALPHAS = {"S1_spacer": 0.005, "S2_spacer_hi": 0.02, "S3_spacer_lo": 0.001}
SPACER_LOGPROB_FLOOR = 3.0  # yaml default (review 必改3)
NUM_ACTIONS = 12  # JERK_LONG 4 × JERK_LAT 3 (constants.h:143-146)


def main_repo_root() -> str | None:
    """Worktrees share .git with the main checkout; artifacts live there only."""
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None
    root = os.path.dirname(os.path.abspath(common))
    return root if os.path.isdir(root) else None


def resolve_ref(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [os.path.join(REPO_ROOT, DEFAULT_REF_REL)]
    if not explicit:
        main = main_repo_root()
        if main:
            candidates.append(os.path.join(main, DEFAULT_REF_REL))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"ref checkpoint not found; tried: {candidates}")


def build_config(num_envs: int, num_agents: int, seed: int):
    import pufferlib.pufferl as pufferl

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        cfg = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    batch = num_envs * num_agents * int(cfg["train"]["bptt_horizon"])
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["tb"] = False
    cfg["run_name"] = "spacer_calibrate"
    cfg["train"].update(
        {
            "device": "cpu",  # CPU only: the training GPUs are busy
            "precision": "float32",
            "compile": False,
            "seed": seed,
            "total_timesteps": 10 * batch,  # scheduler needs >= 1 epoch; never trains
            "batch_size": batch,
            "minibatch_size": batch,
            "max_minibatch_size": batch,
            "data_dir": os.path.join(ART_DIR, "calib_run"),
            "checkpoint_interval": 10**9,  # never checkpoints (no train() call anyway)
            # SPACeR off: the acting policy IS the ref; we only read its logprobs.
            "spacer_kl_coef": 0.0,
            "spacer_reward_coef": 0.0,
        }
    )
    cfg["vec"].update({"backend": "Serial", "num_envs": num_envs, "num_workers": num_envs, "seed": seed})
    cfg["env"]["num_agents"] = num_agents
    cfg = pufferl.normalize_puffer_drive_config(cfg, "training")
    pufferl.validate_puffer_drive_config(cfg, "training")
    return cfg, batch


def measure(ref_path: str, num_envs: int, num_agents: int, rollouts: int, seed: int) -> dict:
    import pufferlib.pufferl as pufferl
    from pufferlib.config_schema import validate_puffer_drive_resources

    cfg, batch = build_config(num_envs, num_agents, seed)
    validate_puffer_drive_resources(cfg, "training")

    torch.manual_seed(seed)
    vecenv = pufferl.load_env("puffer_drive", cfg, seed=cfg["vec"]["seed"])
    policy = pufferl.load_policy(cfg, vecenv, "puffer_drive")
    # The checkpoint becomes the acting policy (strict load; architecture
    # mismatch fails fast, mirroring the anchor/mixer ref contract).
    ref_state = pufferl.clean_policy_state_dict(torch.load(ref_path, map_location="cpu"))
    policy.load_state_dict(ref_state)

    train_config = dict(**cfg["train"], env="puffer_drive", eval={}, run_name="spacer_calibrate")
    trainer = pufferl.PuffeRL(train_config, vecenv, policy)
    horizon = int(cfg["train"]["bptt_horizon"])
    neg_logp_batches = []
    try:
        for _ in range(rollouts):
            trainer.evaluate()  # rollout only: logprobs = logπ_ref(sampled a)
            neg_logp_batches.append(-trainer.logprobs[trainer.masks.bool()].clone())
    finally:
        trainer.utilization.stop()
        vecenv.close()

    neg = torch.cat(neg_logp_batches).double().numpy()
    return {
        "samples": int(neg.size),
        "env_steps_per_agent": rollouts * horizon,
        "neg_logp_nats": {
            "mean": float(np.mean(neg)),
            "std": float(np.std(neg)),
            "p50": float(np.percentile(neg, 50)),
            "p90": float(np.percentile(neg, 90)),
            "max": float(np.max(neg)),
            "min": float(np.min(neg)),
        },
        "frac_above_floor": {
            "floor_nats": SPACER_LOGPROB_FLOOR,
            "fraction": float(np.mean(neg > SPACER_LOGPROB_FLOOR)),
        },
        "face": {
            "vec_backend": "Serial",
            "num_envs": num_envs,
            "num_agents_per_env": num_agents,
            "bptt_horizon": horizon,
            "batch_size": batch,
            "rollouts": rollouts,
            "seed": seed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=None, help=f"reference ckpt (default: {DEFAULT_REF_REL})")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=3, help="buffer fills (each = bptt_horizon env steps/agent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    ref_path = resolve_ref(args.ref)
    result = measure(ref_path, args.num_envs, args.num_agents, args.rollouts, args.seed)

    mean = result["neg_logp_nats"]["mean"]
    lo, hi = PROGRESS_BAND
    alpha_lo, alpha_hi = lo / mean, hi / mean
    alpha_mid = math.sqrt(alpha_lo * alpha_hi)
    result["uniform_ref_baseline_nats"] = math.log(NUM_ACTIONS)
    result["progress_band"] = list(PROGRESS_BAND)
    result["alpha_recommendation"] = {
        "rule": "alpha * mean(neg_logp_ref) within progress band [lo, hi] of the R base",
        "alpha_lo": alpha_lo,
        "alpha_mid": alpha_mid,
        "alpha_hi": alpha_hi,
        "current_s_arm_alphas": {
            arm: {"alpha": alpha, "implied_per_step_anchor_reward": alpha * mean, "in_band": bool(lo <= alpha * mean <= hi)}
            for arm, alpha in S_ARM_ALPHAS.items()
        },
        "note": (
            "α depends on the base recipe (review 必改5): if the E series changes the task "
            "definition (goal_speed_mode), rerun this calibration against the E champion ckpt "
            "before reusing these α values."
        ),
    }
    result["provenance"] = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "ref_ckpt": ref_path,
        "device": "cpu",
        "torch": torch.__version__,
        "command": " ".join(["scripts/spacer_calibrate.py", *[a for a in sys.argv[1:]]]),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    stats = result["neg_logp_nats"]
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"\nCALIBRATION: −logπ_ref mean={stats['mean']:.4f} nats "
        f"(p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}; "
        f"uniform baseline ln12={result['uniform_ref_baseline_nats']:.4f}); "
        f"recommended α tiers lo/mid/hi = {alpha_lo:.2e}/{alpha_mid:.2e}/{alpha_hi:.2e} "
        f"(α·mean ∈ [{lo},{hi}]) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
