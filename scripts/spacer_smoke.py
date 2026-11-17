#!/usr/bin/env python3
"""SPACeR anchor CPU smoke (task step 5): a short real-env training run with
the anchor on (β=0.01, α=0.001, floor 3.0), Serial vectorization, small face.

Assertions (test_spacer_anchor's smoke twin at the integration level):
  1. losses/kl_to_ref appears in the epoch stats and is finite every epoch;
  2. losses/anchor_reward_min >= −α·floor (the clamp bound holds live);
  3. losses/anchor_reward_mean / anchor_kl_coef also present (D5 monitor keys).

CPU ONLY (training GPUs are busy). Output: artifacts/t6-spacer/smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(REPO_ROOT, "artifacts", "t6-spacer")
OUT_PATH = os.path.join(ART_DIR, "smoke.json")

DEFAULT_REF_REL = "artifacts/t3-R4x/ckpt/R_ent5_off23/models/model_puffer_drive_007424.pt"

SPACER_KL_COEF = 0.01
SPACER_REWARD_COEF = 0.001
SPACER_LOGPROB_FLOOR = 3.0


def main_repo_root() -> str | None:
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


def resolve_ref() -> str:
    candidates = [os.path.join(REPO_ROOT, DEFAULT_REF_REL)]
    main = main_repo_root()
    if main:
        candidates.append(os.path.join(main, DEFAULT_REF_REL))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"ref checkpoint not found; tried: {candidates}")


def run_smoke(num_envs: int, num_agents: int, epochs: int, seed: int, ref_path: str) -> dict:
    import pufferlib.pufferl as pufferl
    from pufferlib.config_schema import validate_puffer_drive_resources

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
    cfg["run_name"] = "spacer_smoke"
    cfg["train"].update(
        {
            "device": "cpu",  # CPU only: the training GPUs are busy
            "precision": "float32",
            "compile": False,
            "seed": seed,
            "total_timesteps": epochs * batch,
            "batch_size": batch,
            "minibatch_size": batch,
            "max_minibatch_size": batch,
            "data_dir": os.path.join(ART_DIR, "smoke_run"),
            "checkpoint_interval": 10**9,
            "normalize_rewards": False,
            # SPACeR anchor on (task spec: β=0.01 / α=0.001)
            "spacer_kl_coef": SPACER_KL_COEF,
            "spacer_reward_coef": SPACER_REWARD_COEF,
            "spacer_logprob_floor": SPACER_LOGPROB_FLOOR,
            "selfplay_ref_model_path": ref_path,
        }
    )
    cfg["vec"].update({"backend": "Serial", "num_envs": num_envs, "num_workers": num_envs, "seed": seed})
    cfg["env"]["num_agents"] = num_agents
    cfg = pufferl.normalize_puffer_drive_config(cfg, "training")
    pufferl.validate_puffer_drive_config(cfg, "training")
    validate_puffer_drive_resources(cfg, "training")

    torch.manual_seed(seed)
    vecenv = pufferl.load_env("puffer_drive", cfg, seed=cfg["vec"]["seed"])
    policy = pufferl.load_policy(cfg, vecenv, "puffer_drive")
    train_config = dict(**cfg["train"], env="puffer_drive", eval={}, run_name="spacer_smoke")
    trainer = pufferl.PuffeRL(train_config, vecenv, policy)

    epoch_rows = []
    try:
        assert trainer.anchor_ref is not None, "spacer on must build the anchor ref"
        while trainer.epoch < epochs:
            trainer.evaluate()
            trainer.last_log_time -= 10  # force the stats branch in train()
            logs = trainer.train()
            row = {
                "epoch": int(logs["epoch"]),
                "agent_steps": int(logs["agent_steps"]),
                "kl_to_ref": logs.get("losses/kl_to_ref"),
                "anchor_kl_coef": logs.get("losses/anchor_kl_coef"),
                "anchor_reward_mean": logs.get("losses/anchor_reward_mean"),
                "anchor_reward_min": logs.get("losses/anchor_reward_min"),
                "entropy": logs.get("losses/entropy"),
                "sps": logs.get("SPS"),
            }
            epoch_rows.append(row)
            print(f"[smoke] {json.dumps(row)}", flush=True)
    finally:
        trainer.utilization.stop()
        vecenv.close()
    return {"epochs": epoch_rows, "face": {
        "vec_backend": "Serial",
        "num_envs": num_envs,
        "num_agents_per_env": num_agents,
        "bptt_horizon": int(cfg["train"]["bptt_horizon"]),
        "batch_size": batch,
        "epochs": epochs,
        "seed": seed,
    }}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    ref_path = resolve_ref()
    result = run_smoke(args.num_envs, args.num_agents, args.epochs, args.seed, ref_path)

    bound = -SPACER_REWARD_COEF * SPACER_LOGPROB_FLOOR
    checks = {
        "kl_to_ref_present": all(r["kl_to_ref"] is not None for r in result["epochs"]),
        "kl_to_ref_finite": all(
            r["kl_to_ref"] is not None and math.isfinite(r["kl_to_ref"]) for r in result["epochs"]
        ),
        "anchor_reward_min_at_or_above_neg_alpha_floor": all(
            r["anchor_reward_min"] is not None and r["anchor_reward_min"] >= bound - 1e-9
            for r in result["epochs"]
        ),
        "anchor_monitor_keys_present": all(
            r["anchor_kl_coef"] is not None and r["anchor_reward_mean"] is not None for r in result["epochs"]
        ),
    }
    result["assertions"] = checks
    result["spacer"] = {
        "spacer_kl_coef": SPACER_KL_COEF,
        "spacer_reward_coef": SPACER_REWARD_COEF,
        "spacer_logprob_floor": SPACER_LOGPROB_FLOOR,
        "anchor_reward_min_bound": bound,
    }
    result["provenance"] = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "ref_ckpt": ref_path,
        "device": "cpu",
        "torch": torch.__version__,
        "test_file": "tests/unit_tests/test_spacer_anchor.py",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    ok = all(checks.values())
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'} -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
