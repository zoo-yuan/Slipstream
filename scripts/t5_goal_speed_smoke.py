#!/usr/bin/env python3
"""v5 goal_speed_mode mechanism smoke (review 必改2 redefinition).

MECHANISM ONLY: this smoke never runs t3_eval_ckpt / CPU eval. It verifies on a
short CPU rollout that
  1. goal_speed_mode=lane_limit's conditioned obs column tracks the same-frame
     lane limit (ego col8), with the 15 m/s fallback on off-lane frames, and
  2. goal_speed_mode="constant" is bit-identical to the default (key omitted).

The v5 efficiency gate itself (replay64 speed-attainment 0.0 -> >=0.35 etc.) is
explicitly deferred to the GPU regression chain with lane_limit-trained arms
(E1/E2/E3): every existing checkpoint is constant-mode, the checkpoint/benchmark
mode fail-fast would abort a lane_limit eval of a constant ckpt, and local CPU
eval runs ~13.6 min/scenario.

Output: artifacts/t5-goal-speed/smoke.json
  mode / tracked_frames / mismatch_frames / constant_bit_identical / provenance
"""

import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

NUM_STEPS = 48  # "数十步" mechanism rollout, well inside the 400-step scenario
SCENARIO_LENGTH = 400

REWARD_COEF_GOAL_SPEED = 1  # constants.h:338
GOAL_SPEED_NORM = 20.0  # REWARD_BOUNDS[GOAL_SPEED] = [0, 20]
OBS_SPEED_NORM = 60.0  # drive.py default obs_norm_speed_mps
UNKNOWN_LANE_SPEED_LIMIT_MPS = 15.0
CONDITIONED_IDX = binding.EGO_FEATURES + REWARD_COEF_GOAL_SPEED
TOL = 1e-5


def make_env(**overrides):
    kwargs = dict(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        goal_source="gt",
        report_interval=1,
        num_goals=3,
        goal_radius=2.0,
        reward_conditioning=True,
        goal_speed=3.0,
        goal_speed_randomization=False,
    )
    kwargs.update(overrides)
    return Drive(**kwargs)


def expected_conditioned_col(speed_limit_col):
    limit = speed_limit_col * OBS_SPEED_NORM if speed_limit_col > 0.0 else UNKNOWN_LANE_SPEED_LIMIT_MPS
    return 2.0 * min(max(limit / GOAL_SPEED_NORM, 0.0), 1.0) - 1.0


def lane_limit_tracking():
    env = make_env(goal_speed_mode="lane_limit")
    try:
        env.reset(seed=0)
        actions = np.zeros_like(env.actions)
        tracked = on_lane = mismatch = 0
        for _ in range(NUM_STEPS):
            obs, _, _, truncations, _ = env.step(actions)
            if truncations[0]:
                break
            col8 = float(obs[0][8])
            cond = float(obs[0][CONDITIONED_IDX])
            if abs(cond - expected_conditioned_col(col8)) > TOL:
                mismatch += 1
            tracked += 1
            on_lane += col8 > 0.0
    finally:
        env.close()
    return tracked, on_lane, mismatch


def constant_bit_identical():
    default_env = make_env()
    explicit_env = make_env(goal_speed_mode="constant")
    try:
        default_env.reset(seed=0)
        explicit_env.reset(seed=0)
        identical = np.array_equal(default_env.observations, explicit_env.observations)
        actions = np.zeros_like(default_env.actions)
        for _ in range(NUM_STEPS):
            obs_a, rew_a, _, trunc_a, _ = default_env.step(actions)
            obs_b, rew_b, _, trunc_b, _ = explicit_env.step(actions)
            if trunc_a[0] or trunc_b[0]:
                break
            identical = identical and np.array_equal(obs_a, obs_b) and np.array_equal(rew_a, rew_b)
    finally:
        default_env.close()
        explicit_env.close()
    return bool(identical)


def main():
    tracked, on_lane, mismatch = lane_limit_tracking()
    result = {
        "mode": "lane_limit",
        "tracked_frames": tracked,
        "tracked_on_lane_frames": on_lane,
        "mismatch_frames": mismatch,
        "constant_bit_identical": constant_bit_identical(),
        "provenance": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
            "map": "sdc_replay_test (nuPlan log; lane limits 4.5-13.4 m/s)",
            "steps": NUM_STEPS,
            "tolerance": TOL,
            "goal_speed_norm_mps": GOAL_SPEED_NORM,
            "unknown_lane_speed_limit_mps": UNKNOWN_LANE_SPEED_LIMIT_MPS,
            "ckpt": None,
            "attainment_gate": "NOT validated here; deferred to the GPU regression chain with lane_limit-trained arms (review 必改2)",
            "test_file": "tests/unit_tests/test_goal_speed_mode.py",
        },
    }
    out_dir = os.path.join(REPO_ROOT, "artifacts", "t5-goal-speed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    ok = mismatch == 0 and tracked >= 2 and on_lane >= 1 and result["constant_bit_identical"]
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'} -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
