"""goal_speed_mode mechanism tests (v5 efficiency axis, proposal §2a + review 必改2/3/6).

Scope (review 必改2): these tests are a MECHANISM-ONLY smoke for the single gated
kernel write that pins the goal-speed coef (v_goal) to the lane's live speed limit.
The v5 efficiency gate itself (replay64 speed-attainment 0.0 -> >=0.35, per-map
floors, mean_signed_deficit >= -2.5) is NOT validated on CPU: every existing
checkpoint is constant-mode, the checkpoint/benchmark mode fail-fast would abort a
lane_limit eval of a constant ckpt, and local CPU eval runs ~13.6 min/scenario.
Attainment gate validation is explicitly deferred to the GPU regression chain with
a lane_limit-trained arm (E1/E2/E3).

Semantics under test:
  1. goal_speed_mode="constant" (default) must be bit-identical to omitting the
     key: the kernel write is gated on lane_limit, so the constant stream
     (including generate_reward_coefs RNG draws) is untouched.
  2. goal_speed_mode="lane_limit": each compute_metrics call pins
     reward_coefs[REWARD_COEF_GOAL_SPEED] to target_speed (the overspeed metric's
     lane limit; UNKNOWN_LANE_SPEED_LIMIT_MPS=15 fallback off-lane). Consumers
     (final-goal speed gate, speeding check, conditioning obs) read the coef later
     in the SAME step: compute_metrics -> compute_rewards -> compute_observations,
     so the conditioned obs column tracks the same-frame lane limit with no lag.
  3. Mode takes priority over goal_speed_randomization (the per-step write
     overwrites the spawn-time draw), so the combination
     lane_limit + reward_randomization + goal_speed_randomization fails fast at
     construction (review 必改3).

Review 必改6: assertions bind step>=1 frames only. The reset frame's conditioned
column carries the spawn-time value (goal_speed=3.0 -> 2*3/20-1 = -0.7), not the
lane limit; asserting it would couple the test to reset-frame ordering.

Column layout: ego obs col8 = lane speed limit / obs_norm_speed_mps (60), sentinel
-1/60 off-lane; conditioned goal-speed column sits at ego_features +
REWARD_COEF_GOAL_SPEED (constants.h:338 = 1) = 12, normalized 2*clamp(coef/20)-1
(REWARD_BOUNDS[GOAL_SPEED] = [0, 20]).

Fixture: the sdc_replay_test bin (nuPlan log) drives its SDC on lanes with real
limits (4.5-13.4 m/s), so on-lane frames exercise the tracking branch and off-lane
frames (if any) exercise the 15 m/s fallback -> 0.5.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

SCENARIO_LENGTH = 400
NUM_STEPS = 40  # well inside the first episode; no auto-reset frames asserted

# constants.h: REWARD_COEF_GOAL_SPEED = 1; REWARD_BOUNDS[GOAL_SPEED] = [0, 20]
REWARD_COEF_GOAL_SPEED = 1
GOAL_SPEED_NORM = 20.0
# drive.py default obs_norm_speed_mps; ego col8 = limit / 60
OBS_SPEED_NORM = 60.0
UNKNOWN_LANE_SPEED_LIMIT_MPS = 15.0

EGO_SPEED_LIMIT_IDX = 8  # write_ego_obs col8
CONDITIONED_GOAL_SPEED_IDX = binding.EGO_FEATURES + REWARD_COEF_GOAL_SPEED

TOL = 1e-5  # float32 kernel vs float64 reference


def _expected_conditioned_col(obs_speed_limit_col):
    """Reference for the conditioned column: lane frames track col8's limit
    (clamped to the [0, 20] bounds), off-lane / unknown-limit frames fall back
    to UNKNOWN_LANE_SPEED_LIMIT_MPS."""
    if obs_speed_limit_col > 0.0:
        limit = obs_speed_limit_col * OBS_SPEED_NORM
    else:
        limit = UNKNOWN_LANE_SPEED_LIMIT_MPS
    clamped = min(max(limit / GOAL_SPEED_NORM, 0.0), 1.0)
    return 2.0 * clamped - 1.0


def _make_replay_env(**overrides):
    """Single-agent replay env on the fixture map; the SDC follows the logged
    trajectory, so stepping is deterministic regardless of the zero actions fed."""
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
        goal_speed_randomization=False,  # training pin; per-step write outranks it anyway
    )
    kwargs.update(overrides)
    return Drive(**kwargs)


def test_constant_mode_bit_identical():
    """Default (key omitted) vs explicit goal_speed_mode="constant": identical
    seed, N steps of observations + rewards must match bit-for-bit. Guards the
    kernel gate so the constant default stream can never drift."""
    default_env = _make_replay_env()
    explicit_env = _make_replay_env(goal_speed_mode="constant")
    try:
        default_env.reset(seed=0)
        explicit_env.reset(seed=0)
        assert np.array_equal(default_env.observations, explicit_env.observations)
        actions = np.zeros_like(default_env.actions)
        for _ in range(NUM_STEPS):
            obs_a, rew_a, _, trunc_a, _ = default_env.step(actions)
            obs_b, rew_b, _, trunc_b, _ = explicit_env.step(actions)
            assert not trunc_a[0] and not trunc_b[0], "scenario shorter than the compare window"
            assert np.array_equal(obs_a, obs_b), "constant-mode obs drifted from the default stream"
            assert np.array_equal(rew_a, rew_b), "constant-mode rewards drifted from the default stream"
    finally:
        default_env.close()
        explicit_env.close()


def test_lane_limit_tracks_lane_limit():
    """goal_speed_mode="lane_limit" rollout: every step>=1 conditioned goal-speed
    column must equal 2*clamp(limit/20)-1 with limit read from the SAME frame's
    obs col8 (col8>0); off-lane / unknown-limit frames must sit at the 15 m/s
    fallback (== 0.5)."""
    env = _make_replay_env(goal_speed_mode="lane_limit")
    try:
        env.reset(seed=0)
        actions = np.zeros_like(env.actions)
        tracked_frames = on_lane_frames = 0
        for _ in range(NUM_STEPS):
            obs, _, _, truncations, _ = env.step(actions)
            if truncations[0]:
                break  # post-truncation obs is the next episode's reset frame (spawn value)
            ego_obs = obs[0]
            speed_limit_col = float(ego_obs[EGO_SPEED_LIMIT_IDX])
            conditioned_col = float(ego_obs[CONDITIONED_GOAL_SPEED_IDX])
            expected = _expected_conditioned_col(speed_limit_col)
            assert abs(conditioned_col - expected) <= TOL, (
                f"conditioned goal-speed column {conditioned_col} != expected {expected} "
                f"for same-frame obs col8 {speed_limit_col}"
            )
            tracked_frames += 1
            on_lane_frames += speed_limit_col > 0.0
    finally:
        env.close()

    assert tracked_frames >= 2, f"rollout too short to prove tracking: {tracked_frames} frames"
    assert on_lane_frames >= 1, (
        f"fixture never produced an on-lane frame ({on_lane_frames}); the tracking branch "
        "was not exercised and the test is vacuous"
    )


def test_constructor_validation():
    """Enum gate + 必改3 fail-fast: unknown modes are rejected, and
    lane_limit combined with reward_randomization + goal_speed_randomization
    raises before the hot path (mode overrides randomization per-step, so the
    spawn-time draw would silently mislead any reset-frame reader)."""
    with pytest.raises(ValueError, match="goal_speed_mode"):
        Drive(config_only=True, goal_speed_mode="warp")
    with pytest.raises(ValueError, match="goal_speed_randomization"):
        Drive(
            config_only=True,
            goal_speed_mode="lane_limit",
            reward_randomization=True,
            goal_speed_randomization=True,
        )
    # Both modes accept the randomization pin (goal_speed_randomization=false).
    env = Drive(
        config_only=True,
        goal_speed_mode="lane_limit",
        reward_randomization=True,
        goal_speed_randomization=False,
    )
    assert env.goal_speed_mode == "lane_limit"
    env.close()




class TestHotPathDiscipline:
    """ADR-0005 lineage discipline: the v5 kernel diff is exactly the declared lines
    (single gated write + init-time plumbing), alloc-free, RNG-stream untouched.
    Sibling knob tests (T6/T7) exempt these lines by importing EXPECTED_ADDED_LINES."""

    V5_BASELINE = "6440443"
    EXPECTED_ADDED_LINES = {
        '"goal_speed_mode=lane_limit overrides goal_speed_randomization per step; disable "',
        '"goal_speed_randomization or use goal_speed_mode=constant");',
        '#define GOAL_SPEED_MODE_CONSTANT 0',
        '#define GOAL_SPEED_MODE_LANE_LIMIT 1',
        '&& env->goal_speed_randomization) {',
        '// (c_step order: compute_metrics -> compute_rewards -> compute_observations).',
        '// (drive.h SPEED_LIMIT block), so the final-goal speed gate, the speeding check',
        '// CONSTANT (default) keeps the generate_reward_coefs spawn value bit-identical;',
        "// Kernel-side mirror of the drive.py constructor fail-fast: lane_limit's per-step",
        "// LANE_LIMIT pins it to the lane's live speed limit every compute_metrics call",
        '// UNKNOWN_LANE_SPEED_LIMIT_MPS (=15, set above). No cap at 20: REWARD_BOUNDS-normalized',
        '// and the conditioning obs all follow the lane limit. Mode takes priority over',
        '// before every consumer in the same step: the final-goal speed gate below (same function),',
        '// bit-identical; mode takes priority over goal_speed_randomization (rejected in drive.py',
        '// coef write overrides the spawn-time randomization draw, so the combination would',
        "// compute_rewards' speeding check, and write_reward_target_obs' conditioning column",
        '// consumers clamp, and the final-goal gate / speeding check want the true limit. The write',
        '// goal_speed_mode=constant (default) skips the write entirely, keeping the stream',
        '// goal_speed_randomization (the per-step write overwrites the spawn-time draw), so',
        '// is O(1), malloc-free, consumes no RNG (generate_reward_coefs stream untouched), and runs',
        '// silently mislead any reset-frame reader of the goal-speed coef.',
        '// target by construction. Off-lane / unknown-limit steps fall back to',
        '// that combination is rejected in drive.py at construction time.',
        '// to the same target_speed expression the overspeed metric uses, so v_goal == the speeding',
        '// v5 goal_speed_mode: how the per-agent goal-speed coef (v_goal) is set.',
        '// v5 goal_speed_mode=lane_limit: single gated write pinning the goal-speed coef (v_goal)',
        '// when both are live, so the spawn-time draw can never mislead a reset-frame reader).',
        'PyErr_Format(PyExc_ValueError, "goal_speed_mode must be GOAL_SPEED_MODE_CONSTANT or GOAL_SPEED_MODE_LANE_LIMIT. Got: %d",',
        'PyErr_SetString(',
        'PyExc_ValueError,',
        'agent->reward_coefs[REWARD_COEF_GOAL_SPEED] = target_speed;',
        'env->goal_speed_mode = (int) unpack(kwargs, "goal_speed_mode");',
        'env->goal_speed_mode);',
        'if (env->goal_speed_mode != GOAL_SPEED_MODE_CONSTANT && env->goal_speed_mode != GOAL_SPEED_MODE_LANE_LIMIT) {',
        'if (env->goal_speed_mode == GOAL_SPEED_MODE_LANE_LIMIT && env->reward_randomization',
        'if (env->goal_speed_mode == GOAL_SPEED_MODE_LANE_LIMIT) {',
        'int goal_speed_mode;             // GOAL_SPEED_MODE_*: constant (default) or per-step lane speed limit (v5)',
        'return -1;',
        '}',
    }

    def test_kernel_diff_is_exactly_the_declared_alloc_free_lines(self):
        import subprocess
        proc = subprocess.run(
            ["git", "diff", self.V5_BASELINE, "--",
             "pufferlib/ocean/drive/drive.h", "pufferlib/ocean/drive/binding.c",
             "pufferlib/ocean/drive/env_binding.h", "pufferlib/ocean/drive/constants.h"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            pytest.skip(f"git diff unavailable: {proc.stderr.strip()[:120]}")
        added = [
            line[1:].strip()
            for line in proc.stdout.splitlines()
            if line.startswith("+") and not line.startswith("+++") and line.strip() != "+"
        ]
        unexpected = [line for line in added if line not in self.EXPECTED_ADDED_LINES]
        assert not unexpected, f"kernel diff contains undeclared added lines: {unexpected}"
        missing = self.EXPECTED_ADDED_LINES - set(added)
        assert not missing, f"declared kernel lines absent from diff: {missing}"
        alloc_tokens = ("malloc", "calloc", "realloc", "pymem_", "new ")
        offenders = [line for line in added
                     if not line.lstrip().startswith("//")
                     and any(tok in line.lower() for tok in alloc_tokens)]
        assert not offenders, f"allocation tokens added to kernel: {offenders}"


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
