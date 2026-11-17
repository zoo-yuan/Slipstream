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


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
