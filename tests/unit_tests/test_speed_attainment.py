"""Speed attainment (顶格率) metric: exact synthetic replay, obs codec roundtrip, column guards.

Covers the three contract points of the speed-attainment pipeline (proposal §3 + review):
  1. Exact pooled attainment on a hand-built zlib replay (3/4 = 0.75), including one
     no-lane exclusion step, one limit>cap exclusion step, and frozen (stopped) exclusions.
  2. The (agent, dim, frame) wrapping-int16-delta obs decode roundtrips col0/col8 to
     < 1e-3 m/s when the source obs is float32 (isolates the int16 codec; the real eval
     capture stores float16 first, which dominates at ~0.01 m/s — see script docstring).
  3. Double sentinel guards raise on column-offset drift: obs col8 < 0 <=> agent_i32
     col6 == -1, and |obs col0 * speed_norm| ~= agent_f32 col6 (sim_speed, unsigned).

Synthetic replays are written with pufferlib.viz.save_interactive_replay_zlib using the
same chunk layout the eval capture produces (eval_replay.py -> drive.py replay bundle).
"""

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import speed_attainment as sa

from pufferlib import viz


FRAMES = 6
OBS_DIM = 16
SPEED_NORM = 60.0
SPEED_CAP = 20.0


def _frame_state(v, limit, lane=0, stopped=0, removed=0, active=1, obs_v=None, obs_limit=None):
    """One agent-frame. `obs_v`/`obs_limit` override the obs-side columns to build guard-breaking replays."""
    if lane == -1:
        limit = None
        obs_limit = -1.0  # sentinel written by write_ego_obs: -1.0f / obs_norm_speed_mps
    return {
        "v": v,
        "limit": limit,
        "lane": lane,
        "stopped": stopped,
        "removed": removed,
        "active": active,
        "obs_v": obs_v if obs_v is not None else v,
        "obs_limit": obs_limit if obs_limit is not None else limit,
    }


def _write_replay(directory, map_name, seed, episode_id, agent_states, obs_dtype=np.float32, obs_override=None):
    """Build a minimal legal eval-format replay and save it as .replay.zlib.

    agent_states: list (per agent slot) of per-frame _frame_state dicts. Slot j maps to
    agent index j (agent_i32 col 7 == j), matching an all-agents-active eval scenario.
    obs_override: replacement (frames, agents, obs_dim) array for the obs chunk.
    """
    agent_count = len(agent_states)
    obs = np.zeros((FRAMES, agent_count, OBS_DIM), dtype=obs_dtype)
    agent_f32 = np.zeros((FRAMES, agent_count, 12), dtype=np.float32)
    agent_i32 = np.zeros((FRAMES, agent_count, 10), dtype=np.int32)

    for agent_idx, states in enumerate(agent_states):
        for t, s in enumerate(states):
            obs[t, agent_idx, 0] = s["obs_v"] / SPEED_NORM  # ego col0: sim_speed_signed / norm
            obs[t, agent_idx, 8] = s["obs_limit"] / SPEED_NORM  # ego col8: lane limit / norm
            agent_f32[t, agent_idx, 6] = s["v"]  # f32 col6: sim_speed (unsigned)
            agent_i32[t, agent_idx, 0] = agent_idx
            agent_i32[t, agent_idx, 3] = s["active"]
            agent_i32[t, agent_idx, 4] = s["stopped"]
            agent_i32[t, agent_idx, 5] = s["removed"]
            agent_i32[t, agent_idx, 6] = s["lane"]
            agent_i32[t, agent_idx, 7] = agent_idx  # active slot index

    if obs_override is not None:  # replaces the state-derived cols entirely (codec roundtrip test)
        obs = np.asarray(obs_override)

    replay = {
        "env": {
            "num_goals": 1,
            "goal_radius": 2.0,
            "reward_conditioning": False,
            "obs_slots_partners_n": 1,
            "obs_slots_lane_n": 1,
            "obs_slots_boundary_n": 1,
            "obs_slots_traffic_controls_n": 1,
        },
        "agent_f32": agent_f32,
        "agent_i32": agent_i32,
        "metrics_f32": np.zeros((FRAMES, agent_count, 4), np.float32),
        "puffer_f32": np.zeros((FRAMES, agent_count, 4), np.float32),
        "traffic_i16": np.zeros((FRAMES, 1, 3), np.int16),
        "raw_action": np.ones((FRAMES, agent_count, 2), np.float32),
        "clipped_action": np.ones((FRAMES, agent_count, 2), np.float32),
        "value": np.zeros((FRAMES, agent_count), np.float32),
        "entropy": np.zeros((FRAMES, agent_count), np.float32),
        "obs": obs,
    }
    scenario = {
        "map_name": f"{map_name}.bin",
        "num_total_agents": agent_count,
        "active_agent_indices": list(range(agent_count)),
    }
    path = Path(directory) / f"{map_name}__seed_{seed}__episode_{episode_id:06d}.replay.zlib"
    viz.save_interactive_replay_zlib(scenario, replay, str(path))
    return path


def _exact_attainment_agents():
    """Agent 0: 3 included (2 attained, 1 conservative below moving threshold), 1 capped
    (limit 25 > cap 20), 1 no-lane, 1 frozen. Agent 1: 1 included attained, 5 frozen."""
    agent0 = [
        _frame_state(v=10.0, limit=10.0),  # included, attained (deficit ~0)
        _frame_state(v=9.3, limit=10.0),  # included, attained at eps=1, not at eps=0.5
        _frame_state(v=1.0, limit=10.0),  # included, not attained, below moving threshold
        _frame_state(v=8.0, limit=25.0),  # excluded: limit > speed cap
        _frame_state(v=8.0, limit=None, lane=-1),  # excluded: no current lane
        _frame_state(v=8.0, limit=10.0, stopped=1),  # excluded: frozen (collision_behavior=stop)
    ]
    agent1 = [_frame_state(v=10.0, limit=10.0)] + [
        _frame_state(v=7.0, limit=10.0, stopped=1) for _ in range(FRAMES - 1)
    ]
    return [agent0, agent1]


def test_synthetic_replay_exact_attainment(tmp_path):
    path = _write_replay(tmp_path, "syntheticmap", seed=7, episode_id=3, agent_states=_exact_attainment_agents())
    result = sa.compute_speed_attainment(
        [path], eps=1.0, eps_list=[0.5, 1.0, 2.0], speed_norm=SPEED_NORM, speed_cap=SPEED_CAP
    )

    assert result["episodes"] == 1
    assert result["total_steps"] == 12  # 6 frames x 2 active agents
    assert result["steps_included"] == 4
    assert result["steps_attained"] == 3
    assert result["attainment_rate"] == pytest.approx(0.75)
    assert result["steps_no_lane"] == 1
    assert result["steps_capped_limit"] == 1
    assert result["steps_frozen"] == 6
    assert result["frozen_step_share"] == pytest.approx(0.5)
    # v >= 2 m/s moving subset: the deficit-9 m/s conservative step drops out.
    assert result["attainment_moving"] == pytest.approx(1.0)
    # mean(v - limit | included): (0 + -0.7 + -9 + 0) / 4 with quantized limits (~0.003 m/s off).
    assert result["mean_signed_deficit"] == pytest.approx(-2.425, abs=0.05)

    (episode,) = result["per_episode"]
    assert episode["map"] == "syntheticmap"
    assert episode["seed"] == 7
    assert episode["episode_id"] == 3
    assert episode["agents"] == 2
    assert episode["steps_included"] == 4
    assert episode["steps_attained"] == 3
    assert episode["rate"] == pytest.approx(0.75)

    # Per-agent-episode rate distribution: agent0 = 2/3, agent1 = 1/1.
    assert result["distribution"]["min"] == pytest.approx(2 / 3)
    assert result["distribution"]["max"] == pytest.approx(1.0)

    # eps sensitivity: 0.5 -> the -0.7 m/s step falls out; 1.0 and 2.0 keep it.
    sens = {float(k): v for k, v in result["eps_sensitivity"].items()}
    assert sens[0.5]["attainment_rate"] == pytest.approx(0.5)
    assert sens[1.0]["attainment_rate"] == pytest.approx(0.75)
    assert sens[2.0]["attainment_rate"] == pytest.approx(0.75)

    # Per-map observed-limit audit: lane-having active steps saw limits {10, 25}; 25 is over cap.
    audit = result["per_map_limit_audit"]["syntheticmap"]
    assert audit["distinct_limits"] == [pytest.approx(10.0, abs=0.05), pytest.approx(25.0, abs=0.05)]
    assert audit["min"] == pytest.approx(10.0, abs=0.05)
    assert audit["max"] == pytest.approx(25.0, abs=0.05)
    assert audit["steps"] == 11  # active steps with a lane (frozen steps keep their lane)
    assert audit["steps_over_cap"] == 1


def test_obs_dequantize_roundtrip(tmp_path):
    frames = 4
    obs = np.zeros((frames, 2, OBS_DIM), dtype=np.float32)
    # Full-swing alternation forces |delta| > 32767 LSBs: the stored int16 deltas wrap.
    col0 = np.array([-0.3333, 0.3333, -0.3333, 0.3333], dtype=np.float32)
    col8 = np.array([-1.0 / 60.0, 0.41666, -1.0 / 60.0, 0.41666], dtype=np.float32)
    obs[:, 0, 0] = col0
    obs[:, 0, 8] = col8

    agents = [[_frame_state(v=1.0, limit=1.0)] * frames for _ in range(2)]
    path = _write_replay(tmp_path, "rtmap", seed=1, episode_id=0, agent_states=agents, obs_override=obs)
    _header, chunks = viz.read_replay_zlib(path, chunk_names={"obs", "obs_scale"})
    scale = chunks["obs_scale"]

    # Sanity: the encoder really stored wrapping deltas (true int diffs exceed int16).
    q_true = np.round(col0.astype(np.float64) / scale[0]).astype(np.int64)
    assert np.abs(np.diff(q_true)).max() > 32767

    decoded = sa.dequantize_obs_columns(chunks["obs"], scale, columns=(0, 8))
    assert np.max(np.abs(decoded[0][0] - col0) * SPEED_NORM) < 1e-3
    assert np.max(np.abs(decoded[8][0] - col8) * SPEED_NORM) < 1e-3
    # Untouched (zero) column decodes to zeros.
    assert np.all(decoded[0][1] == 0.0)


def test_column_offset_guard(tmp_path):
    # Sentinel A broken: i32 col6 == 0 while obs col8 carries the -1/60 no-lane sentinel.
    states = [[_frame_state(v=10.0, limit=10.0) for _ in range(FRAMES)] for _ in range(2)]
    states[0][0] = _frame_state(v=10.0, limit=10.0, obs_limit=-1.0)
    path = _write_replay(tmp_path, "guardmap", seed=1, episode_id=0, agent_states=states)
    with pytest.raises(ValueError, match="lane sentinel"):
        sa.compute_speed_attainment([path], eps=1.0, speed_norm=SPEED_NORM, speed_cap=SPEED_CAP)

    # Sentinel B broken: obs col0 drifted from agent_f32 col6 by 5 m/s.
    states = [[_frame_state(v=10.0, limit=10.0) for _ in range(FRAMES)] for _ in range(2)]
    states[0][0] = _frame_state(v=10.0, limit=10.0, obs_v=15.0)
    path = _write_replay(tmp_path, "guardmap", seed=1, episode_id=1, agent_states=states)
    with pytest.raises(ValueError, match="speed cross-check"):
        sa.compute_speed_attainment([path], eps=1.0, speed_norm=SPEED_NORM, speed_cap=SPEED_CAP)
