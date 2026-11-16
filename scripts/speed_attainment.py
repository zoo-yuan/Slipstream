#!/usr/bin/env python3
"""Speed attainment (顶格率) offline metric over eval ``.replay.zlib`` bundles.

Reads the zlib replays produced by ``eval.render_scenarios=true
eval.capture_observations=true eval.keep_zlib_replays=true`` (see
``pufferlib/ocean/evaluation_utils/eval_replay.py``) and reports, per agent-step:

    attained(t)  iff  |v(t) - limit(t)| <= eps        (eps in m/s, default 1.0)

Speed/limit sources (review ruling: agent_f32 is the lossless primary source):

  * v(t)      = ``agent_f32`` column 6 = ``sim_speed`` (unsigned magnitude,
    ``env_binding.h:769``). Lossless, frame-aligned with the env chunks.
  * limit(t)  = obs ego column 8 dequantized. No lossless column exists
    (``metrics_f32``/``puffer_f32`` only store the one-sided overspeed flag),
    so obs col8 is the only viable limit source.

Obs column semantics (kernel ``write_ego_obs``, ``drive.h:5765-5784``) — the
docs/2026-11-13 §2 statements "/15mps" and "sentinel -1" are both wrong:

  * ego col0 = ``sim_speed_signed / obs_norm_speed_mps`` with
    ``obs_norm_speed_mps = 60.0`` (``drive.py:146``), NOT 15. Signed: negative
    while reversing. Used here only as a cross-check of agent_f32 col6
    (|obs col0 * 60| must match the unsigned sim_speed).
  * ego col8 = lane speed limit / 60.0, and when the agent has no current lane
    the kernel writes ``-1.0f`` BEFORE dividing, so the on-disk sentinel is
    **-1/60 ≈ -0.01667**, not -1. Lane existence test: col8 > 0.

Error budgets (why the numbers below are safe):

  * Single-step speed quantum: longitudinal acceleration cap 2.5 m/s^2
    (``constants.h:130`` ACCEL_LONG, NOT the 4 m/s^3 jerk cap at :135) over
    dt = 0.0667 s plus half a jerk step gives |dv| <= 2.5*0.0667 +
    4*0.0667^2/2 ≈ 0.176 m/s per step — far below eps = 0.5.
  * Obs col0/col8 quantization: the eval capture stores obs as float16 first
    (``eval_replay.py:88``; ~2^-12 spacing near 0.3 ⇒ ~7e-3 m/s) and viz then
    applies per-dim int16 scaling (absmax/32767 ⇒ ~4e-4 m/s). Total ≈ 0.01 m/s,
    float16-dominated — the cross-check tolerance (0.05 m/s) and eps values are
    unaffected.

Denominator (proposal §1, five rules): an agent-step is *included* iff the
agent is active (i32 col3 == 1; obs rows only exist for active agents), not
stopped/removed (i32 cols 4/5 — eval ``collision_behavior=stop`` freezes
crashed vehicles), has a current lane (i32 col6 != -1 ⇔ obs col8 > 0), and its
lane limit is <= the physically reachable cap (``--speed-cap``; limits above
``base_max_speed_mps`` = 20 are unattainable by construction). Forced-deceleration
steps (red lights, stop signs, yield) stay IN the denominator; the
``attainment_moving`` field (v >= 2 m/s subset) is the diagnostic view of that
choice. ``frozen_step_share`` / ``per_map_limit_audit`` give the exclusion audit.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/speed_attainment.py \
        --replays-dir /abs/path/to/replays_zlib [--eps 1.0] \
        [--eps-list 0.5,1.0,2.0] [--speed-norm 60] [--speed-cap 20] \
        [--out attainment.json]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pufferlib.viz


REPLAY_SUFFIX = ".replay.zlib"
REPLAY_STEM_RE = re.compile(r"^(?P<map>.+)__seed_(?P<seed>-?\d+)__episode_(?P<episode>\d+)$")

# ego obs columns (write_ego_obs): 0 = signed speed / norm, 8 = lane limit / norm.
OBS_COL_SPEED = 0
OBS_COL_LIMIT = 8
# agent_i32 columns (env_binding.h): 3 active, 4 stopped, 5 removed, 6 lane idx, 7 active slot.
I32_COL_ACTIVE = 3
I32_COL_STOPPED = 4
I32_COL_REMOVED = 5
I32_COL_LANE = 6
I32_COL_SLOT = 7
# agent_f32 column 6 = sim_speed (unsigned), the lossless speed source.
F32_COL_SPEED = 6

MOVING_SPEED_THRESHOLD_MPS = 2.0  # attainment_moving subset (proposal §1 rule 5)
SPEED_CROSSCHECK_TOLERANCE_MPS = 0.05  # vs ~0.01 m/s float16-dominated quantization

_CHUNKS = ("obs", "obs_scale", "agent_f32", "agent_i32")


def dequantize_obs_columns(obs_delta, obs_scale, columns):
    """Decode int16 wrapping frame-delta obs columns back to normalized float.

    ``obs_delta`` is the replay chunk laid out (agent, dim, frame) where each
    frame value is the int16 (mod 2^16) delta to the previous frame — the viz
    encoder keeps an agent's time series inside the deflate window. Accumulate
    in int64, then truncate to int16 for exact wraparound semantics. Returns
    {column: (agent, frame) float32 array in obs-normalized units} (multiply
    by the per-dim normalization constant, e.g. 60 for speed/limit columns).
    """
    obs_delta = np.asarray(obs_delta)
    obs_scale = np.asarray(obs_scale, dtype=np.float32)
    decoded = {}
    for column in columns:
        deltas = obs_delta[:, column, :]
        quantized = np.cumsum(deltas, axis=1, dtype=np.int64).astype(np.int16)
        decoded[column] = quantized.astype(np.float32) * obs_scale[column]
    return decoded


def parse_replay_stem(path):
    """-> (map, seed, episode_id) from the eval naming ``<map>__seed_<s>__episode_<e>``."""
    match = REPLAY_STEM_RE.match(Path(path).name[: -len(REPLAY_SUFFIX)])
    if match is None:
        return None
    return match.group("map"), int(match.group("seed")), int(match.group("episode"))


def _read_replay(path):
    """read_replay_zlib limited to the chunks this metric needs; missing chunks -> clear error."""
    try:
        return pufferlib.viz.read_replay_zlib(str(path), chunk_names=set(_CHUNKS))
    except KeyError as error:
        raise ValueError(
            f"{path}: replay has no {error.args[0]!r} chunk (was eval.capture_observations on when it was written?)"
        ) from error


def _slot_agent_map(agent_i32, active_count, path):
    """Per-frame obs-slot -> agent-index map from i32 col 7; raises unless bijective."""
    frames = agent_i32.shape[0]
    slot_map = np.full((frames, active_count), -1, dtype=np.int64)
    for frame in range(frames):
        slots = agent_i32[frame, :, I32_COL_SLOT]
        mapped = (slots >= 0) & (slots < active_count)
        slot_map[frame, slots[mapped]] = np.where(mapped)[0]
        if np.count_nonzero(mapped) != active_count:
            raise ValueError(
                f"{path}: frame {frame}: obs slot <-> agent mapping via agent_i32 col "
                f"{I32_COL_SLOT} is not a bijection onto {active_count} obs slots "
                f"({np.count_nonzero(mapped)} mapped)"
            )
    return slot_map


def load_episode(path, speed_norm, speed_tolerance=SPEED_CROSSCHECK_TOLERANCE_MPS):
    """Read one replay and return per-(frame, obs-slot) arrays with guards applied.

    Guards (column-offset sentinels, proposal §3 test 3): obs col8 < 0 ⇔ i32 col6
    == -1, and |obs col0 * speed_norm| ≈ agent_f32 col6. Any violation raises —
    a one-column drift can still produce plausible-looking means.
    """
    header, chunks = _read_replay(path)
    missing = [name for name in _CHUNKS if name not in chunks]
    if missing:
        raise ValueError(f"{path}: replay is missing chunks {missing} (was capture_observations on?)")
    if header.get("obs_layout") != "agent_dim_frame_delta":
        raise ValueError(f"{path}: unexpected obs_layout {header.get('obs_layout')!r}")
    if header.get("obs_dim", 0) <= OBS_COL_LIMIT:
        raise ValueError(f"{path}: obs_dim {header.get('obs_dim')} has no ego column {OBS_COL_LIMIT}")

    obs_delta = chunks["obs"]  # (agent=slot, dim, frame)
    agent_f32 = chunks["agent_f32"]  # (frame, agent_cap, fields)
    agent_i32 = chunks["agent_i32"]
    active_count, obs_dim, frames_obs = obs_delta.shape
    frames = agent_f32.shape[0]
    if frames_obs != frames:
        raise ValueError(f"{path}: obs has {frames_obs} frames but env chunks have {frames}")
    if obs_dim != header.get("obs_dim"):
        raise ValueError(f"{path}: obs_dim mismatch chunk {obs_dim} vs header {header.get('obs_dim')}")
    if not np.array_equal(chunks["obs_scale"].shape, (obs_dim,)):
        raise ValueError(f"{path}: obs_scale shape {chunks['obs_scale'].shape} != ({obs_dim},)")

    slot_map = _slot_agent_map(agent_i32, active_count, path)
    frame_idx = np.arange(frames)
    agent_idx = slot_map[frame_idx, :]  # (frame, slot) agent index
    gather = (frame_idx[:, None], agent_idx)  # per-(frame, slot) gather into env chunks

    speed_true = agent_f32[gather + (F32_COL_SPEED,)].astype(np.float32)  # sim_speed, unsigned
    active = agent_i32[gather + (I32_COL_ACTIVE,)] == 1
    stopped = agent_i32[gather + (I32_COL_STOPPED,)] == 1
    removed = agent_i32[gather + (I32_COL_REMOVED,)] == 1
    has_lane_i32 = agent_i32[gather + (I32_COL_LANE,)] != -1

    decoded = dequantize_obs_columns(obs_delta, chunks["obs_scale"], columns=(OBS_COL_SPEED, OBS_COL_LIMIT))
    speed_obs_signed = decoded[OBS_COL_SPEED].T * speed_norm  # (frame, slot) m/s, signed
    limit_obs = decoded[OBS_COL_LIMIT].T * speed_norm  # (frame, slot) m/s; -1 sentinel ≈ -speed_norm/60

    has_lane_obs = limit_obs > 0
    if not np.array_equal(has_lane_obs, has_lane_i32):
        bad = int(np.count_nonzero(has_lane_obs != has_lane_i32))
        raise ValueError(
            f"{path}: lane sentinel cross-check failed on {bad} agent-steps: obs col8 > 0 "
            f"must be equivalent to agent_i32 col {I32_COL_LANE} != -1 (column offset drift?)"
        )
    speed_gap = np.abs(np.abs(speed_obs_signed) - speed_true)
    if np.any(speed_gap > speed_tolerance):
        frame, slot = np.unravel_index(np.argmax(speed_gap), speed_gap.shape)
        raise ValueError(
            f"{path}: speed cross-check failed at frame {frame} slot {slot}: "
            f"|obs col{OBS_COL_SPEED}x{speed_norm:g}|={abs(speed_obs_signed[frame, slot]):.3f} "
            f"vs agent_f32 col{F32_COL_SPEED}={speed_true[frame, slot]:.3f} m/s "
            f"(max gap {speed_gap.max():.3f} > {speed_tolerance} m/s)"
        )
    return {
        "speed": speed_true,
        "limit": limit_obs,
        "active": active,
        "frozen": stopped | removed,
        "has_lane": has_lane_i32,
        "agent_index": agent_idx,
        "frames": frames,
        "active_count": active_count,
    }


def _quantiles(rates):
    if not rates:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "min": float(np.min(rates)),
        "p25": float(np.percentile(rates, 25)),
        "median": float(np.percentile(rates, 50)),
        "p75": float(np.percentile(rates, 75)),
        "max": float(np.max(rates)),
    }


def compute_speed_attainment(
    replay_paths,
    eps=1.0,
    eps_list=(0.5, 1.0, 2.0),
    speed_norm=60.0,
    speed_cap=20.0,
    moving_threshold=MOVING_SPEED_THRESHOLD_MPS,
    speed_tolerance=SPEED_CROSSCHECK_TOLERANCE_MPS,
):
    """Pool speed-attainment stats over the given .replay.zlib files. -> result dict.

    See the module docstring for the attained criterion, denominator rules and
    JSON field semantics. eps is the headline bandwidth; eps_list drives the
    pooled sensitivity table.
    """
    eps_values = sorted({float(eps), *(float(value) for value in eps_list)})
    totals = defaultdict(int)
    agent_rates = []
    per_episode = []
    audit = defaultdict(lambda: defaultdict(int))
    audit_limits = defaultdict(set)

    for episode_number, path in enumerate(replay_paths):
        path = Path(path)
        stem = parse_replay_stem(path)
        if stem is not None:
            map_name, seed, episode_id = stem
        else:
            map_name = path.stem
            seed = None
            episode_id = episode_number
        data = load_episode(path, speed_norm=speed_norm, speed_tolerance=speed_tolerance)

        speed, limit = data["speed"], data["limit"]
        active, frozen, has_lane = data["active"], data["frozen"], data["has_lane"]
        lane_steps = active & has_lane
        over_cap = lane_steps & (limit > speed_cap)
        included = lane_steps & ~over_cap & ~frozen

        totals["total_steps"] += int(np.count_nonzero(active))
        totals["steps_frozen"] += int(np.count_nonzero(active & frozen))
        totals["steps_no_lane"] += int(np.count_nonzero(active & ~frozen & ~has_lane))
        totals["steps_capped_limit"] += int(np.count_nonzero(over_cap & ~frozen))
        totals["steps_included"] += int(np.count_nonzero(included))

        deficit = (speed - limit)[included]
        totals["deficit_sum"] += float(deficit.sum())
        moving = included & (speed >= moving_threshold)
        totals["moving_steps"] += int(np.count_nonzero(moving))
        attained_by_eps = {value: included & (np.abs(speed - limit) <= value) for value in eps_values}
        for value, attained in attained_by_eps.items():
            totals[f"attained@{value:g}"] += int(np.count_nonzero(attained))
            totals[f"attained_moving@{value:g}"] += int(np.count_nonzero(attained & moving))

        # Per-map observed-limit audit (review ruling 4): every lane-having active step.
        audit_step_count = int(np.count_nonzero(lane_steps))
        audit[map_name]["steps"] += audit_step_count
        audit[map_name]["steps_over_cap"] += int(np.count_nonzero(lane_steps & (limit > speed_cap)))
        audit_limits[map_name].update(np.round(limit[lane_steps], 1).tolist())

        # Per-agent-episode pooling keyed by the env agent index (i32 col 0 identity).
        episode_included = episode_attained = 0
        for agent in np.unique(data["agent_index"]):
            rows = data["agent_index"] == agent
            agent_included = int(np.count_nonzero(included & rows))
            agent_attained = int(np.count_nonzero(attained_by_eps[float(eps)] & rows))
            episode_included += agent_included
            episode_attained += agent_attained
            if agent_included:
                agent_rates.append(agent_attained / agent_included)
        per_episode.append(
            {
                "map": map_name,
                "seed": seed,
                "episode_id": episode_id,
                "agents": int(data["active_count"]),
                "steps_included": episode_included,
                "steps_attained": episode_attained,
                "rate": round(episode_attained / episode_included, 6) if episode_included else None,
            }
        )

    steps_included = totals["steps_included"]
    result = {
        "episodes": len(per_episode),
        "total_steps": totals["total_steps"],
        "steps_included": steps_included,
        "steps_no_lane": totals["steps_no_lane"],
        "steps_capped_limit": totals["steps_capped_limit"],
        "steps_frozen": totals["steps_frozen"],
        "frozen_step_share": (
            round(totals["steps_frozen"] / totals["total_steps"], 6) if totals["total_steps"] else None
        ),
        "steps_attained": totals[f"attained@{float(eps):g}"],
        "attainment_rate": (round(totals[f"attained@{float(eps):g}"] / steps_included, 6) if steps_included else None),
        "attainment_moving": (
            round(totals[f"attained_moving@{float(eps):g}"] / totals["moving_steps"], 6)
            if totals["moving_steps"]
            else None
        ),
        "moving_steps": totals["moving_steps"],
        "mean_signed_deficit": (round(totals["deficit_sum"] / steps_included, 6) if steps_included else None),
        "per_episode": per_episode,
        "distribution": _quantiles(agent_rates),
        "per_map_limit_audit": {
            map_name: {
                "distinct_limits": sorted(float(value) for value in audit_limits[map_name]),
                "min": min(audit_limits[map_name]) if audit_limits[map_name] else None,
                "max": max(audit_limits[map_name]) if audit_limits[map_name] else None,
                "steps": entry["steps"],
                "steps_over_cap": entry["steps_over_cap"],
                "steps_over_cap_share": (
                    round(entry["steps_over_cap"] / entry["steps"], 6) if entry["steps"] else None
                ),
            }
            for map_name, entry in audit.items()
        },
        "eps_sensitivity": {
            f"{value:g}": {
                "attainment_rate": (
                    round(totals[f"attained@{value:g}"] / steps_included, 6) if steps_included else None
                ),
                "attainment_moving": (
                    round(totals[f"attained_moving@{value:g}"] / totals["moving_steps"], 6)
                    if totals["moving_steps"]
                    else None
                ),
            }
            for value in eps_values
        },
        "config": {
            "eps": float(eps),
            "eps_list": [float(value) for value in eps_values],
            "speed_norm": float(speed_norm),
            "speed_cap": float(speed_cap),
            "moving_speed_mps": float(moving_threshold),
        },
    }
    return result


def _print_report(result):
    config = result["config"]
    print(
        f"Speed attainment: {result['episodes']} episodes, "
        f"{result['steps_attained']}/{result['steps_included']} attained steps "
        f"-> rate {result['attainment_rate']} (eps={config['eps']:g} m/s, "
        f"moving-subset rate {result['attainment_moving']})"
    )
    print(
        f"Denominator audit: total {result['total_steps']} active steps = "
        f"{result['steps_included']} included + {result['steps_frozen']} frozen "
        f"(share {result['frozen_step_share']}) + {result['steps_no_lane']} no-lane "
        f"+ {result['steps_capped_limit']} capped-limit; "
        f"mean signed deficit {result['mean_signed_deficit']} m/s"
    )
    print(
        "eps sensitivity (pooled attainment_rate / attainment_moving): "
        + ", ".join(
            f"{eps}={entry['attainment_rate']}/{entry['attainment_moving']}"
            for eps, entry in result["eps_sensitivity"].items()
        )
    )
    print("Per-map observed lane speed limit audit (m/s, lane-having active steps):")
    for map_name, entry in result["per_map_limit_audit"].items():
        limits = ", ".join(f"{value:g}" for value in entry["distinct_limits"]) or "-"
        print(
            f"  {map_name}: limits [{limits}] range {entry['min']}-{entry['max']}, "
            f"{entry['steps']} steps, {entry['steps_over_cap']} over cap "
            f"({entry['steps_over_cap_share']} share; cap {config['speed_cap']:g})"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--replays-dir", required=True, help="Directory holding *.replay.zlib files (searched recursively)"
    )
    parser.add_argument("--eps", type=float, default=1.0, help="Headline attainment bandwidth in m/s (default 1.0)")
    parser.add_argument(
        "--eps-list",
        default="0.5,1.0,2.0",
        help="Comma-separated eps values for the pooled sensitivity table (default 0.5,1.0,2.0)",
    )
    parser.add_argument(
        "--speed-norm", type=float, default=60.0, help="obs_norm_speed_mps the obs was divided by (default 60)"
    )
    parser.add_argument(
        "--speed-cap",
        type=float,
        default=20.0,
        help="Physically reachable speed in m/s; lane limits above it are excluded (default 20)",
    )
    parser.add_argument("--out", default=None, help="Optional path to write the full JSON result")
    args = parser.parse_args(argv)

    replay_paths = sorted(Path(args.replays_dir).rglob(f"*{REPLAY_SUFFIX}"))
    if not replay_paths:
        print(f"No *{REPLAY_SUFFIX} files under {args.replays_dir}", file=sys.stderr)
        return 2
    try:
        eps_list = [float(value) for value in args.eps_list.split(",") if value.strip()]
    except ValueError:
        print(f"--eps-list must be comma-separated floats, got {args.eps_list!r}", file=sys.stderr)
        return 2
    result = compute_speed_attainment(
        replay_paths,
        eps=args.eps,
        eps_list=eps_list,
        speed_norm=args.speed_norm,
        speed_cap=args.speed_cap,
    )
    _print_report(result)
    if args.out:
        with open(args.out, "w") as out_file:
            json.dump(result, out_file, indent=2)
        print(f"Wrote JSON to {args.out}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
