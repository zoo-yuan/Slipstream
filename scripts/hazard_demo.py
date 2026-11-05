#!/usr/bin/env python3
"""M0 baseline / M1 static-hazard visualization pipeline (wayfinder ticket 07).

Runs a pure-map gigaflow episode with every agent on the IDM controller,
captures the global agent state every step, and renders top-down artifacts:

- keyframe PNGs reusing the upstream renderer ``pufferlib.viz.plot_simulator_state``
- a cropped follow-cam GIF assembled with pillow
- ``index.html`` (frames + metrics table) and ``summary.json`` under
  ``artifacts/baseline/``

Usage:
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode baseline --seed 42
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode static-hazards --seed 42   # M1: parked/collided-cluster/cones

Notes (see .wayfinder/research/log-to-map-findings.md):
- Pure-map bins carry no log agents; gigaflow replaces them with procedurally
  spawned agents, so goal_source must be route/map (goal_source="gt" would
  dereference NULL log trajectories).
- ``collision/offroad/traffic_light/stop behavior=stop`` (training config
  defaults): agents that hit an infraction latch ``stopped`` and freeze, which
  is visible in the renders.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # repo pufferlib must shadow the venv copy

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw

# --- C-side layouts (pufferlib/ocean/drive/constants.h) ----------------------
# "obs_html_frame array field counts": agent_f32 fields in order.
AGENT_F32_FIELDS = (
    "sim_x", "sim_y", "sim_z", "heading", "length", "width", "speed",
    "steering", "accel_long", "accel_lat", "jerk_long", "jerk_lat", "goal_radius",
)
# agent_i32 fields in order.
AGENT_I32_FIELDS = (
    "id", "type", "sim_valid", "active_agent", "stopped", "removed",
    "lane_idx", "active_idx", "blindness_active", "braking_active",
)
COLLISION_IDX = 0  # Agent.metrics_array[COLLISION_IDX]
OFFROAD_IDX = 1  # Agent.metrics_array[OFFROAD_IDX]
AGENT_TYPE_VEHICLE = 1  # datatypes.h agent types
AGENT_TYPE_PEDESTRIAN = 2
AGENT_TYPE_CYCLIST = 3

DEFAULT_MAP_DIR = "../PufferDrive/pufferlib/resources/drive/binaries/carla"
DEFAULT_MAP = "Town05"


def parse_args():
    p = argparse.ArgumentParser(description="M0 baseline visualization pipeline")
    p.add_argument("--mode", choices=["baseline", "static-hazards"], default="baseline",
                   help="demo mode; mix arrives with M2")
    p.add_argument("--hazard-slots", type=int, default=48,
                   help="static co-sim partner slots reserved for M1 hazards (static-hazards only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200, help="episode steps to run")
    p.add_argument("--envs", type=int, default=2, help="parallel sub-environments")
    p.add_argument("--agents-per-env", type=int, default=150,
                   help="gigaflow agents spawned per sub-environment")
    p.add_argument("--map", default=DEFAULT_MAP,
                   help="map stem under --map-dir (e.g. Town05) or a .bin path")
    p.add_argument("--map-dir", default=DEFAULT_MAP_DIR,
                   help="directory of map .bin files (read-only reference into the sibling repo)")
    p.add_argument("--dt", type=float, default=0.3)
    p.add_argument("--goal-source", choices=["route", "map"], default="route",
                   help="pure-map bins must not use 'gt' (NULL log trajectories)")
    p.add_argument("--keyframes", default="0,40,80,120,160,199",
                   help="comma-separated step indices for full-map PNG keyframes")
    p.add_argument("--gif-stride", type=int, default=3, help="capture every Nth step for the GIF")
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--gif-window-m", type=float, default=220.0,
                   help="GIF camera window side length in meters")
    p.add_argument("--out", default=None,
                   help="output directory (default: artifacts/baseline or artifacts/static-hazards by mode)")
    return p.parse_args()


def resolve_map_bin(map_dir: str, map: str) -> Path:
    root = REPO_ROOT / map_dir if not Path(map_dir).is_absolute() else Path(map_dir)
    if map.endswith(".bin"):
        candidate = Path(map)
        if not candidate.is_absolute():
            candidate = root / map
        if candidate.is_file():
            return candidate
    candidate = root / f"opendrive__{map}.bin"
    if candidate.is_file():
        return candidate
    available = sorted(p.name for p in root.glob("*.bin")) if root.is_dir() else []
    raise FileNotFoundError(
        f"map '{map}' not found under {root}; available: {available or 'directory missing'}"
    )


def build_env(args, map_bin: Path, cosim_partner_slots: int = 0):
    """Gigaflow env with all agents on IDM; knobs mirror pufferlib/config/puffer_drive.yaml.

    cosim_partner_slots > 0 appends CONTROLLER_STATIC vehicle slots after the
    gigaflow-spawned active agents (never spawned/moved by c_step, but included
    in collision/TTC/partner-obs loops). Slot init consumes no env RNG, so the
    dynamic spawn stream for a given seed is identical to the baseline run.
    """
    from pufferlib.ocean.drive.drive import Drive

    return Drive(
        map_dir=str(map_bin),  # pin a single map bin -> deterministic episode
        num_maps=1,
        allow_map_subset=False,
        num_agents=args.envs * args.agents_per_env,
        min_agents_per_env=args.agents_per_env,
        max_agents_per_env=args.agents_per_env,  # fixed split -> exactly --envs sub-envs
        simulation_mode="gigaflow",
        goal_source=args.goal_source,
        sdc_controller="idm",
        non_sdc_controller="idm",
        non_vehicle_controller="auto",  # follows non_sdc_controller -> "replay" (moot: gigaflow spawns vehicles only)
        action_type="continuous",
        dynamics_model="jerk",
        reset_accel_on_stop=True,
        dt=args.dt,
        base_max_speed_mps=20.0,
        collision_behavior="stop",
        offroad_behavior="stop",
        traffic_light_behavior="stop",
        stop_sign_behavior="stop",
        traffic_lights_enabled=True,
        stop_signs_enabled=True,
        yield_signs_enabled=False,
        scenario_length=1200,
        resample_frequency=0,
        termination_mode=True,
        inactive_agent_threshold=0.4,
        speed_limit_random_prob=0.0,  # carla bins carry no SPDZONE1 speed zones
        min_goal_spacing=20.0,
        max_goal_spacing=200.0,
        reward_conditioning=False,
        reward_randomization=False,
        compute_eval_metrics=False,
        obs_slots_lane_n=70,
        obs_slots_boundary_n=50,
        obs_slots_partners_n=20,
        obs_dropout_lane=0.5,
        obs_dropout_boundary=0.4,
        obs_partner_relative_velocity=True,
        obs_goal_lane_distance=True,
        obs_lane_speed_limit=True,
        cosim_partner_slots=cosim_partner_slots,
        seed=args.seed,
    )


def capture_buffers(env, scenarios, extra_agent_slots: int = 0):
    """Allocate the vec_get_obs_html_frame output buffers (env, agent, field).

    extra_agent_slots widens the per-env agent dim so the html frame can carry
    the static co-sim rows (the C side writes min(num_total_agents, cap) rows
    and silently drops the rest).
    """
    from pufferlib.ocean.drive import binding

    caps = [env.agent_offsets[e + 1] - env.agent_offsets[e] for e in range(env.num_envs)]
    cap = max(caps) + extra_agent_slots
    traffic_cap = max(max(len(s.get("traffic_elements") or []) for s in scenarios), 1)
    return {
        "agent_f32": np.zeros((env.num_envs, cap, binding.AGENT_F32_FIELDS), dtype=np.float32),
        "agent_i32": np.zeros((env.num_envs, cap, binding.AGENT_I32_FIELDS), dtype=np.int32),
        "metrics_f32": np.zeros((env.num_envs, cap, binding.METRICS_F32_FIELDS), dtype=np.float32),
        "puffer_f32": np.zeros((env.num_envs, cap, binding.SCORE_F32_FIELDS), dtype=np.float32),
        "traffic_i16": np.zeros((env.num_envs, traffic_cap, binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
        "goals_f32": np.zeros((env.num_envs, cap, env.num_goals * binding.GOAL_XY_FIELDS), dtype=np.float32),
        "rewards_f32": np.zeros((env.num_envs, cap, binding.REWARD_F32_FIELDS), dtype=np.float32),
        "coefs_f32": np.zeros((env.num_envs, cap, binding.NUM_REWARD_COEFS), dtype=np.float32),
    }


def env_active_mask(agent_i32_env):
    """Rows of one env that are live agents (spawned, not removed)."""
    return (agent_i32_env[:, 3] == 1) & (agent_i32_env[:, 2] == 1)


# --- GIF renderer (minimal pillow top-down camera) ---------------------------
GIF_COLORS = {
    "background": "#f5f5f5",
    "lane": "#d9d9d9",  # drivable lane polylines (road type 1/2)
    "line": "#bdbdbd",  # lane markings (road type 10-19)
    "edge": "#555555",  # road edges (road type 20-29)
    "crosswalk": "#a8d5a8",  # crosswalks (road type 31)
    "vehicle": "#1f77b4",
    "sdc": "#d62728",
    "stopped": "#8c8c8c",
    "pedestrian": "#2e8b57",
    "cyclist": "#ff8c00",
    "ink": "#222222",
}
TRAFFIC_STATE_COLORS = {0: "#8c8c8c", 1: "#ff0000", 2: "#ffee00", 3: "#00cc00", 4: "#8c8c8c"}
# traffic control states: constants.h TRAFFIC_CONTROL_STATE_{UNKNOWN,RED,YELLOW,GREEN,OFF}


def _world_to_px(x, y, center, scale, size_px):
    px = (x - center[0]) * scale + size_px / 2
    py = size_px / 2 - (y - center[1]) * scale  # image y axis points down
    return px, py


def _polyline(element, key):
    xs, ys = element.get("x"), element.get("y")
    return [(float(a), float(b)) for a, b in zip(xs or [], ys or [])]


def render_gif_frame(step, agent_f32, agent_i32, road_elements, traffic_elements, sdc_row,
                     center, window_m, size_px, dt, seed, map_name,
                     mode_label="baseline", static_indices=None):
    """Baseline mode must keep the default-argument rendering byte-identical."""
    scale = size_px / window_m
    img = Image.new("RGB", (size_px, size_px), GIF_COLORS["background"])
    draw = ImageDraw.Draw(img)

    for el in road_elements:
        rtype = int(el.get("type", 0))
        pts = _polyline(el, "x")
        if len(pts) < 2:
            continue
        px_pts = [_world_to_px(x, y, center, scale, size_px) for x, y in pts]
        if 1 <= rtype <= 9:
            color, width = GIF_COLORS["lane"], 3
        elif 10 <= rtype <= 19:
            color, width = GIF_COLORS["line"], 1
        elif 20 <= rtype <= 29:
            color, width = GIF_COLORS["edge"], 2
        elif rtype == 31:
            color, width = GIF_COLORS["crosswalk"], 2
        else:
            continue
        draw.line(px_pts, fill=color, width=width)

    # traffic light stop lines, colored by their pregenerated state at this timestep
    for el in traffic_elements:
        sl = el.get("stop_line")
        if sl is None or len(sl) < 4:
            continue
        states = el.get("states") or []
        state = int(states[step]) if step < len(states) else 0
        p1 = _world_to_px(sl[0], sl[1], center, scale, size_px)
        p2 = _world_to_px(sl[3], sl[4], center, scale, size_px)
        draw.line([p1, p2], fill=TRAFFIC_STATE_COLORS.get(state, "#8c8c8c"), width=3)

    # agents as oriented rectangles; color by type/controller role
    active = env_active_mask(agent_i32)
    if static_indices:
        sel = np.zeros(agent_i32.shape[0], dtype=bool)
        sel[np.asarray(static_indices, dtype=int)] = True
        active = active | (sel & (agent_i32[:, 2] == 1) & (agent_i32[:, 5] == 0))
    static_set = set(int(i) for i in (static_indices or ()))
    for i in np.flatnonzero(active):
        x, y = float(agent_f32[i, 0]), float(agent_f32[i, 1])
        heading = float(agent_f32[i, 3])
        length = max(float(agent_f32[i, 4]), 0.5)
        width = max(float(agent_f32[i, 5]), 0.5)
        atype = int(agent_i32[i, 1])
        stopped_latched = int(agent_i32[i, 4]) == 1
        is_sdc = i == sdc_row
        cx, cy = _world_to_px(x, y, center, scale, size_px)
        if atype == AGENT_TYPE_PEDESTRIAN:
            r = max(2, int(0.5 * scale))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GIF_COLORS["pedestrian"])
            continue
        if atype == AGENT_TYPE_CYCLIST:
            r = max(2, int(0.8 * scale))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GIF_COLORS["cyclist"])
            continue
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        corners = [
            (length / 2, width / 2), (length / 2, -width / 2),
            (-length / 2, -width / 2), (-length / 2, width / 2),
        ]
        poly = []
        for lx, ly in corners:
            wx = x + lx * cos_h - ly * sin_h
            wy = y + lx * sin_h + ly * cos_h
            poly.append(_world_to_px(wx, wy, center, scale, size_px))
        if i in static_set:
            fill = "#ff7f0e"  # static hazard: parked / collided cluster / cone
        else:
            fill = GIF_COLORS["stopped"] if stopped_latched else (GIF_COLORS["sdc"] if is_sdc else GIF_COLORS["vehicle"])
        draw.polygon(poly, fill=fill, outline="#222222" if is_sdc or i in static_set else "#333333")
        # heading tick
        tipx, tipy = _world_to_px(x + 0.6 * length * cos_h, y + 0.6 * length * sin_h, center, scale, size_px)
        draw.line([poly[0], (tipx, tipy)], fill="#222222", width=1)

    # HUD + scale bar
    speeds = agent_f32[active, 6]
    mean_v = float(np.mean(speeds)) if speeds.size else 0.0
    n_stopped = int(np.sum(agent_i32[active, 4] == 1))
    n_removed = int(np.sum(agent_i32[active, 5] == 1))
    draw.text((10, 8), f"t={step} ({step * dt:.1f}s)  agents={int(active.sum())}  "
                       f"mean_v={mean_v:.1f}m/s  latched={n_stopped}  removed={n_removed}",
              fill=GIF_COLORS["ink"])
    draw.text((10, 24), f"IDM gigaflow {mode_label} · {map_name} · seed {seed}", fill=GIF_COLORS["ink"])
    if static_indices:
        n_static = len(static_indices)
        draw.text((10, 40), f"orange = {n_static} static hazards (parked / crashed / cones)", fill="#ff7f0e")
    bar_m, bar_px = 50.0, 50.0 * scale
    x0, y0 = size_px - int(bar_px) - 16, size_px - 24
    draw.line([(x0, y0), (x0 + bar_px, y0)], fill=GIF_COLORS["ink"], width=2)
    draw.line([(x0, y0 - 4), (x0, y0 + 4)], fill=GIF_COLORS["ink"], width=2)
    draw.line([(x0 + bar_px, y0 - 4), (x0 + bar_px, y0 + 4)], fill=GIF_COLORS["ink"], width=2)
    draw.text((x0, y0 + 6), "50 m", fill=GIF_COLORS["ink"])
    return img


# --- artifact writers ---------------------------------------------------------

METRICS_ROWS = (
    ("active agents (final)", "active_final"),
    ("collision agents (latched)", "collision_agents"),
    ("offroad agents (latched)", "offroad_agents"),
    ("removed agents (final)", "removed_final"),
    ("speed min / avg / max (m/s)", "speed_summary"),
    ("mean speed final step (m/s)", "speed_final"),
    ("episode ended at step", "ended_step"),
)


def write_index_html(out_dir, args, map_bin, summary, keyframe_paths, gif_path):
    cmd = "PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode baseline"
    episode_note = f'steps={summary["steps_run"]}/{summary["steps_requested"]} · dt={args.dt}s'
    rows = ""
    for e, m in enumerate(summary["per_env"]):
        cells = "".join(
            f"<td>{m.get(key, '-')}</td>" for _, key in METRICS_ROWS
        )
        rows += f"<tr><td>env {e}</td>{cells}</tr>"
    header = "".join(f"<th>{label}</th>" for label, _ in METRICS_ROWS)
    keyframe_html = ""
    for e, paths in keyframe_paths.items():
        imgs = "\n".join(
            f'      <figure><img src="{p}" alt="env {e} step {Path(p).stem}"/>'
            f"<figcaption>env {e} · {Path(p).stem.replace('_', ' ')}</figcaption></figure>"
            for p in paths
        )
        keyframe_html += f"    <h2>env {e} keyframes (full map, upstream viz renderer)</h2>\n{imgs}\n"
    notes = "".join(f"      <li>{n}</li>" for n in summary["notes"])
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Slipstream hazard-demo · M0 baseline</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1, h2 {{ color: #333; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 14px; text-align: left; }}
  th {{ background: #eee; }}
  figure {{ margin: 8px; display: inline-block; }}
  img {{ border: 1px solid #ddd; max-width: 640px; }}
  figcaption {{ font-size: 12px; color: #666; text-align: center; }}
  code {{ background: #eee; padding: 2px 6px; }}
  li {{ margin: 4px 0; }}
</style>
</head>
<body>
  <h1>M0 baseline visualization · gigaflow + IDM</h1>
  <p>命令: <code>{cmd} --seed {args.seed}</code></p>
  <p>地图: <code>{map_bin}</code> · seed={args.seed} · {episode_note}</p>
  <h2>跟车 GIF (env 0, 窗口 {args.gif_window_m:.0f} m)</h2>
  <img src="{gif_path}" alt="baseline gif" width="640"/>
  {keyframe_html}
  <h2>指标 (来自 vec_get_obs_html_frame 每步抓取)</h2>
  <table>
    <tr><th>env</th>{header}</tr>
    {rows}
  </table>
  <h2>看什么 / 已知口径</h2>
  <ul>
{notes}
  </ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


# --- M1 static hazards (wayfinder ticket 07) ---------------------------------
#
# Static long-tail elements placed on top of a gigaflow reset via the co-sim
# static-slot path (zero kernel changes):
#   - Drive(cosim_partner_slots=N) appends N CONTROLLER_STATIC VEHICLE agents
#     after the gigaflow active agents; c_step never moves them, but collision /
#     TTC / partner-obs loops include them, so dynamic agents collide against
#     them and latch (collision_behavior=stop).
#   - env.set_agent_states(idx, x, y, z, heading, vx=vy=0, yaw_rate=0,
#     accel_long=0, steering=0) + env.set_agent_sizes(idx, length, width)
#     teleport them into lane-anchored poses; the C side refreshes lane
#     association per pose (agents_state["current_lane_idx"] != -1 is the
#     kernel's own on-lane evidence). env.recompute_observations() then makes
#     the dynamic agents see them.
#   - Binding caveat: vec_set_agent_states/vec_set_agent_sizes operate on
#     vec->envs[0] only (co-sim single-env assumption, env_binding.h), so this
#     mode pins --envs 1.
#   - world_mean_x/y are declared but never assigned in the kernel (always 0),
#     so road-element coordinates == set_agent_states world coordinates; the
#     get_global_agent_state readback below asserts this at runtime.

VEHICLE_PRESETS = {  # (length_m, width_m) — ticket 07 spec: car/bus/truck
    "car": (4.5, 1.9),
    "bus": (12.0, 2.5),
    "truck": (8.0, 2.4),
}
PARKED_SIZE_PROBS = (("car", 0.6), ("bus", 0.2), ("truck", 0.2))
ASSUMED_LANE_WIDTH_M = 3.5  # carla default; map bins expose no lane width
HAZARD_GIF_WINDOW_M = 150.0
HAZARD_KEYFRAMES = (0, 20, 40, 60, 80)
COMPARE_KEYFRAMES = (0, 40, 80)
ZOOM_WINDOW_M = 90.0
MAX_PLACE_ATTEMPTS = 40
REJECT_REASONS = (
    "dynamic_overlap", "hazard_overlap", "edge_crossing",
    "parked_spacing", "cluster_overlap_failed", "max_attempts", "no_lane",
)


def hazard_rng(seed: int) -> np.random.RandomState:
    """Placement RNG, deliberately independent of the episode RNG stream."""
    return np.random.RandomState((seed * 7919 + 13) % (2 ** 31))


def wrapped_angle_diff(a, b):
    return (np.asarray(a, float) - np.asarray(b, float) + np.pi) % (2 * np.pi) - np.pi


def obb_corners(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    return np.array([
        [x + c * hl - s * hw, y + s * hl + c * hw],
        [x + c * hl + s * hw, y + s * hl - c * hw],
        [x - c * hl + s * hw, y - s * hl - c * hw],
        [x - c * hl - s * hw, y - s * hl + c * hw],
    ])


def obb_overlap(a, b, margin=0.0):
    """2D SAT between oriented boxes a=(x,y,heading,length,width) and b."""
    A, B = obb_corners(*a), obb_corners(*b)
    axes = []
    for corners in (A, B):
        for i in (0, 1):
            e = corners[i + 1] - corners[i]
            ax = np.array([-e[1], e[0]])
            n = np.linalg.norm(ax)
            if n > 1e-9:
                axes.append(ax / n)
    for axis in axes:
        pa, pb = A @ axis, B @ axis
        if pa.max() + margin < pb.min() or pb.max() + margin < pa.min():
            return False
    return True


def _point_segment_distSq(p, a, b):
    ab = b - a
    denom = ab @ ab
    t = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float((p - proj) @ (p - proj))


class LaneIndex:
    """Drivable-lane geometry (road type 1-4 polylines) for hazard anchoring."""

    def __init__(self, road_elements, min_len_m=12.0):
        self.lanes = []
        for el in road_elements:
            rtype = int(el.get("type", 0))
            if not (1 <= rtype <= 4):
                continue
            xs, ys, zs = el.get("x"), el.get("y"), el.get("z")
            if not xs or len(xs) < 3:
                continue
            pts = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
            z = np.asarray(zs if zs is not None else np.zeros(len(xs)), float)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            total = float(seg.sum())
            if total < min_len_m:
                continue
            headings = np.arctan2(np.diff(pts, axis=0)[:, 1], np.diff(pts, axis=0)[:, 0])
            dh = wrapped_angle_diff(headings[1:], headings[:-1])
            self.lanes.append({
                "id": int(el.get("id", len(self.lanes))),
                "pts": pts, "z": z, "seg": seg,
                "cum": np.concatenate([[0.0], np.cumsum(seg)]),
                "headings": headings,
                "total": total,
                "straight": bool(np.all(np.abs(dh) < 0.35)),
            })

    def pick(self, rng, min_len_m, straight_only=False):
        cand = [ln for ln in self.lanes
                if ln["total"] >= min_len_m and (ln["straight"] or not straight_only)]
        if not cand:
            return None
        w = np.array([ln["total"] - min_len_m + 1.0 for ln in cand])
        return cand[int(rng.choice(len(cand), p=w / w.sum()))]

    def pose_at(self, lane, s, lateral):
        """World pose (x, y, z, heading) at arclength s, offset `lateral` left."""
        s = float(np.clip(s, 0.0, lane["total"]))
        i = int(np.searchsorted(lane["cum"], s, side="right") - 1)
        i = min(max(i, 0), len(lane["seg"]) - 1)
        t = (s - lane["cum"][i]) / max(lane["seg"][i], 1e-9)
        p = lane["pts"][i] * (1 - t) + lane["pts"][i + 1] * t
        h = lane["headings"][i]
        z = lane["z"][i] * (1 - t) + lane["z"][i + 1] * t
        x = p[0] + lateral * -np.sin(h)
        y = p[1] + lateral * np.cos(h)
        return float(x), float(y), float(z), float(h)


class EdgeIndex:
    """Road-edge polylines (type 20-29) for Python-side off-road rejection.

    Mirrors the kernel rule (swept box crossing an edge segment) conservatively:
    an edge segment blocks a pose if any of its sample points falls inside the
    box shrunk by `shrink_m`, or any box corner sits within `clearance_m` of it.
    """

    SAMPLE_STEP_M = 0.4
    PRUNE_RADIUS_M = 8.0

    def __init__(self, road_elements):
        segs = []
        for el in road_elements:
            rtype = int(el.get("type", 0))
            if not (20 <= rtype <= 29):
                continue
            xs, ys = el.get("x"), el.get("y")
            if not xs or len(xs) < 2:
                continue
            pts = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
            segs.extend(zip(pts[:-1], pts[1:]))
        self.segs = np.array(segs) if segs else np.zeros((0, 2, 2))

    def blocked(self, pose, shrink_m=0.1, clearance_m=0.15):
        x, y, heading, length, width = pose
        corners = obb_corners(*pose)
        near = [seg for seg in self.segs
                if (seg[:, 0].mean() - x) ** 2 + (seg[:, 1].mean() - y) ** 2
                < (self.PRUNE_RADIUS_M + length) ** 2]
        if not near:
            return False
        c, s = np.cos(heading), np.sin(heading)
        hl = max(length / 2.0 - shrink_m, 0.05)
        hw = max(width / 2.0 - shrink_m, 0.05)
        for seg in near:
            d = float(np.linalg.norm(seg[1] - seg[0]))
            n_samples = max(int(d / self.SAMPLE_STEP_M) + 1, 2)
            for t in np.linspace(0.0, 1.0, n_samples):
                p = seg[0] * (1 - t) + seg[1] * t
                lx, ly = p[0] - x, p[1] - y
                if abs(lx * c + ly * s) <= hl and abs(-lx * s + ly * c) <= hw:
                    return True
            for corner in corners:
                if _point_segment_distSq(corner, seg[0], seg[1]) <= clearance_m ** 2:
                    return True
        return False


def dynamic_boxes(agent_f32, agent_i32):
    """(x, y, heading, length, width) of live dynamic agents from an html frame."""
    mask = (agent_i32[:, 3] == 1) & (agent_i32[:, 2] == 1) & (agent_i32[:, 5] == 0)
    return [tuple(float(v) for v in agent_f32[i, [0, 1, 3, 4, 5]]) for i in np.flatnonzero(mask)]


def pose_conflict_reason(pose, dyn, placed, edges, lane_id=None, s=None,
                         self_cluster=None, skip_parked_spacing=False):
    x, y, heading, length, width = pose
    if edges.blocked((x, y, heading, length, width)):
        return "edge_crossing"
    for d in dyn:
        if obb_overlap(pose, d, margin=0.05):
            return "dynamic_overlap"
    for other in placed:
        if self_cluster is not None and other.get("cluster_id") == self_cluster:
            continue  # intra-cluster overlap is intentional
        if obb_overlap(pose, (other["x"], other["y"], other["heading"],
                              other["length"], other["width"]), margin=0.05):
            return "hazard_overlap"
    if not skip_parked_spacing and lane_id is not None:
        for other in placed:
            if other["kind"] == "parked" and other["lane_id"] == lane_id:
                gap_needed = (length + other["length"]) / 2.0 + 1.0
                if abs(s - other["s"]) < gap_needed:
                    return "parked_spacing"
    return None


def plan_hazard_counts(rng):
    return {
        "parked": int(rng.randint(6, 11)),        # 6-10 per env
        "clusters": int(rng.randint(1, 3)),       # 1-2 crash clusters
        "cone_zones": int(rng.randint(1, 3)),     # 1-2 construction segments
    }


def _pick_preset(rng):
    names, probs = zip(*PARKED_SIZE_PROBS)
    return names[int(rng.choice(len(names), p=probs))]


def place_parked(rng, lanes, edges, dyn, placed, rejects, count):
    placed_n = 0
    for _ in range(count):
        vtype = _pick_preset(rng)
        length, width = VEHICLE_PRESETS[vtype]
        entry = None
        for _attempt in range(MAX_PLACE_ATTEMPTS):
            lane = lanes.pick(rng, min_len_m=length + 10.0)
            if lane is None:
                rejects["no_lane"] += 1
                break
            s = rng.uniform(length / 2 + 2.0, lane["total"] - length / 2 - 2.0)
            side = 1.0 if rng.uniform() < 0.5 else -1.0
            lateral = side * rng.uniform(0.9, 1.35)  # kerbside half-on-road feel
            x, y, z, h = lanes.pose_at(lane, s, lateral)
            pose = (x, y, h, length, width)
            reason = pose_conflict_reason(pose, dyn, placed, edges,
                                          lane_id=lane["id"], s=s)
            if reason:
                rejects[reason] += 1
                continue
            entry = {"kind": "parked", "vtype": vtype, "x": x, "y": y, "z": z,
                     "heading": h, "length": length, "width": width,
                     "lane_id": lane["id"], "s": s, "lateral": lateral}
            placed.append(entry)
            placed_n += 1
            break
        if entry is None:
            rejects["max_attempts"] += 1
    return placed_n


def _cluster_poses(rng, lanes, lane, lenA, wA, lenB, wB):
    """Two deliberately overlapping poses: rear-end or T-bone (per seed)."""
    pattern = "rear_end" if rng.uniform() < 0.5 else "t_bone"
    sA = rng.uniform(8.0, lane["total"] - 8.0)
    xA, yA, zA, hA = lanes.pose_at(lane, sA, rng.uniform(-0.2, 0.2))
    poseA = (xA, yA, hA, lenA, wA)
    if pattern == "rear_end":
        overlap_m = rng.uniform(0.5, 1.0)
        xB, yB, zB, hB = lanes.pose_at(lane, sA - (lenA / 2 + lenB / 2 - overlap_m),
                                       rng.uniform(-0.15, 0.15))
        hB = float(hB + rng.uniform(-0.05, 0.05))
    else:
        hB = float(hA + (1.0 if rng.uniform() < 0.5 else -1.0) * rng.uniform(1.35, 1.75))
        uA = np.array([np.cos(hA), np.sin(hA)])
        uB = np.array([np.cos(hB), np.sin(hB)])
        ctr = np.array([xA, yA]) + uA * (0.35 * lenA) - uB * (0.30 * lenB)
        xB, yB, zB = float(ctr[0]), float(ctr[1]), zA
        # guarantee the intentional overlap even under heading jitter
        for _ in range(6):
            if obb_overlap(poseA, (xB, yB, hB, lenB, wB), margin=0.0):
                break
            ctr -= (ctr - np.array([xA, yA])) / np.linalg.norm(ctr - np.array([xA, yA])) * 0.3
            xB, yB = float(ctr[0]), float(ctr[1])
    return pattern, poseA, (xB, yB, hB, lenB, wB), zA


def place_clusters(rng, lanes, edges, dyn, placed, rejects, count):
    placed_n = 0
    for c in range(count):
        cluster_id = f"cluster{c}"
        entry_pair = None
        for _attempt in range(MAX_PLACE_ATTEMPTS):
            lenA, wA = VEHICLE_PRESETS[_pick_preset(rng)]
            lenB, wB = VEHICLE_PRESETS[_pick_preset(rng)]
            lane = lanes.pick(rng, min_len_m=max(25.0, lenA + lenB + 12.0), straight_only=True)
            if lane is None:
                rejects["no_lane"] += 1
                break
            pattern, poseA, poseB, z = _cluster_poses(rng, lanes, lane, lenA, wA, lenB, wB)
            if not obb_overlap(poseA, poseB, margin=0.0):
                rejects["cluster_overlap_failed"] += 1
                continue
            reason = None
            for pose in (poseA, poseB):
                reason = pose_conflict_reason(pose, dyn, placed, edges,
                                              self_cluster=cluster_id,
                                              skip_parked_spacing=True)
                if reason:
                    break
            if reason:
                rejects[reason] += 1
                continue
            entry_pair = [
                {"kind": "collided", "vtype": "crash", "cluster_id": cluster_id,
                 "pattern": pattern, "x": p[0], "y": p[1], "z": z,
                 "heading": p[2], "length": p[3], "width": p[4],
                 "lane_id": lane["id"], "s": None, "lateral": None}
                for p in (poseA, poseB)
            ]
            break
        if entry_pair is None:
            rejects["max_attempts"] += 1
            continue
        placed.extend(entry_pair)
        placed_n += 1
    return placed_n


def _cone_poses(rng, lanes, lane):
    """Taper (diagonal narrowing) or block (full-lane closure), ~1.5 m spacing."""
    pattern = "taper" if rng.uniform() < 0.5 else "block"
    cone_l = cone_w = 0.5
    poses = []
    if pattern == "taper":
        n = 8
        span = (n - 1) * 1.5
        base_s = rng.uniform(4.0, max(4.1, lane["total"] - span - 4.0))
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        lat0 = side * rng.uniform(0.2, 0.5)
        lat1 = side * rng.uniform(1.1, 1.4)
        for k in range(n):
            s = base_s + k * 1.5
            lat = float(np.interp(k, [0, n - 1], [lat0, lat1])) + rng.uniform(-0.05, 0.05)
            x, y, z, h = lanes.pose_at(lane, s, lat)
            poses.append(((x, y, h, cone_l, cone_w), z, s, lat))
    else:
        # kernel lane-association radius is 3×width (0.5m cone → 1.5m); keep the
        # outermost cone center within 1.45m of the lane centerline so every cone
        # stays kernel-associated (on-lane evidence for the legality report)
        half_lane = min(ASSUMED_LANE_WIDTH_M / 2.0 - 0.35, 1.45)
        base_s = rng.uniform(4.0, max(4.1, lane["total"] - 9.0))
        stagger = (2 * half_lane) / 6 / 2
        for row in range(2):
            lats = np.linspace(-half_lane, half_lane, 7) + (stagger if row else 0.0)
            lats = np.clip(lats, -1.45, 1.45)
            for lat in lats:
                s = base_s + row * 1.6 + rng.uniform(-0.1, 0.1)
                x, y, z, h = lanes.pose_at(lane, s, float(lat) + rng.uniform(-0.05, 0.05))
                poses.append(((x, y, h, cone_l, cone_w), z, s, float(lat)))
    return pattern, poses


def place_cone_zones(rng, lanes, edges, dyn, placed, rejects, count):
    placed_zones, cones_placed = 0, 0
    for zi in range(count):
        zone_id = f"cones{zi}"
        zone_entries = None
        for _attempt in range(MAX_PLACE_ATTEMPTS):
            lane = lanes.pick(rng, min_len_m=20.0, straight_only=True)
            if lane is None:
                rejects["no_lane"] += 1
                break
            pattern, poses = _cone_poses(rng, lanes, lane)
            reason = None
            for pose, _z, _s, _lat in poses:
                reason = pose_conflict_reason(pose, dyn, placed, edges,
                                              skip_parked_spacing=True)
                if reason:
                    break
            if reason:
                rejects[reason] += 1
                continue
            zone_entries = [
                {"kind": "cone", "vtype": "cone", "zone_id": zone_id,
                 "pattern": pattern, "x": pose[0], "y": pose[1], "z": z,
                 "heading": pose[2], "length": pose[3], "width": pose[4],
                 "lane_id": lane["id"], "s": s, "lateral": lat}
                for pose, z, s, lat in poses
            ]
            break
        if zone_entries is None:
            rejects["max_attempts"] += 1
            continue
        placed.extend(zone_entries)
        placed_zones += 1
        cones_placed += len(zone_entries)
    return placed_zones, cones_placed


def apply_static_hazards(env, args, placed):
    """Batch-write accepted poses into the static slots, then refresh obs."""
    idxs = np.array([args.agents_per_env + k for k in range(len(placed))], np.int32)
    for k, e in enumerate(placed):
        e["idx"] = int(idxs[k])
    zeros = np.zeros(len(placed), np.float32)
    env.set_agent_states(
        idxs,
        np.array([e["x"] for e in placed], np.float32),
        np.array([e["y"] for e in placed], np.float32),
        np.array([e["z"] for e in placed], np.float32),
        np.array([e["heading"] for e in placed], np.float32),
        zeros, zeros, zeros, zeros,          # vx=vy=yaw_rate=accel_long=0
        None,                                # seconds_stopped: keep c_step accumulation
        zeros,                               # steering=0
    )
    env.set_agent_sizes(
        idxs,
        np.array([e["length"] for e in placed], np.float32),
        np.array([e["width"] for e in placed], np.float32),
    )
    env.recompute_observations()
    return idxs


def verify_static_hazards(env, placed, idxs, agent_f32, agent_i32):
    """Kernel + geometry evidence: lane association, world readback, OBB postcheck."""
    state = env.get_state()[0]
    agents_state = state.get("agents") or []
    lane_assoc_fail = 0
    lane_assoc_fail_detail = []
    for e in placed:
        a = agents_state[e["idx"]] if e["idx"] < len(agents_state) else {}
        if not a.get("sim_valid") or a.get("current_lane_idx", -1) == -1:
            lane_assoc_fail += 1
            lane_assoc_fail_detail.append(
                {"kind": e["kind"], "idx": e["idx"], "lateral": e.get("lateral"),
                 "width": e["width"]})
    g = env.get_global_agent_state(include_static=True)
    readback_err = 0.0
    for k, e in enumerate(placed):
        row = int(idxs[k])
        readback_err = max(readback_err,
                           abs(float(g["x"][row]) - e["x"]),
                           abs(float(g["y"][row]) - e["y"]))
    dyn = dynamic_boxes(agent_f32, agent_i32)
    overlap_violations = 0
    for e in placed:
        pose = (e["x"], e["y"], e["heading"], e["length"], e["width"])
        if any(obb_overlap(pose, d, margin=0.05) for d in dyn):
            overlap_violations += 1
    cluster_overlaps = 0
    by_cluster = {}
    for e in placed:
        if e.get("cluster_id"):
            by_cluster.setdefault(e["cluster_id"], []).append(e)
    for pair in by_cluster.values():
        if len(pair) == 2 and obb_overlap(
            (pair[0]["x"], pair[0]["y"], pair[0]["heading"], pair[0]["length"], pair[0]["width"]),
            (pair[1]["x"], pair[1]["y"], pair[1]["heading"], pair[1]["length"], pair[1]["width"]),
            margin=0.0,
        ):
            cluster_overlaps += 1
    return {
        "kernel_lane_assoc_failures": int(lane_assoc_fail),
        "kernel_lane_assoc_failures_detail": lane_assoc_fail_detail,
        "dynamic_overlap_violations_postcheck": int(overlap_violations),
        "intentional_intra_cluster_overlaps": int(cluster_overlaps),
        "world_readback_max_err_m": round(float(readback_err), 6),
        "static_agent_indices_head": list(state.get("static_agent_indices") or [])[:6],
    }


def crop_world(png_path, map_corners, center, window_m):
    """Crop a square world-space window out of a viz.plot_simulator_state PNG."""
    img = np.asarray(Image.open(png_path))
    h, w = img.shape[:2]
    cx = (map_corners[0] + map_corners[2]) / 2
    cy = (map_corners[1] + map_corners[3]) / 2
    radius = max(map_corners[2] - map_corners[0], map_corners[3] - map_corners[1]) / 2 * 1.02
    x_min, x_max, y_min, y_max = cx - radius, cx + radius, cy - radius, cy + radius
    half = window_m / 2
    px0 = int(np.clip((center[0] - half - x_min) / (x_max - x_min) * w, 0, w - 2))
    px1 = int(np.clip((center[0] + half - x_min) / (x_max - x_min) * w, px0 + 2, w))
    py0 = int(np.clip((y_max - (center[1] + half)) / (y_max - y_min) * h, 0, h - 2))
    py1 = int(np.clip((y_max - (center[1] - half)) / (y_max - y_min) * h, py0 + 2, h))
    return Image.fromarray(img[py0:py1, px0:px1])


def side_by_side(left_png, right_png, out_path, labels=("baseline", "static-hazards")):
    a, b = Image.open(left_png), Image.open(right_png)
    h = max(a.height, b.height)
    divider = 8
    canvas = Image.new("RGB", (a.width + b.width + divider, h), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + divider, 0))
    draw = ImageDraw.Draw(canvas)
    for x0, label in ((8, labels[0]), (a.width + divider + 8, labels[1])):
        draw.rectangle([x0, 8, x0 + 9 * len(label) + 12, 36], fill="white")
        draw.text((x0 + 6, 14), label, fill="black")
    canvas.save(out_path)


def write_counts_histogram(root, seed_summaries):
    import matplotlib.pyplot as plt

    seeds = sorted(s["seed"] for s in seed_summaries)
    classes = ("parked", "collided clusters", "cone zones")
    placed_by_class = {c: [] for c in classes}
    for seed in seeds:
        s = next(x for x in seed_summaries if x["seed"] == seed)
        pc = s["placement"]
        placed_by_class["parked"].append(pc["parked"]["placed"])
        placed_by_class["collided clusters"].append(pc["collided"]["placed"])
        placed_by_class["cone zones"].append(pc["cones"]["placed"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = np.arange(len(classes))
    width = 0.8 / max(len(seeds), 1)
    for i, seed in enumerate(seeds):
        vals = [placed_by_class[c][i] for c in classes]
        ax.bar(xpos + i * width, vals, width, label=f"seed {seed} (placed)")
    ax.set_xticks(xpos + width * (len(seeds) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("count per env")
    ax.set_title("M1 static hazard placement counts (3 classes × seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "counts_histogram.png", dpi=140)
    plt.close(fig)


def run_static_hazards(args, map_bin):
    """M1: place parked/collided/cone hazards, then re-run the same seed clean
    as the comparison baseline. Co-sim setters only reach vec env 0 -> envs=1."""
    if args.envs != 1:
        print("[hazard-demo] static-hazards: co-sim setters only reach vec env 0; forcing --envs 1")
        args.envs = 1

    root = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    seed_dir = root / f"seed{args.seed}"
    baseline_dir = seed_dir / "baseline" / "frames"
    hazard_dir = seed_dir / "hazard" / "frames"
    compare_dir = seed_dir / "compare"
    for d in (baseline_dir, hazard_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    import pufferlib.viz as viz
    t_start = time.time()
    print(f"[hazard-demo] mode=static-hazards seed={args.seed} map={map_bin}")

    # ---- episode 1: hazards --------------------------------------------------
    slots = max(args.hazard_slots, 1)
    env = build_env(args, map_bin, cosim_partner_slots=slots)
    env.reset(seed=args.seed)
    scenarios = env.get_state()
    buffers = capture_buffers(env, scenarios, extra_agent_slots=slots)
    road_elements = scenarios[0].get("road_elements") or []
    traffic_elements = scenarios[0].get("traffic_elements") or []
    map_corners = scenarios[0].get("map_corners") or []
    sdc_row = int((scenarios[0].get("active_agent_indices") or [0])[0])

    def capture():
        env.get_obs_html_frame(
            buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
            buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
            buffers["rewards_f32"], buffers["coefs_f32"],
        )

    capture()
    lanes = LaneIndex(road_elements)
    edges = EdgeIndex(road_elements)
    dyn = dynamic_boxes(buffers["agent_f32"][0], buffers["agent_i32"][0])
    rng = hazard_rng(args.seed)
    plan = plan_hazard_counts(rng)
    rejects = {r: 0 for r in REJECT_REASONS}
    placed = []
    parked_n = place_parked(rng, lanes, edges, dyn, placed, rejects, plan["parked"])
    clusters_n = place_clusters(rng, lanes, edges, dyn, placed, rejects, plan["clusters"])
    zones_n, cones_n = place_cone_zones(rng, lanes, edges, dyn, placed, rejects, plan["cone_zones"])
    print(f"[hazard-demo] plan={plan} placed: parked={parked_n} clusters={clusters_n} "
          f"cone_zones={zones_n} ({cones_n} cones), slots={len(placed)}/{slots}")
    if len(placed) > slots:
        raise RuntimeError(f"hazard slots exhausted: {len(placed)} > {slots}; raise --hazard-slots")
    if not placed:
        raise RuntimeError("no hazard could be placed on this map/seed")

    idxs = apply_static_hazards(env, args, placed)
    capture()
    legality = verify_static_hazards(env, placed, idxs,
                                     buffers["agent_f32"][0], buffers["agent_i32"][0])
    print(f"[hazard-demo] legality: {legality}")

    static_indices = [e["idx"] for e in placed]
    zone_centers = {}
    for e in placed:
        key = e.get("cluster_id") or e.get("zone_id")
        if key:
            zone_centers.setdefault(key, []).append((e["x"], e["y"]))

    keyframe_paths = []
    episode_end = None
    gif_frames = []
    collision_curve = []  # cumulative latched-collision agent count per step

    def render_keyframes(step):
        scenario = env.get_state()[0]
        img = viz.plot_simulator_state(scenario, step)
        path = hazard_dir / f"env0_t{step:03d}.png"
        Image.fromarray(img).save(path)
        keyframe_paths.append(str(path.relative_to(seed_dir)))

    af0, ai0, mf0 = buffers["agent_f32"][0], buffers["agent_i32"][0], buffers["metrics_f32"][0]
    mask0 = env_active_mask(ai0)
    center = (float(np.mean(af0[mask0, 0])), float(np.mean(af0[mask0, 1])))
    render_keyframes(0)
    gif_frames.append((0, af0.copy(), ai0.copy()))

    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    steps_run = 0
    for step in range(1, args.steps + 1):
        env.step(actions)
        steps_run = step
        capture()
        live = (ai0[:, 3] == 1) & (ai0[:, 2] == 1)
        collision_curve.append(int(np.sum(live & (mf0[:, COLLISION_IDX] > 0))))
        lo, hi = env.agent_offsets[0], env.agent_offsets[1]
        if episode_end is None and bool(np.any(env.truncations[lo:hi])):
            episode_end = step
            print(f"[hazard-demo] env0 hazard episode ended at step {step}")
            break
        if step in HAZARD_KEYFRAMES:
            render_keyframes(step)
        if args.gif_stride > 0 and step % args.gif_stride == 0:
            gif_frames.append((step, af0.copy(), ai0.copy()))

    hazard_offroad = int(np.sum((ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (mf0[:, OFFROAD_IDX] > 0)))
    hazard_metrics = {
        "steps_run": steps_run,
        "ended_step": episode_end if episode_end is not None else f"ran to {steps_run}",
        "collision_agents_final": int(collision_curve[-1]) if collision_curve else 0,
        "offroad_agents_final": hazard_offroad,
    }
    env.close()

    gif_path = seed_dir / "hazard" / "hazard.gif"
    gif_imgs = [
        render_gif_frame(step, af, ai, road_elements, traffic_elements, sdc_row,
                         center, HAZARD_GIF_WINDOW_M, 900, args.dt, args.seed,
                         map_bin.stem, mode_label="static-hazards",
                         static_indices=static_indices)
        for step, af, ai in gif_frames
    ]
    if gif_imgs:
        gif_imgs[0].save(gif_path, save_all=True, append_images=gif_imgs[1:],
                         duration=int(1000 / args.gif_fps), loop=0)

    # ---- episode 2: same-seed clean baseline (comparison side) ---------------
    env_b = build_env(args, map_bin)
    env_b.reset(seed=args.seed)
    scenarios_b = env_b.get_state()
    buffers_b = capture_buffers(env_b, scenarios_b)
    baseline_keyframes = []
    baseline_end = None
    baseline_curve = []

    def capture_b():
        env_b.get_obs_html_frame(
            buffers_b["agent_f32"], buffers_b["agent_i32"], buffers_b["metrics_f32"],
            buffers_b["puffer_f32"], buffers_b["traffic_i16"], buffers_b["goals_f32"],
            buffers_b["rewards_f32"], buffers_b["coefs_f32"],
        )

    capture_b()
    abf, abi, amf = buffers_b["agent_f32"][0], buffers_b["agent_i32"][0], buffers_b["metrics_f32"][0]

    def render_keyframes_b(step):
        scenario = env_b.get_state()[0]
        img = viz.plot_simulator_state(scenario, step)
        path = baseline_dir / f"env0_t{step:03d}.png"
        Image.fromarray(img).save(path)
        baseline_keyframes.append(str(path.relative_to(seed_dir)))

    render_keyframes_b(0)
    for step in range(1, args.steps + 1):
        env_b.step(actions)
        capture_b()
        live = (abi[:, 3] == 1) & (abi[:, 2] == 1)
        baseline_curve.append(int(np.sum(live & (amf[:, COLLISION_IDX] > 0))))
        lo, hi = env_b.agent_offsets[0], env_b.agent_offsets[1]
        if baseline_end is None and bool(np.any(env_b.truncations[lo:hi])):
            baseline_end = step
            break
        if step in COMPARE_KEYFRAMES:
            render_keyframes_b(step)
    baseline_offroad = int(np.sum((abi[:, 3] == 1) & (abi[:, 2] == 1) & (amf[:, OFFROAD_IDX] > 0)))
    baseline_metrics = {
        "steps_run": len(baseline_curve),
        "ended_step": baseline_end if baseline_end is not None else f"ran to {len(baseline_curve)}",
        "collision_agents_final": int(baseline_curve[-1]) if baseline_curve else 0,
        "offroad_agents_final": baseline_offroad,
    }
    env_b.close()

    # ---- comparison artifacts -------------------------------------------------
    missing = []
    for t in COMPARE_KEYFRAMES:
        b_png = baseline_dir / f"env0_t{t:03d}.png"
        h_png = hazard_dir / f"env0_t{t:03d}.png"
        if b_png.is_file() and h_png.is_file():
            side_by_side(b_png, h_png, compare_dir / f"compare_t{t:03d}.png")
        else:
            missing.append(t)
    zooms = []
    for key, pts in zone_centers.items():
        ctr = tuple(np.mean(np.array(pts), axis=0))
        for t in (0, 40):
            b_png = baseline_dir / f"env0_t{t:03d}.png"
            h_png = hazard_dir / f"env0_t{t:03d}.png"
            if not (b_png.is_file() and h_png.is_file()):
                continue
            b_crop = compare_dir / f"_b_{key}_t{t:03d}.png"
            h_crop = compare_dir / f"_h_{key}_t{t:03d}.png"
            crop_world(b_png, map_corners, ctr, ZOOM_WINDOW_M).save(b_crop)
            crop_world(h_png, map_corners, ctr, ZOOM_WINDOW_M).save(h_crop)
            zoom_name = f"zoom_{key}_t{t:03d}.png"
            side_by_side(b_crop, h_crop, compare_dir / zoom_name)
            b_crop.unlink()
            h_crop.unlink()
            zooms.append(f"compare/{zoom_name}")

    window = min(hazard_metrics["steps_run"], baseline_metrics["steps_run"])
    coll_h = collision_curve[window - 1] if window else 0
    coll_b = baseline_curve[window - 1] if window else 0
    denom = args.agents_per_env * max(window, 1)
    rate_h = coll_h / denom
    rate_b = coll_b / denom
    rate_change = (rate_h / rate_b - 1.0) * 100.0 if rate_b > 0 else float("inf")

    placement = {
        "parked": {"planned": plan["parked"], "placed": parked_n},
        "collided": {"planned": plan["clusters"], "placed": clusters_n},
        "cones": {"planned": plan["cone_zones"], "placed": zones_n, "cones_placed": cones_n},
        "reject_reasons": {k: v for k, v in rejects.items() if v},
        "slots_used": len(placed),
    }
    sim = {
        "window_step": window,
        "collision_agents_baseline_at_window": int(coll_b),
        "collision_agents_hazard_at_window": int(coll_h),
        "collision_rate_per_agent_step_baseline": round(float(rate_b), 6),
        "collision_rate_per_agent_step_hazard": round(float(rate_h), 6),
        "collision_rate_change_pct": round(float(rate_change), 1) if np.isfinite(rate_change) else "inf",
        "baseline": baseline_metrics,
        "hazard": hazard_metrics,
    }
    notes = [
        "静态 hazard 走 cosim_partner_slots 静态槽 + set_agent_states/set_agent_sizes 放置，零内核改动。",
        "binding 限制：vec_set_agent_states/sizes 只写 vec->envs[0]（cosim 单 env 假设）→ 本模式固定 --envs 1。",
        "同 seed 保证相同地图与初始撒车布局；hazard 引入后微观轨迹自然分叉（碰撞/绕行）。",
        "橙色 = 静态 hazard（GIF）；keyframe 渲染里静态车为灰色（viz inactive 色）；锥桶 0.5m 全图比例下较小，看 zoom_*。",
        "碰撞簇内部重叠是有意的；簇整体与其他 agent/edge 无重叠（OBB SAT + 内核 lane 关联双重校验）。",
        "lane 宽度 map bin 未暴露，横向偏移按 carla 默认 3.5m 假设；内核 current_lane_idx != -1 为在道证据。",
        "kernel_lane_assoc_failures 口径：贴边锥桶（box 跨 road edge 或超 3×width=1.5m 关联半径）在内核"
        "侧 current_lane_idx=-1；已核实无 gameplay 影响——collision_check 全量遍历含静态槽、"
        "partner obs 的 partner_scan 含全部静态槽（均无 lane 门控，drive.h），offroad 违规只在"
        "active agent 的 compute_metrics 里记（静态槽不走）——仅是在道证据缺失，如实记录不修正。",
    ]
    seed_summary = {
        "mode": "static-hazards",
        "seed": args.seed,
        "map": {"name": map_bin.stem, "path": str(map_bin)},
        "agents_per_env": args.agents_per_env,
        "steps_requested": args.steps,
        "dt": args.dt,
        "placement": placement,
        "legality": legality,
        "simulation": sim,
        "artifacts": {
            "baseline_keyframes": baseline_keyframes,
            "hazard_keyframes": keyframe_paths,
            "hazard_gif": "hazard/hazard.gif" if gif_imgs else None,
            "zooms": zooms,
        },
        "missing_compare_steps": missing,
        "runtime_s": round(time.time() - t_start, 1),
        "notes": notes,
    }
    (seed_dir / "summary.json").write_text(
        json.dumps(seed_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- aggregate artifacts over all seeds run so far -----------------------
    seed_summaries = []
    for d in sorted(root.glob("seed*/summary.json")):
        try:
            seed_summaries.append(json.loads(d.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    write_root_artifacts(root, seed_summaries)
    print(f"[hazard-demo] artifacts -> {root} ({seed_summary['runtime_s']}s); "
          f"collision rate change vs baseline @t={window}: {sim['collision_rate_change_pct']}%")


def write_root_artifacts(root, seed_summaries):
    """Aggregate counts_histogram.png / summary.json / index.html over seed dirs."""
    agg = {
        "mode": "static-hazards",
        "seeds": [s["seed"] for s in seed_summaries],
        "per_seed": [
            {"seed": s["seed"], "placement": s["placement"], "legality": s["legality"],
             "simulation": s["simulation"]}
            for s in sorted(seed_summaries, key=lambda x: x["seed"])
        ],
        "notes": (seed_summaries[-1].get("notes") if seed_summaries else []),
    }
    (root / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not seed_summaries:
        return
    try:
        write_counts_histogram(root, seed_summaries)
    except Exception as exc:  # histogram is cosmetic; never sink the run
        print(f"[hazard-demo] histogram skipped: {exc}")

    rows = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        p, lg, m = s["placement"], s["legality"], s["simulation"]
        cells = (
            f"<td>{p['parked']['placed']}/{p['parked']['planned']}</td>"
            f"<td>{p['collided']['placed']}/{p['collided']['planned']}</td>"
            f"<td>{p['cones']['placed']}/{p['cones']['planned']} ({p['cones']['cones_placed']} cones)</td>"
            f"<td>{json.dumps(p['reject_reasons'])}</td>"
            f"<td>lane-assoc fail={lg['kernel_lane_assoc_failures']} · dyn-overlap={lg['dynamic_overlap_violations_postcheck']}"
            f" · intentional cluster overlaps={lg['intentional_intra_cluster_overlaps']}</td>"
            f"<td>{m['collision_agents_baseline_at_window']} → {m['collision_agents_hazard_at_window']} "
            f"(t={m['window_step']}, {m['collision_rate_change_pct']}%)</td>"
        )
        rows += f"<tr><td>seed {s['seed']}</td>{cells}</tr>"
    compares_html = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        sd = f"seed{s['seed']}"
        imgs = "".join(
            f'<figure><img src="{sd}/compare/{p.name}"/><figcaption>{p.name}</figcaption></figure>'
            for p in sorted((root / sd / "compare").glob("*.png"), key=lambda q: q.name)
            if not p.name.startswith("_")
        )
        gif = (f'<img src="{sd}/hazard/hazard.gif" width="560"/>'
               if (root / sd / "hazard" / "hazard.gif").is_file() else "")
        compares_html += (f"<h2>seed {s['seed']}</h2>"
                          f"<p>左 baseline / 右 static-hazards（同 seed 同视角同范围）</p>{imgs}"
                          f"<h3>跟车 GIF</h3>{gif}\n")
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/><title>Slipstream hazard-demo · M1 static hazards</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1, h2 {{ color: #333; }} table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
  th {{ background: #eee; }} figure {{ margin: 8px; display: inline-block; }}
  img {{ border: 1px solid #ddd; max-width: 640px; }} figcaption {{ font-size: 12px; color: #666; }}
  code {{ background: #eee; padding: 2px 6px; }}
</style></head><body>
  <h1>M1 static hazards · parked / collided clusters / construction cones</h1>
  <p>命令: <code>PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode static-hazards --seed &lt;42|43|44&gt;</code></p>
  <img src="counts_histogram.png" width="560"/>
  <h2>放置计数 / 合法性 / 碰撞率变化</h2>
  <table><tr><th>seed</th><th>parked</th><th>collided</th><th>cone zones</th><th>rejects</th><th>legality</th><th>collision agents (baseline→hazard)</th></tr>
  {rows}</table>
  {compares_html}
  <h2>口径</h2><ul>{''.join(f'<li>{n}</li>' for n in agg['notes'])}</ul>
</body></html>"""
    (root / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    if args.out is None:
        args.out = "artifacts/static-hazards" if args.mode == "static-hazards" else "artifacts/baseline"
    if args.mode == "static-hazards":
        out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        map_bin = resolve_map_bin(args.map_dir, args.map)
        run_static_hazards(args, map_bin)
        return
    t_start = time.time()
    out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    map_bin = resolve_map_bin(args.map_dir, args.map)
    print(f"[hazard-demo] mode={args.mode} seed={args.seed} map={map_bin}")

    env = build_env(args, map_bin)
    print(f"[hazard-demo] env: {env.num_envs} sub-envs x {args.agents_per_env} agents")
    env.reset(seed=args.seed)
    scenarios = env.get_state()
    buffers = capture_buffers(env, scenarios)

    map_corners = scenarios[0].get("map_corners") or []
    sdc_rows = [int((s.get("active_agent_indices") or [0])[0]) for s in scenarios]
    road_elements = scenarios[0].get("road_elements") or []
    traffic_elements = scenarios[0].get("traffic_elements") or []

    keyframes = sorted({int(t) for t in args.keyframes.split(",") if t.strip()} & set(range(args.steps)))
    gif_env = 0  # primary env drives the GIF

    # per-env accumulators
    stats = [
        {"speed_sum": 0.0, "speed_count": 0, "speed_min": float("inf"), "speed_max": 0.0}
        for _ in range(env.num_envs)
    ]
    episode_end = {e: None for e in range(env.num_envs)}  # (step, reason)
    gif_frames = []  # (step, agent_f32 copy, agent_i32 copy)

    def capture_frame():
        env.get_obs_html_frame(
            buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
            buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
            buffers["rewards_f32"], buffers["coefs_f32"],
        )

    def render_keyframes(step):
        for e in range(env.num_envs):
            if episode_end[e] is not None:
                continue  # episode already auto-reset; the snapshot would show a new episode
            scenario = env.get_state()[e]
            img = viz.plot_simulator_state(scenario, step)
            path = frames_dir / f"env{e}_t{step:03d}.png"
            Image.fromarray(img).save(path)
            keyframe_paths[e].append(str(path.relative_to(out_dir)))

    import pufferlib.viz as viz  # upstream renderer for keyframes

    keyframe_paths = {e: [] for e in range(env.num_envs)}
    capture_frame()
    af0 = buffers["agent_f32"][gif_env]
    ai0 = buffers["agent_i32"][gif_env]
    mask0 = env_active_mask(ai0)
    center = (float(np.mean(af0[mask0, 0])), float(np.mean(af0[mask0, 1])))  # follow the traffic centroid
    render_keyframes(0)
    gif_frames.append((0, af0.copy(), ai0.copy()))

    actions = np.zeros((env.num_agents, 2), dtype=np.float32)  # IDM controllers ignore policy actions
    steps_run = 0
    env_infos = []
    for step in range(1, args.steps + 1):
        _, _, terminals, _, info = env.step(actions)
        steps_run = step
        if info:
            env_infos.extend(info)
        capture_frame()
        for e in range(env.num_envs):
            mask = env_active_mask(buffers["agent_i32"][e])
            speeds = buffers["agent_f32"][e][mask, 6]
            s = stats[e]
            s["speed_sum"] += float(np.sum(speeds))
            s["speed_count"] += int(speeds.size)
            if speeds.size:
                s["speed_min"] = min(s["speed_min"], float(np.min(speeds)))
                s["speed_max"] = max(s["speed_max"], float(np.max(speeds)))
        # episode-end detection: episode autoreset (inactive-ratio early reset or scenario_length end)
        # is signalled on the truncations slice. Do NOT use `terminals` here: with behavior=stop the C
        # side raises per-agent terminals for every individually latched agent while the episode continues.
        for e in range(env.num_envs):
            lo, hi = env.agent_offsets[e], env.agent_offsets[e + 1]
            if episode_end[e] is None and bool(np.any(env.truncations[lo:hi])):
                episode_end[e] = (step, "episode truncation (inactive-ratio early reset or scenario_length end)")
        if episode_end[gif_env] is not None:
            print(f"[hazard-demo] env {gif_env} episode ended at step {step}; stopping run")
            break
        if step in keyframes:
            render_keyframes(step)
        if args.gif_stride > 0 and step % args.gif_stride == 0:
            gif_frames.append(
                (step, buffers["agent_f32"][gif_env].copy(), buffers["agent_i32"][gif_env].copy())
            )

    env.close()
    t_loop = time.time()

    # --- assemble artifacts ---
    gif_path = "baseline.gif"
    gif_imgs = [
        render_gif_frame(
            step, af, ai, road_elements, traffic_elements, sdc_rows[gif_env],
            center, args.gif_window_m, 900, args.dt, args.seed, map_bin.stem,
        )
        for step, af, ai in gif_frames
    ]
    if gif_imgs:
        gif_imgs[0].save(
            out_dir / gif_path, save_all=True, append_images=gif_imgs[1:],
            duration=int(1000 / args.gif_fps), loop=0,
        )

    # final captured arrays (kept from the last capture_frame before close)
    per_env, total_collision, total_offroad = [], 0, 0
    for e in range(env.num_envs):
        ai = buffers["agent_i32"][e]
        mf = buffers["metrics_f32"][e]
        af = buffers["agent_f32"][e]
        mask = env_active_mask(ai)
        s = stats[e]
        avg = s["speed_sum"] / s["speed_count"] if s["speed_count"] else 0.0
        collisions = int(np.sum(mask & (mf[:, COLLISION_IDX] > 0)))
        offroads = int(np.sum(mask & (mf[:, OFFROAD_IDX] > 0)))
        total_collision += collisions
        total_offroad += offroads
        ended = episode_end[e]
        per_env.append({
            "agents_final": int(mask.sum()),
            "active_final": int(mask.sum()),
            "collision_agents": collisions,
            "offroad_agents": offroads,
            "removed_final": int(np.sum(ai[mask, 5] == 1)),
            "speed_summary": f"{s['speed_min']:.2f} / {avg:.2f} / {s['speed_max']:.2f}"
                             if s["speed_count"] else "n/a",
            "speed_final": f"{float(np.mean(af[mask, 6])):.2f}" if mask.any() else "n/a",
            "ended_step": ended[0] if ended else f"ran to {steps_run}",
        })

    notes = [
        "所有 agent 走 IDM 控制器 (sdc/non_sdc_controller=idm, controller=CONTROLLER_IDM)；策略动作被忽略。",
        "纯地图 bin：gigaflow 用程序化 spawn 替换 log agents，goal_source=route（gt 会 NULL 崩）。",
        "起步 ~5s 是 0 初速 IDM 加速段（spawn_initial_speed=0），随后进入正常跟车。",
        "collision/offroad/traffic_light behavior=stop：违规 agent latch stopped 并冻结（灰色/红色车），不是路口等灯。",
        "IDM 对红灯停车、绿灯起步；对 stop sign 不停牌（idm.h 上游口径差，见票 07 补充）。",
        "gigaflow spawn 全为 type=1 车辆：图中不会出现行人/骑行者（M2 行人原型另做）。",
        "限速随机化关闭（carla bin 无 SPDZONE1 速度区）。",
    ]

    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "steps_requested": args.steps,
        "steps_run": steps_run,
        "dt": args.dt,
        "map": {"name": map_bin.stem, "path": str(map_bin)},
        "map_size_m": [round(map_corners[2] - map_corners[0], 1), round(map_corners[3] - map_corners[1], 1)]
                      if len(map_corners) >= 4 else None,
        "num_envs": env.num_envs,
        "agents_per_env": args.agents_per_env,
        "traffic_elements": len(traffic_elements),
        "road_elements": len(road_elements),
        "simulation": {
            "simulation_mode": "gigaflow",
            "sdc_controller": "idm",
            "non_sdc_controller": "idm",
            "goal_source": args.goal_source,
            "dynamics_model": "jerk",
            "collision_behavior": "stop",
            "offroad_behavior": "stop",
            "termination_mode": True,
            "inactive_agent_threshold": 0.4,
        },
        "episode_end": {f"env{e}": episode_end[e] for e in range(env.num_envs)},
        "total": {"collision_agents": total_collision, "offroad_agents": total_offroad},
        "per_env": per_env,
        "artifacts": {
            "keyframes": {e: keyframe_paths.get(e, []) for e in sorted(keyframe_paths)},
            "gif": gif_path if gif_imgs else None,
            "index_html": "index.html",
        },
        "runtime_s": {
            "total": round(time.time() - t_start, 1),
            "env_build_and_loop": round(t_loop - t_start, 1),
            "rendering": round(time.time() - t_loop, 1),
        },
        "notes": notes,
    }
    if env_infos:
        summary["episode_logs"] = env_infos
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    write_index_html(out_dir, args, map_bin, summary, keyframe_paths, gif_path)

    print(f"[hazard-demo] steps={steps_run} collisions={total_collision} offroad={total_offroad} "
          f"gif_frames={len(gif_imgs)}")
    print(f"[hazard-demo] artifacts -> {out_dir} ({summary['runtime_s']['total']}s)")


if __name__ == "__main__":
    main()
