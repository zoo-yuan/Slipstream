#!/usr/bin/env python3
"""M0 baseline visualization pipeline (wayfinder ticket 07).

Runs a pure-map gigaflow episode with every agent on the IDM controller,
captures the global agent state every step, and renders top-down artifacts:

- keyframe PNGs reusing the upstream renderer ``pufferlib.viz.plot_simulator_state``
- a cropped follow-cam GIF assembled with pillow
- ``index.html`` (frames + metrics table) and ``summary.json`` under
  ``artifacts/baseline/``

Usage:
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode baseline --seed 42

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
    p.add_argument("--mode", choices=["baseline"], default="baseline",
                   help="demo mode; static-hazards/mix arrive with M1/M2")
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
    p.add_argument("--out", default="artifacts/baseline", help="output directory")
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


def build_env(args, map_bin: Path):
    """Gigaflow env with all agents on IDM; knobs mirror pufferlib/config/puffer_drive.yaml."""
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
        seed=args.seed,
    )


def capture_buffers(env, scenarios):
    """Allocate the vec_get_obs_html_frame output buffers (env, agent, field)."""
    from pufferlib.ocean.drive import binding

    caps = [env.agent_offsets[e + 1] - env.agent_offsets[e] for e in range(env.num_envs)]
    cap = max(caps)
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
                     center, window_m, size_px, dt, seed, map_name):
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
        fill = GIF_COLORS["stopped"] if stopped_latched else (GIF_COLORS["sdc"] if is_sdc else GIF_COLORS["vehicle"])
        draw.polygon(poly, fill=fill, outline="#222222" if is_sdc else "#333333")
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
    draw.text((10, 24), f"IDM gigaflow baseline · {map_name} · seed {seed}", fill=GIF_COLORS["ink"])
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


def main():
    args = parse_args()
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
