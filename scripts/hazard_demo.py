#!/usr/bin/env python3
"""M0 baseline / M1 static-hazard visualization pipeline (wayfinder ticket 07).

Runs a pure-map gigaflow episode with every agent on the IDM controller,
captures the global agent state every step, and renders top-down artifacts:

- full-map keyframe PNGs and the follow-cam GIF from ONE pillow renderer
  (below): class-based coloring — SDC=red, vehicles=blue, latched=gray, static
  hazards tinted by class (parked=navy, collided=dark red, cones=orange diamond)
- ``index.html`` (frames + metrics table) and ``summary.json`` under
  ``artifacts/<mode>/``
- the legend is a standalone ``legend.png`` (first item of index.html); frames
  carry no legend box, only a small top-right "legend: see legend.png" line
- cones draw as diamonds with a forced >=12 px diagonal on full maps (zooms
  re-render at true scale); every image that should show cones is pixel-audited
  into ``summary.json`` ``render_audit`` — any zero count fails the run
- M2 ``jaywalker``: python-stepped 0.5 m pedestrian slots (static co-sim slots
  overwritten via set_agent_states every step BEFORE env.step) cross the road
  mid-block; metrics cover crossing completion, min distance to traffic and
  "obvious slowdown" reactions (same-seed baseline comparison); walker-magenta
  pixels are audited with the same hard-fail mechanism as cones
- M3 ``mix``: config-driven (``config/hazard_demo.yaml``, override with
  ``--hazard-config``) — every class the config enables shares ONE episode
  (static slots placed once, walkers/cut-in stepped per step), fail-fast
  validated before env init; ``enable: false`` degenerates to pure gigaflow
  (bit-identical to baseline — the M3 regression contract). Prototype boundary:
  the knobs drive the demo's co-sim placement only; the real kernel subsystems
  (C reset placement, PEDESTRIAN type, per-agent IDM) belong to the next-phase
  spec ticket. M0/M1/M2 modes keep their frozen hardcoded behavior.

Usage:
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode baseline --seed 42
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode static-hazards --seed 42   # M1: parked/collided-cluster/cones
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode jaywalker --seed 42        # M2: python-stepped crossing + reactions
    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py \
        --mode mix --seed 42              # M3: all config-enabled classes in one scene

Notes (see .wayfinder/research/log-to-map-findings.md):
- Pure-map bins carry no log agents; gigaflow replaces them with procedurally
  spawned agents, so goal_source must be route/map (goal_source="gt" would
  dereference NULL log trajectories).
- ``collision/offroad/traffic_light/stop behavior=stop`` (training config
  defaults): agents that hit an infraction latch ``stopped`` and freeze, which
  is visible in the renders.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # repo pufferlib must shadow the venv copy

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    p = argparse.ArgumentParser(description="M0/M1/M2/M3 hazard visualization pipeline")
    p.add_argument("--mode", choices=["baseline", "static-hazards", "jaywalker", "mix"], default="baseline",
                   help="demo mode: baseline / static-hazards (M1) / jaywalker (M2) / mix (M3 config-driven)")
    p.add_argument("--hazard-slots", type=int, default=48,
                   help="static co-sim partner slots reserved for M1 hazards "
                        "(static-hazards only; mix derives its capacity from the hazard config)")
    p.add_argument("--walkers", type=int, default=None,
                   help="jaywalker walkers placed per seed (jaywalker only, default 2; "
                        "--mode mix reads the count from hazard config jaywalker.n_range)")
    p.add_argument("--hazard-config", default=None,
                   help="hazard config YAML for --mode mix (default: config/hazard_demo.yaml)")
    p.add_argument("--placement", choices=["python", "kernel"], default="python",
                   help="static-hazards placement engine: python = M1 prototype via set_agent_states "
                        "(default, frozen behavior); kernel = K1 C-reset placement via hazard_* env knobs "
                        "(artifacts land under artifacts/kernel-static by default)")
    p.add_argument("--skip-cut-in", action="store_true",
                   help="jaywalker only: do not attempt the scripted cut-in vehicle")
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
                   help="output directory (default: artifacts/<mode> by mode)")
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


# --- kernel provenance (in-repo kernel pinning) ------------------------------
# The demos borrow only the sibling-repo interpreter (PYTHONPATH=$PWD
# ../PufferDrive/.venv/bin/python); pufferlib and the drive binding .so must
# come from THIS repo. Early artifacts ran on the sibling venv's old-kernel
# binary (135-step truncation / 78 collisions seed42); the pinned in-repo
# build runs the full 200 steps / 31 collisions. verify_in_repo_kernel()
# rejects any future import-chain drift before a run starts.
KERNEL_BUILD_COMMAND = "NO_TRAIN=1 python setup.py build_ext --inplace --force"


def verify_in_repo_kernel() -> None:
    """Refuse to run unless the import chain resolves to this repo's kernel."""
    import pufferlib
    import pufferlib.ocean.drive.binding as binding

    pl = Path(pufferlib.__file__).resolve()
    so = Path(binding.__file__).resolve()
    if not pl.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"in-repo kernel required: pufferlib resolved to {pl}, not under {REPO_ROOT}; "
            "run with PYTHONPATH=$PWD so the repo package shadows the venv copy"
        )
    if not so.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"in-repo kernel required: drive binding .so resolved to {so}, not under {REPO_ROOT}; "
            f"rebuild with: {KERNEL_BUILD_COMMAND}"
        )
    print(f"[hazard-demo] kernel pinned: {pl}")
    print(f"[hazard-demo] binding .so: {so} sha256={kernel_binding_sha256()}")


def kernel_binding_sha256() -> str:
    import pufferlib.ocean.drive.binding as binding

    return hashlib.sha256(Path(binding.__file__).resolve().read_bytes()).hexdigest()


def kernel_provenance() -> dict:
    """Provenance block embedded top-level in every mode's summary.json."""
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        git_sha = "unknown"
    import pufferlib
    import pufferlib.ocean.drive.binding as binding
    return {
        "pufferlib_path": str(Path(pufferlib.__file__).resolve()),
        "binding_so": str(Path(binding.__file__).resolve()),
        "binding_sha256": kernel_binding_sha256(),
        "git_sha": git_sha,
        "built_by": KERNEL_BUILD_COMMAND,
    }


def build_env(args, map_bin: Path, cosim_partner_slots: int = 0, hazard_cfg=None):
    """Gigaflow env with all agents on IDM; knobs mirror pufferlib/config/puffer_drive.yaml.

    cosim_partner_slots > 0 appends CONTROLLER_STATIC vehicle slots after the
    gigaflow-spawned active agents (never spawned/moved by c_step, but included
    in collision/TTC/partner-obs loops). Slot init consumes no env RNG, so the
    dynamic spawn stream for a given seed is identical to the baseline run.
    hazard_cfg (validated config/hazard_demo.yaml dict) maps onto the K1
    hazard_* env kwargs: placement then happens INSIDE the C reset path.
    """
    from pufferlib.ocean.drive.drive import Drive

    hazard_kwargs = {}
    if hazard_cfg is not None:
        pc, cc, kc = hazard_cfg["parked"], hazard_cfg["clusters"], hazard_cfg["cones"]
        hazard_kwargs = {
            "hazard_enable": bool(hazard_cfg["enable"]),
            "hazard_parked_enable": bool(pc["enable"]),
            "hazard_parked_n_lo": int(pc["n_range"][0]),
            "hazard_parked_n_hi": int(pc["n_range"][1]),
            # weights arrive aligned with the config sizes subset; the kernel knob
            # face is the full car/bus/truck triple, so missing sizes get weight 0
            "hazard_parked_car_weight": pc["size_weights"][pc["sizes"].index("car")] if "car" in pc["sizes"] else 0.0,
            "hazard_parked_bus_weight": pc["size_weights"][pc["sizes"].index("bus")] if "bus" in pc["sizes"] else 0.0,
            "hazard_parked_truck_weight": pc["size_weights"][pc["sizes"].index("truck")] if "truck" in pc["sizes"] else 0.0,
            "hazard_clusters_enable": bool(cc["enable"]),
            "hazard_clusters_n_lo": int(cc["n_range"][0]),
            "hazard_clusters_n_hi": int(cc["n_range"][1]),
            "hazard_clusters_layout": (
                "both" if set(cc["layouts"]) == set(CLUSTER_LAYOUTS) else cc["layouts"][0]
            ),
            "hazard_cones_enable": bool(kc["enable"]),
            "hazard_cones_zones_lo": int(kc["zones_range"][0]),
            "hazard_cones_zones_hi": int(kc["zones_range"][1]),
            "hazard_cones_taper_cones": int(kc["taper_cones"]),
            "hazard_cones_block_rows": int(kc["block_rows"]),
            "hazard_cones_block_cols": int(kc["block_cols"]),
            "hazard_cones_spacing_m": float(kc["spacing_m"]),
            "hazard_cones_taper_prob": float(kc["taper_prob"]),
            # K2 in-kernel scripted crossing walkers (kernel knob face of the jaywalker section)
            "hazard_walkers_enable": bool(hazard_cfg["jaywalker"]["enable"]),
            "hazard_walkers_n_lo": int(hazard_cfg["jaywalker"]["n_range"][0]),
            "hazard_walkers_n_hi": int(hazard_cfg["jaywalker"]["n_range"][1]),
            "hazard_walkers_speed_lo_mps": float(hazard_cfg["jaywalker"]["speed_range"][0]),
            "hazard_walkers_speed_hi_mps": float(hazard_cfg["jaywalker"]["speed_range"][1]),
        }

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
        **hazard_kwargs,
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
    "walker": "#FF00FF",  # M2 jaywalker: magenta dot + trail (never spawned by baseline/static modes)
    "ink": "#222222",
}
TRAFFIC_STATE_COLORS = {0: "#8c8c8c", 1: "#ff0000", 2: "#ffee00", 3: "#00cc00", 4: "#8c8c8c"}
# traffic control states: constants.h TRAFFIC_CONTROL_STATE_{UNKNOWN,RED,YELLOW,GREEN,OFF}

FULL_MAP_PX = 2000       # full-map PNG side length in px
CONE_MIN_DIAM_PX = 12    # forced minimum cone diamond diagonal on full maps
ZOOM_PX = 900            # zoom render side length (true-scale cones)


def _world_to_px(x, y, center, scale, size_px):
    px = (x - center[0]) * scale + size_px / 2
    py = size_px / 2 - (y - center[1]) * scale  # image y axis points down
    return px, py


def _polyline(element, key):
    xs, ys = element.get("x"), element.get("y")
    return [(float(a), float(b)) for a, b in zip(xs or [], ys or [])]


def render_gif_frame(step, agent_f32, agent_i32, road_elements, traffic_elements, sdc_row,
                     center, window_m, size_px, dt, seed, map_name,
                     mode_label="baseline", static_indices=None, static_class_of=None,
                     min_cone_px=CONE_MIN_DIAM_PX, walker_indices=None, walker_trails=None,
                     walker_min_px=None):
    """Top-down frame; the single renderer for full-map keyframes and GIFs.

    Class-based coloring (owner acceptance 2026-11-05 R2): SDC red, vehicles
    blue, latched gray, static slots tinted by ``static_class_of``. Cones draw
    as a diamond marker with diagonal max(min_cone_px, true size); pass
    min_cone_px=0 for true-scale zoom renders. Walkers (M2 jaywalker, kernel
    type still 1) draw as magenta dots with a forced minimum diameter on full
    maps (walker_min_px=None -> WALKER_MIN_DIAM_PX; 0 -> true scale in zooms)
    plus their trajectory polyline up to ``step``. No in-frame legend box: the
    HUD ends with a "legend: see legend.png" pointer to the standalone image."""
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

    # walker trails: magenta polyline up to the current step (trails.png is the
    # final frame carrying the COMPLETE trajectory per the M2 render contract)
    for row, pts in (walker_trails or {}).items():
        seg = pts[:step + 1]
        if len(seg) >= 2:
            draw.line([_world_to_px(float(p[0]), float(p[1]), center, scale, size_px)
                       for p in seg], fill=GIF_COLORS["walker"], width=2)

    # agents as oriented rectangles; color by type/controller role
    active = env_active_mask(agent_i32)
    if static_indices:
        sel = np.zeros(agent_i32.shape[0], dtype=bool)
        sel[np.asarray(static_indices, dtype=int)] = True
        active = active | (sel & (agent_i32[:, 2] == 1) & (agent_i32[:, 5] == 0))
    static_set = set(int(i) for i in (static_indices or ()))
    walker_set = set(int(i) for i in (walker_indices or ()))
    min_walker_px = WALKER_MIN_DIAM_PX if walker_min_px is None else float(walker_min_px)
    for i in np.flatnonzero(active):
        x, y = float(agent_f32[i, 0]), float(agent_f32[i, 1])
        heading = float(agent_f32[i, 3])
        length = max(float(agent_f32[i, 4]), 0.5)
        width = max(float(agent_f32[i, 5]), 0.5)
        atype = int(agent_i32[i, 1])
        stopped_latched = int(agent_i32[i, 4]) == 1
        is_sdc = i == sdc_row
        cx, cy = _world_to_px(x, y, center, scale, size_px)
        if i in walker_set:
            # python-stepped jaywalker (0.5 m slot, kernel type 1): magenta dot,
            # forced minimum diameter on full maps (sub-meter marker rule)
            r = max(min_walker_px / 2.0, 0.25 * scale)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GIF_COLORS["walker"],
                         outline=(0, 0, 0))
            continue
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
        if i in static_set and (static_class_of or {}).get(i) == "cone":
            # 0.5 m cones are sub-pixel on full maps: draw a diamond marker with
            # diagonal max(min_cone_px, true size); min_cone_px=0 -> true scale
            ccx, ccy = _world_to_px(x, y, center, scale, size_px)
            diag = max(float(min_cone_px), length * scale)
            r = diag / 2.0
            draw.polygon([(ccx, ccy - r), (ccx + r, ccy), (ccx, ccy + r), (ccx - r, ccy)],
                         fill=HAZARD_CLASS_COLORS["cone"], outline=(0, 0, 0),
                         width=2 if diag >= 16 else 1)
            continue
        if i in static_set:
            fill = HAZARD_CLASS_COLORS[(static_class_of or {}).get(i, "cone")]
        else:
            fill = GIF_COLORS["stopped"] if stopped_latched else (GIF_COLORS["sdc"] if is_sdc else GIF_COLORS["vehicle"])
        draw.polygon(poly, fill=fill, outline="#222222" if is_sdc or i in static_set else "#333333")
        # heading tick
        tipx, tipy = _world_to_px(x + 0.6 * length * cos_h, y + 0.6 * length * sin_h, center, scale, size_px)
        draw.line([poly[0], (tipx, tipy)], fill="#222222", width=1)

    # HUD + scale bar (top-right only: the legend lives in the standalone
    # legend.png; frames end the HUD with a pointer to it)
    speeds = agent_f32[active, 6]
    mean_v = float(np.mean(speeds)) if speeds.size else 0.0
    n_stopped = int(np.sum(agent_i32[active, 4] == 1))
    n_removed = int(np.sum(agent_i32[active, 5] == 1))
    hud = [f"t={step} ({step * dt:.1f}s)  agents={int(active.sum())}  "
           f"mean_v={mean_v:.1f}m/s  latched={n_stopped}  removed={n_removed}",
           f"IDM gigaflow {mode_label} · {map_name} · seed {seed}"]
    if static_class_of:
        counts = {k: sum(1 for v in static_class_of.values() if v == k)
                  for k in ("parked", "collided", "cone")}
        hud.append(f"hazards: parked={counts['parked']} cluster={counts['collided']} "
                   f"cones={counts['cone']}")
        nw = sum(1 for v in static_class_of.values() if v == "walker")
        nc = sum(1 for v in static_class_of.values() if v == "cut_in")
        if nw or nc:
            stepping = ("kernel-stepped, PEDESTRIAN type" if mode_label == "kernel-jaywalker"
                        else "python-stepped")
            hud.append(f"walkers={nw} cut_in={nc} ({stepping})")
    hud.append("legend: see legend.png")
    for i, line in enumerate(hud):
        x = size_px - int(draw.textlength(line)) - 10
        draw.text((x, 8 + i * 16), line, fill=GIF_COLORS["ink"])
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
  <h2>图例 (legend.png — 渲染帧内不再内嵌图例，帧右上角仅一行小字指认)</h2>
  <img src="legend.png" width="460"/>
  <p>地图: <code>{map_bin}</code> · seed={args.seed} · {episode_note}</p></p>
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


# --- M3 hazard config face (prototype knobs, Python side) --------------------
#
# Ticket 07 M3: the hazard knobs move into config/hazard_demo.yaml and are
# fail-fast validated BEFORE the env is built (mirrors the kernel trust
# boundary: validate external config before init). Prototype boundary: these
# knobs drive the demo's co-sim placement logic only (--mode mix); the real
# kernel subsystems (C reset placement, PEDESTRIAN type, per-agent IDM) belong
# to the next-phase spec ticket. The static-hazards / jaywalker modes keep
# their frozen M1/M2 hardcoded behavior; defaults below reproduce exactly
# those numbers so the config face can never silently rewrite prior modes.

DEFAULT_HAZARD_CONFIG_PATH = "config/hazard_demo.yaml"
CLUSTER_LAYOUTS = ("rear_end", "t_bone")

HAZARD_CONFIG_DEFAULTS = {
    "enable": True,
    "parked": {"enable": True, "n_range": [6, 10],
               "sizes": ["car", "bus", "truck"],
               "size_weights": [0.6, 0.2, 0.2]},
    "clusters": {"enable": True, "n_range": [1, 2],
                 "layouts": ["rear_end", "t_bone"]},
    "cones": {"enable": True, "zones_range": [1, 2], "taper_cones": 8,
              "block_rows": 2, "block_cols": 7, "spacing_m": 1.5,
              "taper_prob": 0.5},
    "jaywalker": {"enable": True, "n_range": [1, 2], "speed_range": [1.2, 1.6]},
    "cut_in": {"enable": False},
}
HAZARD_CONFIG_KEYS = {cls: set(sec) for cls, sec in HAZARD_CONFIG_DEFAULTS.items()
                      if isinstance(sec, dict)}  # top-level 'enable' is a scalar, not a section


class HazardConfigError(ValueError):
    """Raised for any missing/malformed/out-of-range hazard config value."""


def _cfg_fail(key, value, why):
    raise HazardConfigError(f"hazard config '{key}': {why} (got {value!r})")


def _cfg_bool(key, v):
    if not isinstance(v, bool):
        _cfg_fail(key, v, "must be a bool (true/false)")
    return v


def _cfg_int(key, v, minimum=None):
    # bool is an int subclass in Python — reject it explicitly
    if isinstance(v, bool) or not isinstance(v, int):
        _cfg_fail(key, v, "must be an int")
    if minimum is not None and v < minimum:
        _cfg_fail(key, v, f"must be >= {minimum}")
    return v


def _cfg_num(key, v, minimum=None, maximum=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _cfg_fail(key, v, "must be a number")
    v = float(v)
    if minimum is not None and v < minimum:
        _cfg_fail(key, v, f"must be >= {minimum}")
    if maximum is not None and v > maximum:
        _cfg_fail(key, v, f"must be <= {maximum}")
    return v


def _cfg_int_range(key, v, minimum=None):
    """[lo, hi] integer range, inclusive at both ends."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        _cfg_fail(key, v, "must be a [lo, hi] pair of ints")
    lo = _cfg_int(f"{key}[0]", v[0], minimum=minimum)
    hi = _cfg_int(f"{key}[1]", v[1], minimum=minimum)
    if lo > hi:
        _cfg_fail(key, v, f"lo must be <= hi ([{lo}, {hi}])")
    return [lo, hi]


def _cfg_num_range(key, v):
    """[lo, hi] number range with 0 < lo <= hi (e.g. walker speed m/s)."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        _cfg_fail(key, v, "must be a [lo, hi] pair of numbers")
    lo = _cfg_num(f"{key}[0]", v[0])
    hi = _cfg_num(f"{key}[1]", v[1])
    if lo <= 0:
        _cfg_fail(key, v, f"lo must be > 0 ({lo})")
    if lo > hi:
        _cfg_fail(key, v, f"lo must be <= hi ([{lo}, {hi}])")
    return [lo, hi]


def validate_hazard_config(raw):
    """Merge ``raw`` over the built-in defaults and fail-fast on anything
    malformed, out of range, or of the wrong type. Returns the normalized
    config dict (no defaults left implicit, so downstream placement code can
    read every knob unconditionally). Independent of the kernel — unit-tested
    in tests/test_hazard_demo.py."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _cfg_fail("<root>", raw, "must be a YAML mapping")
    unknown = sorted(set(raw) - set(HAZARD_CONFIG_DEFAULTS))
    if unknown:
        _cfg_fail("<root>", unknown,
                  f"unknown key(s); allowed: {sorted(HAZARD_CONFIG_DEFAULTS)}")

    def section(name):
        sec_raw = raw.get(name, {})
        if sec_raw is None:
            sec_raw = {}
        if not isinstance(sec_raw, dict):
            _cfg_fail(name, sec_raw, "must be a mapping")
        unknown_k = sorted(set(sec_raw) - HAZARD_CONFIG_KEYS[name])
        if unknown_k:
            _cfg_fail(name, unknown_k,
                      f"unknown key(s); allowed: {sorted(HAZARD_CONFIG_KEYS[name])}")
        merged = dict(HAZARD_CONFIG_DEFAULTS[name])
        merged.update(sec_raw)
        return merged, sec_raw

    cfg = {"enable": _cfg_bool("enable", raw.get("enable", True))}

    parked, sec_raw = section("parked")
    parked["enable"] = _cfg_bool("parked.enable", parked["enable"])
    parked["n_range"] = _cfg_int_range("parked.n_range", parked["n_range"], minimum=0)
    sizes = parked["sizes"]
    if (not isinstance(sizes, list) or not sizes
            or any(s not in VEHICLE_PRESETS for s in sizes)):
        _cfg_fail("parked.sizes", sizes,
                  f"must be a non-empty subset of {sorted(VEHICLE_PRESETS)}")
    if "sizes" in sec_raw and "size_weights" not in sec_raw:
        parked["size_weights"] = [1.0 / len(sizes)] * len(sizes)  # uniform draw
    w = parked["size_weights"]
    if (not isinstance(w, list) or len(w) != len(sizes)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0
                   for x in w)):
        _cfg_fail("parked.size_weights", w,
                  f"must be {len(sizes)} non-negative numbers aligned with sizes")
    if abs(sum(float(x) for x in w) - 1.0) > 1e-6:
        _cfg_fail("parked.size_weights", w,
                  f"must sum to 1.0 (sums to {sum(float(x) for x in w)})")  # noqa: partial dup ok
    parked["size_weights"] = [float(x) for x in w]
    cfg["parked"] = parked

    clusters, _ = section("clusters")
    clusters["enable"] = _cfg_bool("clusters.enable", clusters["enable"])
    clusters["n_range"] = _cfg_int_range("clusters.n_range", clusters["n_range"], minimum=0)
    layouts = clusters["layouts"]
    if (not isinstance(layouts, list) or not layouts
            or any(x not in CLUSTER_LAYOUTS for x in layouts)):
        _cfg_fail("clusters.layouts", layouts,
                  f"must be a non-empty subset of {list(CLUSTER_LAYOUTS)}")
    cfg["clusters"] = clusters

    cones, _ = section("cones")
    cones["enable"] = _cfg_bool("cones.enable", cones["enable"])
    cones["zones_range"] = _cfg_int_range("cones.zones_range", cones["zones_range"], minimum=0)
    cones["taper_cones"] = _cfg_int("cones.taper_cones", cones["taper_cones"], minimum=2)
    cones["block_rows"] = _cfg_int("cones.block_rows", cones["block_rows"], minimum=1)
    cones["block_cols"] = _cfg_int("cones.block_cols", cones["block_cols"], minimum=2)
    cones["spacing_m"] = _cfg_num("cones.spacing_m", cones["spacing_m"], minimum=1e-6)
    cones["taper_prob"] = _cfg_num("cones.taper_prob", cones["taper_prob"], minimum=0.0, maximum=1.0)
    cfg["cones"] = cones

    jw, _ = section("jaywalker")
    jw["enable"] = _cfg_bool("jaywalker.enable", jw["enable"])
    jw["n_range"] = _cfg_int_range("jaywalker.n_range", jw["n_range"], minimum=0)
    jw["speed_range"] = _cfg_num_range("jaywalker.speed_range", jw["speed_range"])
    cfg["jaywalker"] = jw

    cut, _ = section("cut_in")
    cut["enable"] = _cfg_bool("cut_in.enable", cut["enable"])
    cfg["cut_in"] = cut
    return cfg


def load_hazard_config(path=None):
    """Read + validate the hazard config YAML (init 前校验, fail-fast)."""
    path = Path(path or DEFAULT_HAZARD_CONFIG_PATH)
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.is_file():
        raise HazardConfigError(
            f"hazard config not found: {resolved} (pass --hazard-config <yaml>)")
    try:
        import yaml
    except ImportError as exc:
        raise HazardConfigError("PyYAML is required to read the hazard config") from exc
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HazardConfigError(f"hazard config {resolved} is not valid YAML: {exc}") from exc
    cfg = validate_hazard_config(raw)
    cfg["_path"] = str(resolved)
    return cfg


def mix_slot_capacity(cfg):
    """Upper bound of co-sim slots the config can ever consume (planned-count
    highs; placement may use fewer, never more). All-disabled configs get 0 so
    the mix build is byte-identical to the plain baseline build."""
    if not cfg["enable"]:
        return 0
    pc, cc, kc = cfg["parked"], cfg["clusters"], cfg["cones"]
    cap = 0
    if pc["enable"]:
        cap += pc["n_range"][1]
    if cc["enable"]:
        cap += 2 * cc["n_range"][1]
    if kc["enable"]:
        cap += kc["zones_range"][1] * max(kc["taper_cones"], kc["block_rows"] * kc["block_cols"])
    if cfg["jaywalker"]["enable"]:
        cap += cfg["jaywalker"]["n_range"][1]
    if cfg["cut_in"]["enable"]:
        cap += 1
    return cap


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


def plan_hazard_counts(rng, parked_range=(6, 10), cluster_range=(1, 2), zone_range=(1, 2)):
    """Per-class counts drawn from inclusive [lo, hi] ranges. Defaults are the
    frozen M1 numbers; --mode mix passes the validated config ranges. Disabled
    classes arrive as (0, 0) and draw 0 (one rng draw each — the placement RNG
    is independent of the episode stream, so this never touches dynamics)."""
    return {
        "parked": int(rng.randint(parked_range[0], parked_range[1] + 1)),
        "clusters": int(rng.randint(cluster_range[0], cluster_range[1] + 1)),
        "cone_zones": int(rng.randint(zone_range[0], zone_range[1] + 1)),
    }


def _pick_preset(rng, size_table=None):
    table = size_table or PARKED_SIZE_PROBS
    names = [t[0] for t in table]
    # M1 table rows are (name, prob); M3 config tables are (name, size, weight)
    probs = [t[-1] for t in table]
    return names[int(rng.choice(len(names), p=probs))]


def place_parked(rng, lanes, edges, dyn, placed, rejects, count, size_table=None):
    """``size_table``: ((name, (length, width), weight), ...) — config sizes
    set (M3 mix); None keeps the frozen M1 car/bus/truck draw."""
    placed_n = 0
    for _ in range(count):
        vtype = _pick_preset(rng, size_table)
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


def _cluster_poses(rng, lanes, lane, lenA, wA, lenB, wB, layouts=None):
    """Two deliberately overlapping poses per the config layout set (M3 mix);
    default/both layouts keep the frozen M1 uniform coin (same rng stream)."""
    layouts = list(CLUSTER_LAYOUTS) if layouts is None else list(layouts)
    if set(layouts) == set(CLUSTER_LAYOUTS):
        pattern = "rear_end" if rng.uniform() < 0.5 else "t_bone"
    else:
        pattern = layouts[0]  # single-layout config: no coin drawn
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


def place_clusters(rng, lanes, edges, dyn, placed, rejects, count, layouts=None):
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
            pattern, poseA, poseB, z = _cluster_poses(rng, lanes, lane, lenA, wA, lenB, wB,
                                                      layouts=layouts)
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


def _cone_poses(rng, lanes, lane, cones=None):
    """Taper (diagonal narrowing) or block (full-lane closure) at ``spacing_m``
    intervals. ``cones`` is the validated config section (M3 mix); None keeps
    the frozen M1 defaults: 8-cone taper / 2x7 block at 1.5 m, 50/50 patterns."""
    taper_n = int(cones["taper_cones"]) if cones else 8
    block_rows = int(cones["block_rows"]) if cones else 2
    block_cols = int(cones["block_cols"]) if cones else 7
    spacing = float(cones["spacing_m"]) if cones else 1.5
    taper_prob = float(cones["taper_prob"]) if cones else 0.5
    pattern = "taper" if rng.uniform() < taper_prob else "block"
    cone_l = cone_w = 0.5
    poses = []
    if pattern == "taper":
        n = taper_n
        span = (n - 1) * spacing
        base_s = rng.uniform(4.0, max(4.1, lane["total"] - span - 4.0))
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        lat0 = side * rng.uniform(0.2, 0.5)
        lat1 = side * rng.uniform(1.1, 1.4)
        for k in range(n):
            s = base_s + k * spacing
            lat = float(np.interp(k, [0, n - 1], [lat0, lat1])) + rng.uniform(-0.05, 0.05)
            x, y, z, h = lanes.pose_at(lane, s, lat)
            poses.append(((x, y, h, cone_l, cone_w), z, s, lat))
    else:
        # kernel lane-association radius is 3×width (0.5m cone → 1.5m); keep the
        # outermost cone center within 1.45m of the lane centerline so every cone
        # stays kernel-associated (on-lane evidence for the legality report)
        half_lane = min(ASSUMED_LANE_WIDTH_M / 2.0 - 0.35, 1.45)
        base_s = rng.uniform(4.0, max(4.1, lane["total"] - 9.0))
        stagger = (2 * half_lane) / (block_cols - 1) / 2
        for row in range(block_rows):
            lats = np.linspace(-half_lane, half_lane, block_cols) + (stagger if row else 0.0)
            lats = np.clip(lats, -1.45, 1.45)
            for lat in lats:
                # row gap = spacing + cone diameter margin (M1 default: 1.6 m)
                s = base_s + row * (spacing + 0.1) + rng.uniform(-0.1, 0.1)
                x, y, z, h = lanes.pose_at(lane, s, float(lat) + rng.uniform(-0.05, 0.05))
                poses.append(((x, y, h, cone_l, cone_w), z, s, float(lat)))
    return pattern, poses


def place_cone_zones(rng, lanes, edges, dyn, placed, rejects, count, cones=None):
    placed_zones, cones_placed = 0, 0
    for zi in range(count):
        zone_id = f"cones{zi}"
        zone_entries = None
        for _attempt in range(MAX_PLACE_ATTEMPTS):
            lane = lanes.pick(rng, min_len_m=20.0, straight_only=True)
            if lane is None:
                rejects["no_lane"] += 1
                break
            pattern, poses = _cone_poses(rng, lanes, lane, cones=cones)
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


# --- K1 kernel placement (C-reset hazard engine, readback + audit helpers) ---

def kernel_slot_capacity(cfg):
    """Static-slot budget for the three kernel-placed classes (planned-count
    highs; matches drive.py Drive.hazard_slot_capacity). Jaywalker/cut_in stay
    python-stepped and are not part of the kernel placement budget."""
    if not cfg["enable"]:
        return 0
    cap = 0
    if cfg["parked"]["enable"]:
        cap += cfg["parked"]["n_range"][1]
    if cfg["clusters"]["enable"]:
        cap += 2 * cfg["clusters"]["n_range"][1]
    if cfg["cones"]["enable"]:
        kc = cfg["cones"]
        cap += kc["zones_range"][1] * max(kc["taper_cones"], kc["block_rows"] * kc["block_cols"])
    return cap


def hazards_from_kernel_readback(env, buffers, n_dyn):
    """Reconstruct the kernel-placed hazards from the html frame readback.

    The C reset path owns the static slots, so the demo never writes poses:
    placed slots = sim_valid & not-removed rows at [n_dyn, n_dyn + slots).
    Classification by geometry: length < 1 m -> cone; overlapping car pair
    (margin 0) -> intentional collision cluster; otherwise parked. Cones are
    chained into zone ids by 2.5 m proximity for callouts/zooms."""
    af, ai = buffers["agent_f32"][0], buffers["agent_i32"][0]
    entries = []
    for k in range(env.cosim_partner_slots):
        row = n_dyn + k
        if row >= ai.shape[0] or ai[row, 2] != 1 or ai[row, 5] != 0:
            continue
        x, y, heading, length, width = (float(v) for v in af[row, [0, 1, 3, 4, 5]])
        entries.append({"kind": None, "vtype": "cone", "x": x, "y": y, "z": float(af[row, 2]),
                        "heading": heading, "length": length, "width": width,
                        "lane_id": None, "s": None, "lateral": None, "idx": int(row)})

    for e in entries:
        if e["length"] < 1.0:
            e["kind"] = "cone"
    cars = [e for e in entries if e["kind"] is None]
    used = set()
    for i, a in enumerate(cars):
        if i in used:
            continue
        pose_a = (a["x"], a["y"], a["heading"], a["length"], a["width"])
        for j in range(i + 1, len(cars)):
            if j in used:
                continue
            b = cars[j]
            pose_b = (b["x"], b["y"], b["heading"], b["length"], b["width"])
            if obb_overlap(pose_a, pose_b, margin=0.0):
                key = f"cluster{len(used) // 2}"
                a["kind"] = b["kind"] = "collided"
                a["vtype"] = b["vtype"] = "crash"
                a["cluster_id"] = b["cluster_id"] = key
                used.update({i, j})
                break
    for e in entries:
        if e["kind"] is None:
            e["kind"] = "parked"
            e["vtype"] = "car"

    cones = [e for e in entries if e["kind"] == "cone"]
    unassigned = set(range(len(cones)))
    zone_id = 0
    while unassigned:
        seed_i = min(unassigned)
        unassigned.discard(seed_i)
        key = f"cones{zone_id}"
        zone_id += 1
        stack, comp = [seed_i], [seed_i]
        while stack:
            cur = stack.pop()
            for o in list(unassigned):
                if np.hypot(cones[cur]["x"] - cones[o]["x"], cones[cur]["y"] - cones[o]["y"]) <= 2.5:
                    unassigned.discard(o)
                    stack.append(o)
                    comp.append(o)
        for i in comp:
            cones[i]["zone_id"] = key
    return entries


def kernel_edge_audit(placed, edges):
    """Prototype-consistency off-road postcheck: no hazard box may cross a road
    edge under the Python EdgeIndex rule (the kernel's own check_spawn_offroad
    already rejected such poses at placement, so 0 is the only passing value)."""
    return sum(1 for e in placed
               if edges.blocked((e["x"], e["y"], e["heading"], e["length"], e["width"])))


def load_python_seed_summary(seed, root="artifacts/static-hazards"):
    """Legality evidence of the frozen M1 python placement for the same seed
    (produced by --placement python runs recorded under artifacts/)."""
    path = REPO_ROOT / root / f"seed{seed}" / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- legend, class coloring & pixel audit (ticket 07 owner acceptance R2) ----
#
# Owner feedback round 2: in-frame legends overlapped the map; the 18-color
# agent-id palette (upstream viz.py) read as pedestrians/cyclists; 0.5 m cones
# were sub-pixel. Response: ONE pillow renderer (render_gif_frame) for ALL
# images — full-map keyframes and GIF frames share class coloring; the legend
# is a standalone legend.png (first item of index.html); cones draw as diamonds
# with a forced minimum diagonal on full maps while zooms re-render at true
# scale; every image that should show cones is pixel-audited into summary
# render_audit (zero cone pixels in any audited image fails the run).
HAZARD_CLASS_COLORS = {
    "cone": "#FF8C00",     # orange; GIF cyclist (#ff8c00) never spawns in gigaflow
    "parked": "#0D1F8C",   # deep navy, clearly darker than vehicle steel blue
    "collided": "#8B0000",  # dark red vs SDC #d62728 / latched gray
    "cut_in": "#7B2FBF",   # M2 jaywalker scripted merge vehicle (purple; distinct from blue/red/gray)
}
HAZARD_CLASS_LABELS = {
    "parked": ("parked", "路侧停车"),
    "collided": ("cluster", "碰撞簇"),
    "cone": ("cones", "锥桶"),
}


def detect_cjk_font():
    """Path of a CJK-capable font, or None (labels fall back to English only,
    so no tofu blocks ever render). Probes fc-list :lang=zh, then known paths."""
    try:
        out = subprocess.run(["fc-list", ":lang=zh", "file"], capture_output=True,
                             text=True, timeout=10).stdout
        for line in out.splitlines():
            cand = line.strip().rstrip(":")
            if cand and cand.lower().endswith((".ttf", ".ttc", ".otf")) and Path(cand).is_file():
                return cand
    except (OSError, subprocess.SubprocessError):
        pass
    known = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    )
    for p in known:
        if Path(p).is_file():
            return p
    return None


def _load_font(size_px, cjk_path):
    if cjk_path:
        try:
            return ImageFont.truetype(cjk_path, size_px)
        except OSError:
            pass  # fall through to the bitmap default (ASCII-only)
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def _label(entry, cjk_path):
    return entry["en"] + (f" {entry['zh']}" if cjk_path and entry.get("zh") else "")


def hazard_legend_entries():
    return [
        {"sw": "diamond", "cls": "cone", "colors": [HAZARD_CLASS_COLORS["cone"]],
         "en": "cones: construction cone (orange diamond, >=12 px full map)",
         "zh": "锥桶"},
        {"sw": "rect", "cls": "parked", "colors": [HAZARD_CLASS_COLORS["parked"]],
         "en": "parked: kerbside static vehicle", "zh": "路侧停车"},
        {"sw": "rect", "cls": "collided", "colors": [HAZARD_CLASS_COLORS["collided"]],
         "en": "cluster: collided pair (overlap intentional)", "zh": "碰撞簇"},
    ]


def walker_legend_entries(kernel_mode=False):
    text = ("walker: kernel-stepped jaywalker (true PEDESTRIAN type, dot + trail)"
            if kernel_mode else "walker: python-stepped jaywalker (dot + trail)")
    zh = "行人（内核步进·真 PEDESTRIAN）" if kernel_mode else "行人（轨迹线）"
    return [
        {"sw": "traildot", "cls": "walker", "colors": [GIF_COLORS["walker"]],
         "en": text, "zh": zh},
    ] + ([] if kernel_mode else [
        {"sw": "rect", "cls": "cut_in", "colors": [HAZARD_CLASS_COLORS["cut_in"]],
         "en": "cut-in maker: scripted merging vehicle", "zh": "cut-in 车"},
    ])


def legend_entries(mode, spawned=None):
    """Rows for the standalone legend.png. Mirrors render_gif_frame 1:1 (it is
    now the only renderer, so one legend covers full-map PNGs and GIF frames).
    Categories that never spawn are NOT listed (owner acceptance rule):
    gigaflow spawns type-1 vehicles only, so the baseline/static legends have
    no pedestrian/cyclist rows; ``spawned`` (a set of class keys) additionally
    filters the mix-mode rows down to classes that actually spawned this run
    (e.g. cut_in rows are omitted whenever cut_in.enable is false). No goal
    ring / agent-id rows either (this renderer draws neither)."""
    ent = []
    if mode == "static-hazards":
        ent += hazard_legend_entries()
    if mode == "jaywalker":
        ent += walker_legend_entries()
    if mode == "kernel-jaywalker":
        ent += walker_legend_entries(kernel_mode=True)
    if mode == "mix":
        rows = hazard_legend_entries() + walker_legend_entries()
        ent += [e for e in rows if spawned is None or e.get("cls") in spawned]
    ent += [
        {"sw": "rect", "colors": [GIF_COLORS["vehicle"]], "en": "vehicle (IDM)", "zh": "车辆"},
        {"sw": "rect", "colors": [GIF_COLORS["sdc"]], "en": "SDC (camera target)", "zh": "主车"},
        {"sw": "rect", "colors": [GIF_COLORS["stopped"]],
         "en": "latched (collision/offroad/red light)", "zh": "违规锁停"},
        {"sw": "line", "colors": [GIF_COLORS["lane"]], "en": "lane (type 1-9)", "zh": "车道"},
        {"sw": "line", "colors": [GIF_COLORS["line"]], "en": "lane marking (10-19)", "zh": "标线"},
        {"sw": "line", "colors": [GIF_COLORS["edge"]], "en": "road edge (20-29)", "zh": "边线"},
        {"sw": "line", "colors": [GIF_COLORS["crosswalk"]], "en": "crosswalk (31)", "zh": "斑马线"},
        {"sw": "lights", "colors": ["#ff0000", "#ffee00", "#00cc00", "#8c8c8c"],
         "en": "traffic stop line by state R/Y/G/other", "zh": "停止线状态"},
    ]
    return ent


def _draw_swatch(d, entry, x, cy, sw, sh, small_font):
    kind, colors = entry["sw"], entry.get("colors") or []
    ink = (40, 40, 40)
    if kind == "rect":
        d.rectangle([x, cy - sh // 2, x + sw, cy + sh // 2], fill=colors[0], outline=ink, width=1)
    elif kind == "diamond":
        r = sh * 0.9
        dcx = x + sw / 2
        d.polygon([(dcx, cy - r), (dcx + r, cy), (dcx, cy + r), (dcx - r, cy)],
                  fill=colors[0], outline=ink)
    elif kind == "palette":
        step = sw / len(colors)
        for i, c in enumerate(colors):
            d.rectangle([x + i * step, cy - sh // 2, x + (i + 1) * step, cy + sh // 2],
                        fill=c, outline=ink, width=1)
    elif kind == "lights":
        step = sw / len(colors)
        for i, c in enumerate(colors):
            d.rectangle([x + i * step, cy - sh // 2, x + (i + 1) * step - 2, cy + sh // 2],
                        fill=c, outline=ink, width=1)
    elif kind == "line":
        d.line([x, cy, x + sw, cy], fill=colors[0], width=max(2, sh // 3))
    elif kind == "traildot":  # M2 walker: magenta trail line + dot
        d.line([x, cy, x + sw, cy], fill=colors[0], width=2)
        r = sh * 0.6
        dcx = x + sw * 0.75
        d.ellipse([dcx - r, cy - r, dcx + r, cy + r], fill=colors[0], outline=ink)
    elif kind == "dash":
        for dx in range(0, sw, 7):
            d.line([x + dx, cy, x + dx + 4, cy], fill=colors[0], width=2)
    elif kind == "dotring":
        r = sh * 1.1
        for a0 in range(0, 360, 90):  # dashed ring, viz draws it linestyles="--"
            d.arc([x + sw / 2 - r, cy - r, x + sw / 2 + r, cy + r], a0, a0 + 55,
                  fill=colors[0], width=2)
        d.ellipse([x + sw / 2 - 2, cy - 2, x + sw / 2 + 2, cy + 2], fill=colors[0])
    elif kind == "text":
        d.text((x + 2, cy - sh // 2), "id", fill=(0, 0, 0), font=small_font)


def draw_legend(img, entries, font_px=18, cjk_path=None):
    """Draw the legend box at (8, 8) onto ``img``. Returns (img, box_w, box_h)
    so render_legend_png can crop a standalone legend image tightly."""
    font = _load_font(font_px, cjk_path)
    small = _load_font(max(font_px - 4, 8), cjk_path)
    d = ImageDraw.Draw(img)
    pad = max(int(font_px * 0.9), 6)
    line_h = int(font_px * 1.55)
    sw_w = int(font_px * 1.7)
    sw_h = max(int(font_px * 0.8), 6)
    labels = [_label(e, cjk_path) for e in entries]
    title = "Legend" + (" 图例" if cjk_path else "")
    box_w = int(max([d.textlength(t, font=font) for t in labels + [title]])) \
        + sw_w + pad * 3
    box_h = pad * 2 + int(font_px * 1.3) + line_h * len(entries)
    d.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(255, 255, 255), outline=(110, 110, 110), width=2)
    ty = 8 + pad
    d.text((8 + pad, ty), title, fill=(0, 0, 0), font=font)
    ty += int(font_px * 1.3)
    for entry, text in zip(entries, labels):
        cy = ty + line_h // 2
        _draw_swatch(d, entry, 8 + pad, cy, sw_w, sw_h, small)
        d.text((8 + pad + sw_w + pad // 2, cy - font_px // 2 - 1), text, fill=(0, 0, 0), font=font)
        ty += line_h
    return img, box_w, box_h


def map_window(map_corners):
    """Full-map camera; reproduces the previous viz.py bounds (_get_bounds:
    center ± max_half_span*1.02) so the whole map fills the frame once."""
    c0, c1, c2, c3 = map_corners[:4]
    center = ((c0 + c2) / 2.0, (c1 + c3) / 2.0)
    return center, max(c2 - c0, c3 - c1) * 1.02


def full_map_world_to_px(map_corners):
    """world→px callable for FULL_MAP_PX renders produced via map_window."""
    center, window_m = map_window(map_corners)
    scale = FULL_MAP_PX / window_m
    return lambda x, y: _world_to_px(x, y, center, scale, FULL_MAP_PX)


def render_full_map(step, agent_f32, agent_i32, road_elements, traffic_elements,
                    sdc_row, map_corners, dt, seed, map_name, mode_label,
                    static_indices=None, static_class_of=None,
                    walker_indices=None, walker_trails=None):
    """Whole-map keyframe from the shared pillow renderer (class coloring,
    forced >=12 px cone diamonds, forced >=12 px walker dots)."""
    center, window_m = map_window(map_corners)
    return render_gif_frame(step, agent_f32, agent_i32, road_elements, traffic_elements,
                            sdc_row, center, window_m, FULL_MAP_PX, dt, seed, map_name,
                            mode_label=mode_label, static_indices=static_indices,
                            static_class_of=static_class_of, walker_indices=walker_indices,
                            walker_trails=walker_trails)


def render_legend_png(path, mode, cjk_path=None, spawned=None):
    """Standalone legend image (owner acceptance: the legend must not overlap
    the map). Measure on a probe canvas, then re-draw on a tight white crop."""
    entries = legend_entries(mode, spawned=spawned)
    probe = Image.new("RGB", (1400, 1000), "white")
    _, box_w, box_h = draw_legend(probe, entries, font_px=20, cjk_path=cjk_path)
    img = Image.new("RGB", (box_w + 20, box_h + 20), "white")
    draw_legend(img, entries, font_px=20, cjk_path=cjk_path)
    img.save(path)


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


CONE_RGB = _hex_to_rgb(HAZARD_CLASS_COLORS["cone"])


def count_color_pixels(png_path, rgb):
    arr = np.asarray(Image.open(png_path).convert("RGB"))
    return int(np.sum(np.all(arr == np.asarray(rgb, np.uint8), axis=-1)))


def audit_pixels(seed_dir, rel_paths, rgb, label):
    """Owner acceptance gate: every image that should show ``label``-colored
    elements must contain at least one such pixel (forced minimum marker sizes
    make a zero count impossible unless the marker regressed). Any zero count
    fails the run."""
    counts, failed = {}, []
    for rel in rel_paths:
        n = count_color_pixels(Path(seed_dir) / rel, rgb)
        counts[str(rel)] = int(n)
        if n == 0:
            failed.append(str(rel))
    if failed:
        raise RuntimeError(f"{label} pixel audit failed — 0 {label}-colored px in: {failed}")
    return counts


def audit_cone_pixels(seed_dir, rel_paths):
    """Owner acceptance gate: every image that should show cones must contain
    cone-orange pixels (the forced >=12 px diamonds make a zero count
    impossible unless the marker regressed). Any zero count fails the run."""
    return audit_pixels(seed_dir, rel_paths, CONE_RGB, "cone")


def hazards_in_window(placed, center, window_m):
    """Hazards whose center falls inside the square world window; zoom callouts
    only tag these, so off-window hazards don't smear text at the frame edge."""
    half = window_m / 2.0
    x0, x1 = center[0] - half, center[0] + half
    y0, y1 = center[1] - half, center[1] + half
    return [e for e in placed if x0 <= e["x"] <= x1 and y0 <= e["y"] <= y1]


def add_hazard_callouts(pil, placed, world_to_px, cjk_path=None, min_y=0):
    """English text tags ("parked"/"cluster"/"cones", + CJK when a font was
    detected) with leader arrows on every hazard; hazard side only.
    ``world_to_px`` maps world coords into the given image (full map or zoom)."""
    d = ImageDraw.Draw(pil)
    w, h = pil.size
    font = _load_font(18, cjk_path)
    groups = []  # (class, world_x, world_y): each parked car, cluster/zone centers
    for e in placed:
        if e["kind"] == "parked":
            groups.append(("parked", e["x"], e["y"]))
    centers = {}
    for e in placed:
        key = e.get("cluster_id") or e.get("zone_id")
        if key:
            centers.setdefault((key, e["kind"]), []).append((e["x"], e["y"]))
    for (_key, kind), pts in centers.items():
        groups.append((kind, float(np.mean([p[0] for p in pts])),
                       float(np.mean([p[1] for p in pts]))))
    for kind, wx, wy in groups:
        cx, cy = world_to_px(wx, wy)
        tx = min(cx + 26, w - 120)
        ty = max(min(cy - 26, h - 40), min_y + 8)
        d.line([(tx, ty), (cx, cy)], fill=(0, 0, 0), width=2)
        vx, vy = cx - tx, cy - ty
        norm = max(float(np.hypot(vx, vy)), 1e-6)
        ux, uy = vx / norm, vy / norm
        px_, py_ = -uy, ux
        d.polygon([(cx, cy),
                   (cx - ux * 12 + px_ * 5, cy - uy * 12 + py_ * 5),
                   (cx - ux * 12 - px_ * 5, cy - uy * 12 - py_ * 5)], fill=(0, 0, 0))
        en, zh = HAZARD_CLASS_LABELS[kind]
        text = en + (f" {zh}" if cjk_path else "")
        box = d.textbbox((tx + 4, ty - 9), text, font=font)
        d.rectangle(box, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        d.text((tx + 4, ty - 9), text, fill=(0, 0, 0), font=font)
    return pil


def _ensure_image(src):
    """Callers pass either a path or an already-rendered PIL frame."""
    return src if isinstance(src, Image.Image) else Image.open(src)


def side_by_side(left_png, right_png, out_path, labels=("baseline", "static-hazards")):
    a, b = _ensure_image(left_png), _ensure_image(right_png)
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

    t_start = time.time()
    kernel_mode = args.placement == "kernel"
    mode_label = "kernel-static" if kernel_mode else "static-hazards"
    print(f"[hazard-demo] mode={mode_label} (placement={args.placement}) seed={args.seed} map={map_bin}")
    cjk_font = detect_cjk_font()
    print(f"[hazard-demo] CJK font: {cjk_font or 'none -> English-only labels (no tofu)'}")

    # ---- episode 1: hazards --------------------------------------------------
    if kernel_mode:
        hazard_cfg = load_hazard_config(args.hazard_config)
        hazard_cfg = json.loads(json.dumps(hazard_cfg))  # deep copy; flip only this run's face
        # kernel-static = K1 static classes only (K2 contract: the walker tail slots and
        # the c_step unicycle advance belong to --mode jaywalker --placement kernel;
        # keeping jaywalker off here preserves the K1 RNG stream bit-for-bit)
        hazard_cfg["jaywalker"]["enable"] = False
        slots = kernel_slot_capacity(hazard_cfg)
        env = build_env(args, map_bin, cosim_partner_slots=slots, hazard_cfg=hazard_cfg)
    else:
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
    edges = EdgeIndex(road_elements)
    if kernel_mode:
        # K1: the C reset path already placed hazards; the demo only reads back,
        # classifies and audits. No set_agent_states writes, no python placement.
        placed = hazards_from_kernel_readback(env, buffers, args.agents_per_env)
        plan = {
            "parked": hazard_cfg["parked"]["n_range"][1] if hazard_cfg["parked"]["enable"] else 0,
            "clusters": hazard_cfg["clusters"]["n_range"][1] if hazard_cfg["clusters"]["enable"] else 0,
            "cone_zones": hazard_cfg["cones"]["zones_range"][1] if hazard_cfg["cones"]["enable"] else 0,
        }
        parked_n = sum(1 for e in placed if e["kind"] == "parked")
        clusters_n = sum(1 for e in placed if e["kind"] == "collided") // 2
        zones_n = len({e.get("zone_id") for e in placed if e["kind"] == "cone"})
        cones_n = sum(1 for e in placed if e["kind"] == "cone")
        deficit = (max(0, plan["parked"] - parked_n) + max(0, plan["clusters"] - clusters_n)
                   + max(0, plan["cone_zones"] - zones_n))
        rejects = {"planned_deficit": deficit}  # kernel rejection sampling, aggregated
        idxs = np.array([e["idx"] for e in placed], np.int32)
        legality = verify_static_hazards(env, placed, idxs,
                                         buffers["agent_f32"][0], buffers["agent_i32"][0])
        legality["offroad_violations_postcheck"] = int(kernel_edge_audit(placed, edges))
        print(f"[hazard-demo] kernel placement (C reset): parked={parked_n} clusters={clusters_n} "
              f"cone_zones={zones_n} ({cones_n} cones), slots={len(placed)}/{slots}")
        print(f"[hazard-demo] legality: {legality}")
    else:
        lanes = LaneIndex(road_elements)
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
    if len(placed) > slots:
        raise RuntimeError(f"hazard slots exhausted: {len(placed)} > {slots}")
    if not placed:
        raise RuntimeError("no hazard could be placed on this map/seed")

    static_indices = [e["idx"] for e in placed]
    static_class_of = {e["idx"]: e["kind"] for e in placed}
    zone_centers = {}
    for e in placed:
        key = e.get("cluster_id") or e.get("zone_id")
        if key:
            zone_centers.setdefault(key, []).append((e["x"], e["y"]))

    keyframe_paths = []
    zoom_snaps = {}  # COMPARE_KEYFRAMES step -> (agent_f32, agent_i32) copies for true-scale zooms
    episode_end = None
    gif_frames = []
    collision_curve = []  # cumulative latched-collision agent count per step

    def render_keyframes(step):
        pil = render_full_map(step, af0, ai0, road_elements, traffic_elements, sdc_row,
                              map_corners, args.dt, args.seed, map_bin.stem,
                              mode_label, static_indices=static_indices,
                              static_class_of=static_class_of)
        path = hazard_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        keyframe_paths.append(str(path.relative_to(seed_dir)))
        if step in COMPARE_KEYFRAMES:  # snapshots for true-scale zoom re-renders
            zoom_snaps[step] = (af0.copy(), ai0.copy())

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
                         map_bin.stem, mode_label=mode_label,
                         static_indices=static_indices, static_class_of=static_class_of)
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
    baseline_snaps = {}  # COMPARE_KEYFRAMES step -> buffer copies for zoom re-renders
    baseline_end = None
    baseline_curve = []
    sdc_row_b = int((scenarios_b[0].get("active_agent_indices") or [0])[0])

    def capture_b():
        env_b.get_obs_html_frame(
            buffers_b["agent_f32"], buffers_b["agent_i32"], buffers_b["metrics_f32"],
            buffers_b["puffer_f32"], buffers_b["traffic_i16"], buffers_b["goals_f32"],
            buffers_b["rewards_f32"], buffers_b["coefs_f32"],
        )

    capture_b()
    abf, abi, amf = buffers_b["agent_f32"][0], buffers_b["agent_i32"][0], buffers_b["metrics_f32"][0]

    def render_keyframes_b(step):
        pil = render_full_map(step, abf, abi, road_elements, traffic_elements,
                              sdc_row_b, map_corners, args.dt, args.seed, map_bin.stem,
                              "baseline")
        path = baseline_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        baseline_keyframes.append(str(path.relative_to(seed_dir)))
        baseline_snaps[step] = (abf.copy(), abi.copy())

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
            # right panel: class-colored full map + text callouts (hazard side only)
            h_annot = add_hazard_callouts(Image.open(h_png), placed,
                                          full_map_world_to_px(map_corners), cjk_font)
            side_by_side(b_png, h_annot, compare_dir / f"compare_t{t:03d}.png")
        else:
            missing.append(t)
    zooms = []
    zoom_cone_paths = []  # zooms of cone zones: subject to the cone pixel audit
    cone_zone_keys = {e.get("zone_id") for e in placed if e["kind"] == "cone"}
    for key, pts in zone_centers.items():
        ctr = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        zoom_w2p = lambda x, y, ctr=ctr: _world_to_px(
            x, y, ctr, ZOOM_PX / ZOOM_WINDOW_M, ZOOM_PX)
        for t in (0, 40):
            if t not in zoom_snaps or t not in baseline_snaps:
                continue
            # true-scale zooms re-rendered from snapshots: cones at real size
            # (min_cone_px=0), not the forced full-map diamond
            b_img = render_gif_frame(t, baseline_snaps[t][0], baseline_snaps[t][1],
                                     road_elements, traffic_elements, sdc_row_b,
                                     ctr, ZOOM_WINDOW_M, ZOOM_PX, args.dt, args.seed,
                                     map_bin.stem, mode_label="baseline", min_cone_px=0)
            h_img = render_gif_frame(t, zoom_snaps[t][0], zoom_snaps[t][1],
                                     road_elements, traffic_elements, sdc_row,
                                     ctr, ZOOM_WINDOW_M, ZOOM_PX, args.dt, args.seed,
                                     map_bin.stem, mode_label=mode_label,
                                     static_indices=static_indices,
                                     static_class_of=static_class_of, min_cone_px=0)
            h_img = add_hazard_callouts(h_img, hazards_in_window(placed, ctr, ZOOM_WINDOW_M),
                                        zoom_w2p, cjk_font)
            zoom_name = f"zoom_{key}_t{t:03d}.png"
            side_by_side(b_img, h_img, compare_dir / zoom_name)
            zooms.append(f"compare/{zoom_name}")
            if key in cone_zone_keys:
                zoom_cone_paths.append(f"compare/{zoom_name}")

    # pixel-existence audit: every image that should show cones must contain
    # cone-orange pixels (hazard keyframes + compares + cone-zone zooms; the
    # other zooms target collided clusters and legitimately have no cones);
    # any 0 -> fail
    audit_paths = (list(keyframe_paths)
                   + [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                      if t not in missing]
                   + zoom_cone_paths)
    cone_audit = audit_cone_pixels(seed_dir, audit_paths)
    print(f"[hazard-demo] cone pixel audit: {len(cone_audit)} images, "
          f"min count {min(cone_audit.values()) if cone_audit else 0}")

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
        "engine": "kernel C reset (place_static_hazards)" if kernel_mode
        else "python prototype (set_agent_states)",
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
        "K1 内核放置：静态 hazard 由 C reset 路径直接写入静态槽（drive.h place_static_hazards，"
        "spawn 动态车之后、compute_observations 之前）；demo 只读回放分类，不写任何槽位。",
        "旋钮 = env kwargs hazard_*（drive.py 校验 + binding fail-fast，init 前生效）；本 run 语义"
        "映射自 config/hazard_demo.yaml；slots = 规划数上界预算（parked_hi + 2*cluster_hi + "
        "zones_hi*max(taper_cones, block_rows*block_cols)）。",
        "合法性复用内核 spawn 语义：check_obb_collision（margin 0.05m，簇内豁免——簇内交叠有意）"
        "+ check_spawn_offroad + refresh_lane_association；RNG = episode rng_state（与动态撒车"
        "同流，use_exact_episode_seed 语义下同 seed 全布局可复现）。",
        "与 python 原型的已知口径差：内核 lane 锚定只认 types 1-2 drivable（原型 LaneIndex 收 1-4）；"
        "冲突检查走内核 OBB/SAT 而非 python numpy SAT；rng 流不同 → 具体位置不逐位对应，只对照"
        "合法性不变量（见 equivalence 段）。",
        "hazard 关 = 零 rng 消耗、零放置，episode 与基线逐位一致（C 测试锁定）；锥桶 block 布局相邻"
        "锥允许贴接（原型对簇内锥不做 intra-zone 检查，同口径）。",
        "图例/配色/像素审计标准同 M1：单独 legend.png、类别着色、全图锥桶 ≥12px 菱形、"
        "render_audit.cone_pixel_counts 任一为 0 即失败；compare/zoom hazard 侧带文字标注。",
    ] if kernel_mode else [
        "静态 hazard 走 cosim_partner_slots 静态槽 + set_agent_states/set_agent_sizes 放置，零内核改动。",
        "binding 限制：vec_set_agent_states/sizes 只写 vec->envs[0]（cosim 单 env 假设）→ 本模式固定 --envs 1。",
        "同 seed 保证相同地图与初始撒车布局；hazard 引入后微观轨迹自然分叉（碰撞/绕行）。",
        "图例：单独成图 legend.png（index.html 首项），逐项对应脚本渲染器；渲染帧内无图例框，"
        "仅右上角一行 legend: see legend.png 小字。全图/跟车 GIF 统一类别着色"
        "（SDC=红、车=蓝、锁停=灰），不再按 agent id 彩虹取色。",
        "hazard 三类配色：深蓝 parked、暗红碰撞簇（重叠有意）、橙色锥桶（全图强制 ≥12px 菱形，"
        "zoom 按真实尺度）；compare/zoom 的 hazard 侧另有文字+箭头指认，baseline 侧无标注；"
        "render_audit.cone_pixel_counts 逐图审计锥桶像素，任何一张为 0 即运行失败。",
        "碰撞簇内部重叠是有意的；簇整体与其他 agent/edge 无重叠（OBB SAT + 内核 lane 关联双重校验）。",
        "lane 宽度 map bin 未暴露，横向偏移按 carla 默认 3.5m 假设；内核 current_lane_idx != -1 为在道证据。",
        "kernel_lane_assoc_failures 口径：贴边锥桶（box 跨 road edge 或超 3×width=1.5m 关联半径）在内核"
        "侧 current_lane_idx=-1；已核实无 gameplay 影响——collision_check 全量遍历含静态槽、"
        "partner obs 的 partner_scan 含全部静态槽（均无 lane 门控，drive.h），offroad 违规只在"
        "active agent 的 compute_metrics 里记（静态槽不走）——仅是在道证据缺失，如实记录不修正。",
    ]
    equivalence = None
    if kernel_mode:
        py_summary = load_python_seed_summary(args.seed)
        py_legality = (py_summary or {}).get("legality") or {}
        equivalence = {
            "rng_note": "内核放置消耗 episode rng_state（与动态撒车同流），python 原型用独立 "
                        "RandomState —— 两者布局无逐位对应，仅对照合法性不变量。",
            "python_source": ("artifacts/static-hazards/seed{}/summary.json".format(args.seed)
                              if py_summary else None),
            "invariants": {
                "dynamic_overlap_violations": {"python": py_legality.get("dynamic_overlap_violations_postcheck"),
                                                "kernel": legality["dynamic_overlap_violations_postcheck"]},
                "kernel_lane_assoc_failures": {"python": py_legality.get("kernel_lane_assoc_failures"),
                                                "kernel": legality["kernel_lane_assoc_failures"]},
                "intentional_intra_cluster_overlaps": {
                    "python": py_legality.get("intentional_intra_cluster_overlaps"),
                    "kernel": legality["intentional_intra_cluster_overlaps"]},
                "offroad_violations": {"python": "placement-time rejection (M1 contract)",
                                        "kernel": legality["offroad_violations_postcheck"]},
            },
            "all_non_intentional_violations_zero": bool(
                legality["dynamic_overlap_violations_postcheck"] == 0
                and legality["offroad_violations_postcheck"] == 0
                and legality["kernel_lane_assoc_failures"] == 0),
        }
    seed_summary = {
        "mode": mode_label,
        "placement_engine": args.placement,
        "kernel": kernel_provenance(),
        "seed": args.seed,
        "map": {"name": map_bin.stem, "path": str(map_bin)},
        "agents_per_env": args.agents_per_env,
        "steps_requested": args.steps,
        "dt": args.dt,
        "placement": placement,
        "legality": legality,
        "equivalence": equivalence,
        "simulation": sim,
        "render_audit": {
            "cone_color_rgb": list(CONE_RGB),
            "cone_min_diam_px_full_map": CONE_MIN_DIAM_PX,
            "zoom_cone_scale": "true size (min_cone_px=0)",
            "cone_pixel_counts": cone_audit,
            "images_checked": len(cone_audit),
            "all_nonzero": True,
        },
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
    if kernel_mode:
        eq_root = {
            "rng_note": equivalence["rng_note"],
            "contract": "kernel 与 python 的非 intentional 违例均 = 0；具体位置可不同（RNG 流不同）",
            "seeds": {},
        }
        for d in sorted(root.glob("seed*/summary.json")):
            try:
                s = json.loads(d.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if s.get("equivalence"):
                eq_root["seeds"][str(s["seed"])] = s["equivalence"]
        (root / "equivalence.json").write_text(
            json.dumps(eq_root, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    seed_summaries = []
    for d in sorted(root.glob("seed*/summary.json")):
        try:
            seed_summaries.append(json.loads(d.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    write_root_artifacts(root, seed_summaries, cjk_path=cjk_font, mode_label=mode_label)
    print(f"[hazard-demo] artifacts -> {root} ({seed_summary['runtime_s']}s); "
          f"collision rate change vs baseline @t={window}: {sim['collision_rate_change_pct']}%")


def write_root_artifacts(root, seed_summaries, cjk_path=None, mode_label="static-hazards"):
    """Aggregate legend.png / counts_histogram.png / summary.json / index.html over seed dirs."""
    render_legend_png(root / "legend.png", "static-hazards", cjk_path=cjk_path)
    agg = {
        "mode": mode_label,
        "kernel": kernel_provenance(),
        "seeds": [s["seed"] for s in seed_summaries],
        "per_seed": [
            {"seed": s["seed"], "placement": s["placement"], "legality": s["legality"],
             "simulation": s["simulation"],
             "render_audit": {"images_checked": s.get("render_audit", {}).get("images_checked", 0),
                              "all_nonzero": s.get("render_audit", {}).get("all_nonzero", False)}}
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
  <h2>图例（legend.png 单独成图，渲染帧内无图例框）</h2>
  <img src="legend.png" width="460"/>
  <img src="counts_histogram.png" width="560"/>
  <h2>图例（与渲染逐项对应）</h2>
  <ul>
    <li>本页首项 legend.png 为完整图例，逐项对应脚本渲染器；所有渲染帧内不再画图例框，
        仅右上角一行小字 “legend: see legend.png”。</li>
    <li>全图 PNG 与跟车 GIF 统一类别着色（不按 agent id 彩虹取色）：SDC=红、普通车=蓝、违规锁停=灰、
        <span style="color:#0D1F8C"><b>■ parked（深蓝）</b></span> ·
        <span style="color:#8B0000"><b>■ 碰撞簇 cluster（暗红，重叠有意）</b></span> ·
        <span style="color:#FF8C00"><b>◆ 锥桶 cones（橙菱形，全图强制 ≥12px，zoom 按真实尺度）</b></span>；
        车道浅灰、标线灰、边线深灰、斑马线浅绿、停止线按红/黄/绿/灰状态。</li>
    <li>行人/骑行者本轮不 spawn（gigaflow 仅 type-1 车辆），图例不列条目（M2 启用时再补）。</li>
    <li>像素审计：summary.json 的 render_audit.cone_pixel_counts 逐图统计锥桶色像素，
        任何一张为 0 即运行失败；compare/zoom 的 hazard 侧带文字+箭头标注，baseline 侧无标注。</li>
  </ul>
  <h2>放置计数 / 合法性 / 碰撞率变化</h2>
  <table><tr><th>seed</th><th>parked</th><th>collided</th><th>cone zones</th><th>rejects</th><th>legality</th><th>collision agents (baseline→hazard)</th></tr>
  {rows}</table>
  {compares_html}
  <h2>口径</h2><ul>{''.join(f'<li>{n}</li>' for n in agg['notes'])}</ul>
</body></html>"""
    (root / "index.html").write_text(html, encoding="utf-8")


# --- M2 jaywalker: python-stepped crossing + reaction metrics (ticket 07) ----
#
# Zero-kernel-change behavior hazards. Walkers are 0.5 m co-sim static slots
# whose pose is OVERWRITTEN via set_agent_states every step BEFORE env.step(),
# so vehicle decisions use last-step observations while collision/TTC loops see
# the fresh pose; c_step never moves static slots. Known deviations (contract:
# must stay in summary notes): the kernel agent type stays 1 (VEHICLE) —
# observations and collisions treat the walker as a small car — and the
# setters only reach vec->envs[0], so this mode pins --envs 1.
WALKER_SIZE_M = 0.5
WALKER_RGB = (255, 0, 255)              # GIF_COLORS["walker"] = #FF00FF
WALKER_MIN_DIAM_PX = 12                 # sub-meter hazard marker rule (full maps)
WALKER_SPEED_RANGE = (1.2, 1.6)         # m/s, drawn per walker from the seed rng
WALKER_START_STAGGER_STEPS = 12
WALKER_REACTION_RADIUS_M = 15.0         # "obvious slowdown" scan radius
WALKER_CROSSWALK_CLEAR_M = 12.0         # mid-block criterion (no junction elems in bin)
WALKER_EDGE_SEARCH_M = 25.0
WALKER_MIN_EDGE_M = 1.0                 # corridor sanity: road narrow/wide bounds
WALKER_MAX_EDGE_M = 15.0
WALKER_START_OUT_M = 1.2                # start this far OUTSIDE the near road edge
WALKER_START_MAX_OUT_M = 5.0
WALKER_SPACING_M = 15.0
SLOWDOWN_REF_FLOOR_MPS = 2.0
SLOWDOWN_DROP_FRAC = 0.5
SLOWDOWN_SUSTAIN_STEPS = 3
WALKER_HIT_DIST_M = 3.0
CUT_IN_SIZE = VEHICLE_PRESETS["car"]
CUT_IN_DLATERAL_PER_STEP = 0.3          # m per step squeeze-in (ticket spec)
CUT_IN_MAX_STRAIGHT_BEND_RAD = 0.08     # per-segment heading change bound (target lane)
CUT_IN_PAIR_DH_MAX_RAD = 0.06           # lane-pair heading alignment bound
CUT_IN_OFFSET_RANGE_M = (2.6, 5.5)      # acceptable adjacent-lane centerline distance
CUT_IN_CRUISE_RANGE = (4.0, 9.0)        # scripted cruise speed bounds (m/s)
CUT_IN_TAIL_STEPS = 6                   # reaction window extends past the merge
CUT_IN_CONTACT_DIST_M = 6.0
CUT_IN_MAX_TRACKING_ERR_M = 0.25
CUT_IN_ALTERNATIVE = ("kernel per-agent IDM parameter variation (per-agent aggressiveness / "
                      "desired speed) — kernel change, out of M2 zero-kernel scope")
WALKER_REJECT_REASONS = ("no_lane", "near_crosswalk", "edge_not_found", "edge_weird",
                         "dynamic_overlap", "static_overlap", "walker_spacing", "max_attempts")


def walker_rng(seed: int) -> np.random.RandomState:
    """Placement/kinematics RNG, independent of the episode RNG stream."""
    return np.random.RandomState((seed * 104729 + 7) % (2 ** 31))


def crosswalk_segments(road_elements):
    """Crosswalk (road type 31) polylines as a (N,2,2) segment array."""
    segs = []
    for el in road_elements:
        if int(el.get("type", 0)) != 31:
            continue
        xs, ys = el.get("x"), el.get("y")
        if not xs or len(xs) < 2:
            continue
        pts = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
        segs.extend(zip(pts[:-1], pts[1:]))
    return np.array(segs) if segs else np.zeros((0, 2, 2))


def segs_min_dist(p, segs):
    """Min distance from point p to a (N,2,2) segment array."""
    p = np.asarray(p, float)
    best = np.inf
    for a, b in segs:
        d2 = _point_segment_distSq(p, np.asarray(a, float), np.asarray(b, float))
        if d2 < best:
            best = d2
    return float(np.sqrt(best)) if np.isfinite(best) else np.inf


def ray_first_hit(p, d, segs, max_m):
    """Smallest t>0 with p + t*d on any segment (0<=u<=1); None within max_m.
    Used to measure kerb-to-kerb road width along the crossing direction."""
    if len(segs) == 0:
        return None
    a = segs[:, 0, :]
    v = segs[:, 1, :] - segs[:, 0, :]
    rhs = a - np.asarray(p, float)
    det = v[:, 0] * d[1] - v[:, 1] * d[0]
    ok = np.abs(det) > 1e-9
    safe = np.where(ok, det, 1.0)
    t = (v[:, 0] * rhs[:, 1] - v[:, 1] * rhs[:, 0]) / safe
    u = (rhs[:, 0] * d[1] - rhs[:, 1] * d[0]) / safe
    valid = ok & (t > 0.05) & (u >= 0.0) & (u <= 1.0) & (t <= max_m)
    return float(t[valid].min()) if np.any(valid) else None


def _nearest_on_polyline(pts, p):
    """(dist, point) nearest point on polyline (n,2) to p."""
    p = np.asarray(p, float)
    best_d2, best_p = np.inf, None
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d2 = float((p - proj) @ (p - proj))
        if d2 < best_d2:
            best_d2, best_p = d2, proj
    return float(np.sqrt(best_d2)), best_p


def plan_walkers(rng, lanes, edges, cw_segs, dyn, count, static_placed=None):
    """Mid-block crossing spots: straight lane, >=12 m from any crosswalk (the
    bin has no junction-type elements, so crosswalk distance is the
    intersection-avoidance proxy), road edges ray-cast on both sides, start
    OUTSIDE the near kerb (jaywalking is a violation by design — no off-road
    check), but start/end poses must not overlap any dynamic vehicle (kept)
    nor any already-placed static hazard (``static_placed``, M3 mix only —
    None keeps the frozen M2 behavior/rng stream)."""
    placed, rejects = [], {r: 0 for r in WALKER_REJECT_REASONS}
    for k in range(count):
        entry = None
        for _attempt in range(MAX_PLACE_ATTEMPTS):
            lane = lanes.pick(rng, min_len_m=30.0, straight_only=True)
            if lane is None:
                rejects["no_lane"] += 1
                break
            s = rng.uniform(8.0, lane["total"] - 8.0)
            x, y, z, h = lanes.pose_at(lane, s, 0.0)
            if segs_min_dist((x, y), cw_segs) < WALKER_CROSSWALK_CLEAR_M:
                rejects["near_crosswalk"] += 1
                continue
            perp = np.array([-np.sin(h), np.cos(h)])
            d_pos = ray_first_hit((x, y), perp, edges.segs, WALKER_EDGE_SEARCH_M)
            d_neg = ray_first_hit((x, y), -perp, edges.segs, WALKER_EDGE_SEARCH_M)
            if d_pos is None or d_neg is None:
                rejects["edge_not_found"] += 1
                continue
            if not (WALKER_MIN_EDGE_M <= min(d_pos, d_neg)
                    and max(d_pos, d_neg) <= WALKER_MAX_EDGE_M):
                rejects["edge_weird"] += 1
                continue
            side = 1.0 if rng.uniform() < 0.5 else -1.0
            d_start, d_end = (d_pos, d_neg) if side > 0 else (d_neg, d_pos)
            out0 = min(d_start + WALKER_START_OUT_M, WALKER_START_MAX_OUT_M)
            out1 = min(d_end + WALKER_START_OUT_M, WALKER_START_MAX_OUT_M)
            start = np.array([x, y]) + side * perp * out0
            end = np.array([x, y]) - side * perp * out1
            wp = (float(start[0]), float(start[1]), h, WALKER_SIZE_M, WALKER_SIZE_M)
            ep = (float(end[0]), float(end[1]), h, WALKER_SIZE_M, WALKER_SIZE_M)
            if any(obb_overlap(wp, d, margin=0.05) or obb_overlap(ep, d, margin=0.05)
                   for d in dyn):
                rejects["dynamic_overlap"] += 1
                continue
            if static_placed and any(
                    obb_overlap(wp, (o["x"], o["y"], o["heading"], o["length"], o["width"]),
                                margin=0.05)
                    or obb_overlap(ep, (o["x"], o["y"], o["heading"], o["length"], o["width"]),
                                   margin=0.05)
                    for o in static_placed):
                rejects["static_overlap"] += 1
                continue
            if any(np.linalg.norm(start - np.asarray(o["start"])) < WALKER_SPACING_M
                   for o in placed):
                rejects["walker_spacing"] += 1
                continue
            entry = {"k": k, "kind": "walker", "lane_id": lane["id"], "s": float(s),
                     "x": float(start[0]), "y": float(start[1]), "z": float(z),
                     "heading": float(np.arctan2(-side * perp[1], -side * perp[0])),
                     "length": WALKER_SIZE_M, "width": WALKER_SIZE_M,
                     "start": (float(start[0]), float(start[1])),
                     "end": (float(end[0]), float(end[1])),
                     "dir": (float(-side * perp[0]), float(-side * perp[1])),
                     "cross_pt": (float(x), float(y)),
                     "crossing_width_m": float(np.linalg.norm(end - start)),
                     "edge_out_m": [float(out0), float(out1)]}
            break
        if entry is None:
            rejects["max_attempts"] += 1
            continue
        placed.append(entry)
    return placed, rejects


def plan_cut_in(rng, lanes, dyn, walkers, cruise):
    """Pick a straight near-parallel lane pair for the scripted merge (cut-in
    maker). The vehicle starts on the adjacent lane, drives forward at
    ``cruise`` and squeezes toward the target lane at 0.3 m lateral per step.
    Returns None when no usable pair / collision-free start exists."""
    cand = []
    for ln in lanes.lanes:
        if ln["total"] < 40.0:
            continue
        dh = wrapped_angle_diff(ln["headings"][1:], ln["headings"][:-1])
        if len(dh) and float(np.max(np.abs(dh))) > CUT_IN_MAX_STRAIGHT_BEND_RAD:
            continue
        cand.append(ln)
    if not cand:
        return None
    mids = np.array([ln["pts"].mean(axis=0) for ln in cand])
    mids_h = np.array([ln["headings"][len(ln["headings"]) // 2] for ln in cand])
    for ci in rng.permutation(len(cand)):
        lane = cand[int(ci)]
        for frac in (0.3, 0.5, 0.7):
            s0 = lane["total"] * frac
            x, y, z, h = lanes.pose_at(lane, s0, 0.0)
            p = np.array([x, y])
            perp = np.array([-np.sin(h), np.cos(h)])
            for jj in range(len(cand)):
                if jj == int(ci):
                    continue
                other = cand[jj]
                if abs(float(wrapped_angle_diff(mids_h[jj], h))) > CUT_IN_PAIR_DH_MAX_RAD:
                    continue
                if float(np.linalg.norm(mids[jj] - p)) > 40.0:
                    continue
                d0, proj = _nearest_on_polyline(other["pts"], p)
                if proj is None or not (CUT_IN_OFFSET_RANGE_M[0] <= d0 <= CUT_IN_OFFSET_RANGE_M[1]):
                    continue
                side = float(np.sign((proj - p) @ perp))
                if side == 0.0:
                    continue
                start = p + perp * side * d0
                if any(np.linalg.norm(start - np.asarray(w["start"])) < 10.0 for w in walkers):
                    continue
                spose = (float(start[0]), float(start[1]), h, CUT_IN_SIZE[0], CUT_IN_SIZE[1])
                if any(obb_overlap(spose, d, margin=0.05) for d in dyn):
                    continue
                return {"kind": "cut_in", "lane_id": lane["id"],
                        "adjacent_lane_id": other["id"], "lateral_offset_m": float(d0),
                        "side": side, "start": (float(start[0]), float(start[1])),
                        "x": float(start[0]), "y": float(start[1]), "z": float(z),
                        "heading": float(h),
                        "dir": (float(np.cos(h)), float(np.sin(h))),
                        "perp": (float(perp[0]), float(perp[1])),
                        "length": CUT_IN_SIZE[0], "width": CUT_IN_SIZE[1],
                        "cruise_mps": cruise, "s0": float(s0)}
    return None


def cut_in_pose(cut, t, dt):
    """Scripted merge pose at loop step t (t=0 is the placed start pose):
    forward at cruise along the target-lane heading while the lateral offset
    toward the target lane shrinks by 0.3 m per step (both lanes straight and
    near-parallel, so the parametric form stays on-lane)."""
    lat = cut["side"] * max(cut["lateral_offset_m"] - CUT_IN_DLATERAL_PER_STEP * t, 0.0)
    adv = cut["cruise_mps"] * dt * t
    return (cut["start"][0] + cut["dir"][0] * adv + cut["perp"][0] * lat,
            cut["start"][1] + cut["dir"][1] * adv + cut["perp"][1] * lat,
            cut["z"], cut["heading"])


def slowdown_onsets(speeds, t0, t1):
    """'Obvious slowdown' events in captures [t0, t1]: ref = speed at t0
    (>= 2 m/s floor); an event is >=3 consecutive captures at <= 0.5*ref,
    counted at onsets only so one sustained deceleration counts once."""
    t1 = min(t1, len(speeds) - 1)
    if t1 < t0 or t1 - t0 + 1 < SLOWDOWN_SUSTAIN_STEPS:
        return 0
    ref = float(speeds[t0])
    if not np.isfinite(ref) or ref < SLOWDOWN_REF_FLOOR_MPS:
        return 0
    below = speeds[t0:t1 + 1] <= SLOWDOWN_DROP_FRAC * ref
    n = 0
    for i in range(len(below)):
        if below[i] and (i == 0 or not below[i - 1]) \
                and bool(np.all(below[i:i + SLOWDOWN_SUSTAIN_STEPS])):
            n += 1
    return n


def _window_rows(pos_series, vx, vy, valid, t0, t1, radius):
    """Vehicle rows that come within ``radius`` of the scripted trajectory at
    any capture in [t0, t1]."""
    rows = set()
    t1 = min(t1, len(pos_series) - 1, len(vx) - 1)
    for t in range(t0, t1 + 1):
        wp = pos_series[t]
        d = np.hypot(vx[t] - wp[0], vy[t] - wp[1])
        rows.update(int(r) for r in np.flatnonzero(valid[t] & (d <= radius)))
    return rows


def _window_min_dist(pos_series, vx, vy, valid, t0, t1):
    """(min center distance, (capture, row)) between the scripted trajectory
    and live dynamic vehicles over captures [t0, t1]; (None, None) if none."""
    best, arg = np.inf, None
    t1 = min(t1, len(pos_series) - 1, len(vx) - 1)
    for t in range(t0, t1 + 1):
        wp = pos_series[t]
        d = np.hypot(vx[t] - wp[0], vy[t] - wp[1])
        d[~valid[t]] = np.inf
        j = int(np.argmin(d))
        if d[j] < best:
            best, arg = float(d[j]), (t, j)
    return (best if np.isfinite(best) else None), arg


def contact_rows(pos_series, vx, vy, coll, valid, t0, t1, thresh_m):
    """Rows whose collision metric first latched within the window while
    within thresh_m of the scripted actor (walker-hit / cut-in contact)."""
    rows = set()
    t1 = min(t1, len(pos_series) - 1, len(coll) - 1)
    for t in range(max(t0, 1), t1 + 1):
        onset = coll[t] & ~coll[t - 1] & valid[t]
        wp = pos_series[t]
        for r in np.flatnonzero(onset):
            if np.hypot(vx[t, r] - wp[0], vy[t, r] - wp[1]) <= thresh_m:
                rows.add(int(r))
    return sorted(rows)


def add_walker_callouts(pil, targets, world_to_px, cjk_path=None):
    """Arrow + boxed-text tags on the hazard side of compares/zooms (baseline
    side stays untagged). targets: (en, zh, world_x, world_y, hex_color)."""
    d = ImageDraw.Draw(pil)
    w, h = pil.size
    font = _load_font(18, cjk_path)
    for en, zh, wx, wy, color in targets:
        cx, cy = world_to_px(wx, wy)
        tx = min(cx + 26, w - 150)
        ty = max(min(cy - 26, h - 40), 8)
        d.line([(tx, ty), (cx, cy)], fill=(0, 0, 0), width=2)
        vx, vy = cx - tx, cy - ty
        norm = max(float(np.hypot(vx, vy)), 1e-6)
        ux, uy = vx / norm, vy / norm
        px_, py_ = -uy, ux
        d.polygon([(cx, cy), (cx - ux * 12 + px_ * 5, cy - uy * 12 + py_ * 5),
                   (cx - ux * 12 - px_ * 5, cy - uy * 12 - py_ * 5)], fill=(0, 0, 0))
        text = en + (f" {zh}" if cjk_path else "")
        box = d.textbbox((tx + 4, ty - 9), text, font=font)
        d.rectangle(box, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        d.text((tx + 4, ty - 9), text, fill=rgb, font=font)
    return pil


def add_trail_annotations(pil, walkers, trails, world_to_px, cut_in, cut_trail,
                          cjk_path=None):
    """trails.png overlay: start (hollow) / end (filled) markers per walker
    plus one stat line each (done/incomplete + distance + duration); the
    cut-in maker labeled at its final pose."""
    d = ImageDraw.Draw(pil)
    font = _load_font(20, cjk_path)
    for w in walkers:
        pts = trails[w["row"]]
        sx, sy = world_to_px(*pts[0])
        ex, ey = world_to_px(*pts[-1])
        d.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], outline=WALKER_RGB, width=3)
        d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=WALKER_RGB, outline=(0, 0, 0))
        status = "done" if w["done_step"] is not None else "incomplete"
        text = f"walker{w['k']}: {status} {w['crossing_width_m']:.1f} m"
        if w.get("crossing_duration_s") is not None:
            text += f" in {w['crossing_duration_s']:.1f} s"
        box = d.textbbox((sx + 14, sy - 34), text, font=font)
        d.rectangle(box, fill=(255, 255, 255), outline=WALKER_RGB, width=2)
        d.text((sx + 14, sy - 34), text, fill=(0, 0, 0), font=font)
    if cut_in and cut_trail:
        fx, fy = world_to_px(*cut_trail[-1])
        text = f"cut-in maker (merge {cut_in['lateral_offset_m']:.1f} m, " \
               f"{cut_in['cruise_mps']:.1f} m/s)"
        box = d.textbbox((fx + 14, fy - 34), text, font=font)
        d.rectangle(box, fill=(255, 255, 255), outline=(123, 47, 191), width=2)
        d.text((fx + 14, fy - 34), text, fill=(123, 47, 191), font=font)
    return pil


def run_jaywalker(args, map_bin):
    """M2: python-stepped jaywalker crossings + traffic reaction metrics.

    Each step: set_agent_states(new walker poses) -> env.step() — vehicle
    decisions use last-step observations while collision/TTC loops see the
    fresh pose. A same-seed clean baseline episode provides the reaction
    comparison and the compare/zoom artifacts. Co-sim setters only reach vec
    env 0 -> envs=1 (forced).

    K2 (--placement kernel): the walkers are placed AND stepped inside the C
    kernel (type=PEDESTRIAN static slots, unicycle advance in c_step); the demo
    only reads back poses per capture and the per-walker metrics via
    get_walker_metrics(). No set_agent_states writes, no python placement, no
    cut-in puppeteering."""
    kernel_mode = args.placement == "kernel"
    mode_label = "kernel-jaywalker" if kernel_mode else "jaywalker"
    if args.envs != 1:
        print("[hazard-demo] jaywalker: co-sim setters only reach vec env 0; forcing --envs 1")
        args.envs = 1
    if args.walkers is None:
        args.walkers = 2  # frozen M2 default (mix mode reads the count from config)
    if args.walkers < 1:
        raise ValueError("--walkers must be >= 1")

    root = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    seed_dir = root / f"seed{args.seed}"
    baseline_dir = seed_dir / "baseline" / "frames"
    jw_dir = seed_dir / "jaywalker" / "frames"
    compare_dir = seed_dir / "compare"
    for d in (baseline_dir, jw_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[hazard-demo] mode={mode_label} (placement={args.placement}) seed={args.seed} map={map_bin}")
    cjk_font = detect_cjk_font()
    print(f"[hazard-demo] CJK font: {cjk_font or 'none -> English-only labels (no tofu)'}")

    # ---- episode 1: jaywalker -----------------------------------------------
    if kernel_mode:
        # kernel knob face: hazard_walkers_* sourced from the jaywalker config section;
        # the per-seed walker count is pinned (M2 parity: default 2) so the vs-M2 tables
        # compare like-for-like; speed band comes from the config unchanged.
        hazard_cfg = load_hazard_config(args.hazard_config)
        hazard_cfg = json.loads(json.dumps(hazard_cfg))  # deep copy; flip only this run's face
        hazard_cfg["enable"] = True
        for cls in ("parked", "clusters", "cones"):
            hazard_cfg[cls]["enable"] = False  # kernel-jaywalker = walkers only (M2 parity)
        hazard_cfg["jaywalker"]["enable"] = True
        hazard_cfg["jaywalker"]["n_range"] = [args.walkers, args.walkers]
        slots = args.walkers
        env = build_env(args, map_bin, cosim_partner_slots=slots, hazard_cfg=hazard_cfg)
    else:
        slots = args.walkers + (0 if args.skip_cut_in else 1)
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
    af0, ai0, mf0 = buffers["agent_f32"][0], buffers["agent_i32"][0], buffers["metrics_f32"][0]
    lanes = LaneIndex(road_elements)
    edges = EdgeIndex(road_elements)
    cw_segs = crosswalk_segments(road_elements)
    dyn = dynamic_boxes(af0, ai0)
    dt = args.dt

    def kernel_walker_readback(getter):
        """K2: kernel-truth walker entries from get_walker_metrics() + type audit.
        The demo never writes a pose; lane_id/s stay None (kernel placement is
        lane-anchored internally but the anchor is not part of the getter face)."""
        rows = [int(r) for r in getter["idx"][: len(getter["idx"])] if r >= 0]
        entries = []
        for k, row in enumerate(rows):
            types = int(ai0[row, 1])
            if types != 2:
                raise RuntimeError(f"kernel walker row {row} has agent type {types} (expected 2 PEDESTRIAN)")
            entries.append({
                "k": k, "row": row,
                "x": float(af0[row, 0]), "y": float(af0[row, 1]), "z": float(af0[row, 2]),
                "heading": float(af0[row, 3]), "length": float(af0[row, 4]), "width": float(af0[row, 5]),
                "agent_type": types,
                "lane_id": None, "s": None,
                "start": (float(af0[row, 0]), float(af0[row, 1])),
                "cross_pt": (float(af0[row, 0]), float(af0[row, 1])),
                "speed_mps": float(getter["speed_mps"][k]),
                "start_step": int(getter["start_step"][k]),
                "done_step": None,  # latched post-run from the getter
                "kernel_done_step": int(getter["done_step"][k]),
                "completed": False,
                "crossing_width_m": float(getter["crossing_width_m"][k]),
                "min_dist_to_vehicle_m": (float(getter["min_dist_m"][k])
                                          if getter["min_dist_m"][k] >= 0 else None),
                "contact_events": int(getter["contact_events"][k]),
                "duration_s": (float(getter["duration_s"][k]) if getter["duration_s"][k] >= 0 else None),
                "trailing_key": "kernel",
            })
        return entries

    if kernel_mode:
        getter0 = env.get_walker_metrics()
        walkers = kernel_walker_readback(getter0)
        walker_rejects = {}
        if len(walkers) < args.walkers:
            raise RuntimeError(f"kernel placed {len(walkers)}/{args.walkers} walkers")
        for k, w in enumerate(walkers):
            w["row"] = args.agents_per_env + k
            est = int(np.ceil(w["crossing_width_m"] / (w["speed_mps"] * dt)))
            w["mid_step"] = min(w["start_step"] + max(est // 2, 1), args.steps - 1)
            w["traveled"] = 0.0
            w["tracking_max_err_m"] = None
    else:
        rng = walker_rng(args.seed)
        walkers, walker_rejects = plan_walkers(rng, lanes, edges, cw_segs, dyn, args.walkers)
        if len(walkers) < args.walkers:
            raise RuntimeError(f"only {len(walkers)}/{args.walkers} walkers placed; "
                               f"rejects={ {k: v for k, v in walker_rejects.items() if v} }")
        for k, w in enumerate(walkers):
            w["row"] = args.agents_per_env + k
            w["speed_mps"] = float(rng.uniform(*WALKER_SPEED_RANGE))
            w["start_step"] = 1 + k * WALKER_START_STAGGER_STEPS
            est = int(np.ceil(w["crossing_width_m"] / (w["speed_mps"] * dt)))
            w["mid_step"] = min(w["start_step"] + max(est // 2, 1), args.steps - 1)
            w["traveled"] = 0.0
            w["done_step"] = None
            w["tracking_max_err_m"] = 0.0
    valid0 = (ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (ai0[:, 5] == 0)
    med_v = float(np.median(af0[valid0, 6])) if valid0.any() else 6.0
    cruise = float(np.clip(med_v, *CUT_IN_CRUISE_RANGE)) if np.isfinite(med_v) else 6.0
    if kernel_mode:
        cut_in = None
    else:
        cut_in = None if args.skip_cut_in else plan_cut_in(rng, lanes, dyn, walkers, cruise)
    if cut_in:
        cut_in["row"] = args.agents_per_env + len(walkers)
        cut_in["merge_steps"] = int(np.ceil(cut_in["lateral_offset_m"] / CUT_IN_DLATERAL_PER_STEP))
        cut_in["start_step"] = 1
        cut_in["window"] = (1, min(1 + cut_in["merge_steps"] + CUT_IN_TAIL_STEPS, args.steps))
        cut_in["tracking_max_err_m"] = 0.0
    initial = walkers + ([cut_in] if cut_in else [])
    print(f"[hazard-demo] walkers={len(walkers)} "
          f"cut_in={'yes (merge %.1f m over %d steps)' % (cut_in['lateral_offset_m'], cut_in['merge_steps']) if cut_in else 'no'} "
          f"slots={len(initial)}/{slots} cruise={cruise:.1f} m/s")

    if kernel_mode:
        # K2: the C reset already placed (and will step) the walkers — the demo only
        # audits the readback: start/end poses must not overlap any dynamic vehicle.
        capture()
        post_overlap = sum(1 for e in initial
                           if any(obb_overlap((e["x"], e["y"], e["heading"], e["length"], e["width"]),
                                              d, margin=0.05) for d in dyn))
        place_err = 0.0  # kernel writes poses directly; no python->world readback round-trip
        print(f"[hazard-demo] kernel walker placement (C reset): n={len(walkers)} "
              f"types={[e['agent_type'] for e in walkers]} slots={len(walkers)}/{slots}")
        print(f"[hazard-demo] placement dynamic_overlap_postcheck={post_overlap}")
    else:
        apply_static_hazards(env, args, initial)
        capture()
        g_state = env.get_global_agent_state(include_static=True)
        place_err = max(max(abs(float(g_state["x"][e["idx"]]) - e["x"]),
                            abs(float(g_state["y"][e["idx"]]) - e["y"])) for e in initial)
        post_overlap = sum(1 for e in initial
                           if any(obb_overlap((e["x"], e["y"], e["heading"], e["length"], e["width"]),
                                              d, margin=0.05) for d in dyn))
        print(f"[hazard-demo] placement readback_err={place_err} dynamic_overlap_postcheck={post_overlap}")

    class_of = {**{w["row"]: "walker" for w in walkers},
                **({cut_in["row"]: "cut_in"} if cut_in else {})}
    static_indices = list(class_of)
    walker_rows = [w["row"] for w in walkers]
    trails = {w["row"]: [(float(af0[w["row"], 0]), float(af0[w["row"], 1]))] for w in walkers}
    cut_trail = ([(float(af0[cut_in["row"], 0]), float(af0[cut_in["row"], 1]))]
                 if cut_in else None)

    per_step = []

    def record_step():
        valid = (ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (ai0[:, 5] == 0)
        per_step.append({"x": af0[:, 0].copy(), "y": af0[:, 1].copy(),
                         "v": af0[:, 6].copy(), "valid": valid.copy(),
                         "coll": (mf0[:, COLLISION_IDX] > 0).copy()})

    record_step()
    keyframe_paths = []
    snap_jw = {}

    def render_keyframes(step):
        pil = render_full_map(step, af0, ai0, road_elements, traffic_elements, sdc_row,
                              map_corners, dt, args.seed, map_bin.stem, mode_label,
                              static_indices=static_indices, static_class_of=class_of,
                              walker_indices=walker_rows,
                              walker_trails={r: np.asarray(p) for r, p in trails.items()})
        path = jw_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        keyframe_paths.append(str(path.relative_to(seed_dir)))
        if any(step == w["mid_step"] for w in walkers):
            snap_jw[step] = (af0.copy(), ai0.copy())

    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    steps_run = 0
    episode_end = None
    gif_frames = []
    gif_center = (walkers[0]["cross_pt"][0], walkers[0]["cross_pt"][1])
    jw_keyframes = sorted(set(HAZARD_KEYFRAMES) | {w["mid_step"] for w in walkers})
    render_keyframes(0)
    gif_frames.append((0, af0.copy(), ai0.copy()))

    for step in range(1, args.steps + 1):
        written = {}
        if kernel_mode:
            pass  # K2: walkers advance inside c_step; the demo never writes poses
        else:
            idxs, xs, ys, zs, hs, vxs, vys = [], [], [], [], [], [], []
            for w in walkers:
                if w["done_step"] is not None or step < w["start_step"]:
                    continue  # standing at the kerb / already across (c_step never moves slots)
                w["traveled"] += w["speed_mps"] * dt
                if w["traveled"] >= w["crossing_width_m"]:
                    w["traveled"] = w["crossing_width_m"]
                    w["done_step"] = step
                px = w["start"][0] + w["dir"][0] * w["traveled"]
                py = w["start"][1] + w["dir"][1] * w["traveled"]
                v = w["speed_mps"] if w["done_step"] is None else 0.0
                idxs.append(w["row"])
                xs.append(px); ys.append(py); zs.append(w["z"]); hs.append(w["heading"])
                vxs.append(w["dir"][0] * v); vys.append(w["dir"][1] * v)
                written[w["row"]] = (px, py)
            if cut_in is not None:
                px, py, pz, ph = cut_in_pose(cut_in, step, dt)
                idxs.append(cut_in["row"])
                xs.append(px); ys.append(py); zs.append(pz); hs.append(ph)
                vxs.append(cut_in["dir"][0] * cut_in["cruise_mps"])
                vys.append(cut_in["dir"][1] * cut_in["cruise_mps"])
                written[cut_in["row"]] = (px, py)
            if idxs:
                env.set_agent_states(
                    np.asarray(idxs, np.int32),
                    np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                    np.asarray(zs, np.float32), np.asarray(hs, np.float32),
                    np.asarray(vxs, np.float32), np.asarray(vys, np.float32),
                    np.zeros(len(idxs), np.float32), np.zeros(len(idxs), np.float32),
                    None, np.zeros(len(idxs), np.float32),
                )
        env.step(actions)
        steps_run = step
        capture()
        record_step()
        if kernel_mode:
            # latch the kernel-side completion/metric truth into the walker entries
            getter_t = env.get_walker_metrics()
            for k, w in enumerate(walkers):
                if int(getter_t["done_step"][k]) >= 0 and w["done_step"] is None:
                    w["done_step"] = int(getter_t["done_step"][k])
                    w["completed"] = True
                    w["duration_s"] = float(getter_t["duration_s"][k])
                w["min_dist_to_vehicle_m"] = (float(getter_t["min_dist_m"][k])
                                              if getter_t["min_dist_m"][k] >= 0 else None)
                w["contact_events"] = int(getter_t["contact_events"][k])
            walkers_final_getter = getter_t
        for w in walkers:
            trails[w["row"]].append((float(af0[w["row"], 0]), float(af0[w["row"], 1])))
            if not kernel_mode and w["row"] in written:
                px, py = written[w["row"]]
                w["tracking_max_err_m"] = max(
                    w["tracking_max_err_m"],
                    max(abs(px - float(af0[w["row"], 0])), abs(py - float(af0[w["row"], 1]))))
        if cut_in is not None:
            cut_trail.append((float(af0[cut_in["row"], 0]), float(af0[cut_in["row"], 1])))
            if cut_in["row"] in written:
                px, py = written[cut_in["row"]]
                cut_in["tracking_max_err_m"] = max(
                    cut_in["tracking_max_err_m"],
                    max(abs(px - float(af0[cut_in["row"], 0])),
                        abs(py - float(af0[cut_in["row"], 1]))))
        lo, hi = env.agent_offsets[0], env.agent_offsets[1]
        if episode_end is None and bool(np.any(env.truncations[lo:hi])):
            episode_end = step
            print(f"[hazard-demo] env0 jaywalker episode ended at step {step}")
            break
        if step in jw_keyframes:
            render_keyframes(step)
        if args.gif_stride > 0 and step % args.gif_stride == 0:
            gif_frames.append((step, af0.copy(), ai0.copy()))

    for w in walkers:
        if kernel_mode and w.get("duration_s") is not None:
            w["crossing_duration_s"] = round(w["duration_s"], 2)  # kernel getter truth
        else:
            w["crossing_duration_s"] = (round((w["done_step"] - w["start_step"] + 1) * dt, 2)
                                        if w["done_step"] is not None else None)
    jw_metrics = {
        "steps_run": steps_run,
        "ended_step": episode_end if episode_end is not None else f"ran to {steps_run}",
        "collision_agents_final": int(np.sum((ai0[:, 3] == 1) & (ai0[:, 2] == 1)
                                             & (mf0[:, COLLISION_IDX] > 0))),
    }
    env.close()

    trail_arrays = {r: np.asarray(p) for r, p in trails.items()}
    gif_path = seed_dir / "jaywalker" / "jaywalker.gif"
    gif_imgs = [
        render_gif_frame(st, af, ai, road_elements, traffic_elements, sdc_row,
                         gif_center, HAZARD_GIF_WINDOW_M, 900, dt, args.seed,
                         map_bin.stem, mode_label=mode_label,
                         static_indices=static_indices, static_class_of=class_of,
                         walker_indices=walker_rows, walker_trails=trail_arrays)
        for st, af, ai in gif_frames
    ]
    if gif_imgs:
        gif_imgs[0].save(gif_path, save_all=True, append_images=gif_imgs[1:],
                         duration=int(1000 / args.gif_fps), loop=0)

    # ---- episode 2: same-seed clean baseline (comparison side) ---------------
    env_b = build_env(args, map_bin)
    env_b.reset(seed=args.seed)
    scenarios_b = env_b.get_state()
    buffers_b = capture_buffers(env_b, scenarios_b)
    sdc_row_b = int((scenarios_b[0].get("active_agent_indices") or [0])[0])

    def capture_b():
        env_b.get_obs_html_frame(
            buffers_b["agent_f32"], buffers_b["agent_i32"], buffers_b["metrics_f32"],
            buffers_b["puffer_f32"], buffers_b["traffic_i16"], buffers_b["goals_f32"],
            buffers_b["rewards_f32"], buffers_b["coefs_f32"],
        )

    capture_b()
    abf, abi, amf = buffers_b["agent_f32"][0], buffers_b["agent_i32"][0], buffers_b["metrics_f32"][0]
    per_step_b = []

    def record_step_b():
        valid = (abi[:, 3] == 1) & (abi[:, 2] == 1) & (abi[:, 5] == 0)
        per_step_b.append({"x": abf[:, 0].copy(), "y": abf[:, 1].copy(),
                           "v": abf[:, 6].copy(), "valid": valid.copy(),
                           "coll": (amf[:, COLLISION_IDX] > 0).copy()})

    record_step_b()
    baseline_keyframes = []
    snap_b = {}
    baseline_end = None

    def render_keyframes_b(step):
        pil = render_full_map(step, abf, abi, road_elements, traffic_elements,
                              sdc_row_b, map_corners, dt, args.seed, map_bin.stem,
                              "baseline")
        path = baseline_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        baseline_keyframes.append(str(path.relative_to(seed_dir)))
        if any(step == w["mid_step"] for w in walkers):
            snap_b[step] = (abf.copy(), abi.copy())

    baseline_keyframe_steps = sorted(set(COMPARE_KEYFRAMES) | {w["mid_step"] for w in walkers})
    render_keyframes_b(0)
    for step in range(1, args.steps + 1):
        env_b.step(actions)
        capture_b()
        record_step_b()
        lo, hi = env_b.agent_offsets[0], env_b.agent_offsets[1]
        if baseline_end is None and bool(np.any(env_b.truncations[lo:hi])):
            baseline_end = step
            print(f"[hazard-demo] env0 baseline episode ended at step {step}")
            break
        if step in baseline_keyframe_steps:
            render_keyframes_b(step)
    baseline_metrics = {
        "steps_run": len(per_step_b) - 1,
        "ended_step": baseline_end if baseline_end is not None else f"ran to {len(per_step_b) - 1}",
        "collision_agents_final": int(np.sum((abi[:, 3] == 1) & (abi[:, 2] == 1)
                                             & (amf[:, COLLISION_IDX] > 0))),
    }
    env_b.close()

    # ---- reaction metrics -----------------------------------------------------
    Xj = np.stack([r["x"] for r in per_step]); Yj = np.stack([r["y"] for r in per_step])
    Vj = np.stack([r["v"] for r in per_step]); Lj = np.stack([r["valid"] for r in per_step])
    Cj = np.stack([r["coll"] for r in per_step])
    Xb = np.stack([r["x"] for r in per_step_b]); Yb = np.stack([r["y"] for r in per_step_b])
    Vb = np.stack([r["v"] for r in per_step_b]); Lb = np.stack([r["valid"] for r in per_step_b])
    Cb = np.stack([r["coll"] for r in per_step_b])
    len_b = len(per_step_b)

    ev_j_total = ev_b_total = hit_total = 0
    walker_outputs = []
    for w in walkers:
        t0 = w["start_step"]
        t1 = w["done_step"] if w["done_step"] is not None else steps_run
        pos = trail_arrays[w["row"]].astype(float)
        rows_j = _window_rows(pos, Xj, Yj, Lj, t0, t1, WALKER_REACTION_RADIUS_M)
        rows_b = _window_rows(pos, Xb, Yb, Lb, t0, min(t1, len_b - 1), WALKER_REACTION_RADIUS_M)
        ev_j = sum(slowdown_onsets(Vj[:, r], t0, t1) for r in rows_j)
        ev_b = sum(slowdown_onsets(Vb[:, r], t0, min(t1, len_b - 1)) for r in rows_b)
        if kernel_mode:
            # K2: completion/min-dist/contact come from the kernel getter (pure env truth,
            # OBB rising-edge contact); the demo-side window scan only feeds the slowdown
            # reaction comparison, which needs both episodes' speed captures either way.
            min_d = w["min_dist_to_vehicle_m"]
            arg = None
            contacts = []
            contact_total = w["contact_events"]
        else:
            min_d, arg = _window_min_dist(pos, Xj, Yj, Lj, t0, t1)
            contacts = contact_rows(pos, Xj, Yj, Cj, Lj, t0, t1, WALKER_HIT_DIST_M)
            contact_total = len(contacts)
        ev_j_total += ev_j
        ev_b_total += ev_b
        hit_total += contact_total
        walker_outputs.append({
            "k": w["k"], "kernel_row": int(w["row"]), "lane_id": w["lane_id"],
            "s": round(w["s"], 2) if w["s"] is not None else None,
            "start_m": [round(v, 2) for v in w["start"]],
            "end_m": ([round(v, 2) for v in trails[w["row"]][-1]] if kernel_mode
                      else [round(v, 2) for v in w["end"]]),
            "crossing_width_m": round(w["crossing_width_m"], 2),
            "speed_mps": round(w["speed_mps"], 3),
            "start_step": w["start_step"],
            "completed": w["done_step"] is not None,
            "completed_step": w["done_step"],
            "crossing_duration_s": w["crossing_duration_s"],
            "min_dist_to_vehicle_m": round(min_d, 3) if min_d is not None else None,
            "min_dist_capture_and_row": list(arg) if arg else None,
            "vehicles_in_radius": len(rows_j),
            "slowdown_events_jaywalker": ev_j,
            "slowdown_events_baseline": ev_b,
            "walker_hit_vehicle_rows": contacts,
            "kernel_contact_events": w["contact_events"] if kernel_mode else None,
            "tracking_max_err_m": (None if kernel_mode else round(float(w["tracking_max_err_m"]), 4)),
        })

    # ---- cut-in verdict (stretch; abandonment is an acceptable outcome) -------
    if kernel_mode:
        cut_out = {"attempted": False, "placed": False, "verdict": "skipped",
                   "reason": "kernel mode: walkers are c_step-driven; python cut-in puppeteering "
                             "is out of K2 scope (kernel alternative = per-agent IDM parameters)",
                   "alternative": CUT_IN_ALTERNATIVE}
    elif args.skip_cut_in:
        cut_out = {"attempted": False, "placed": False, "verdict": "skipped",
                   "reason": "--skip-cut-in", "alternative": CUT_IN_ALTERNATIVE}
    elif cut_in is None:
        cut_out = {"attempted": True, "placed": False, "verdict": "abandoned",
                   "reason": "no straight near-parallel lane pair with a collision-free start",
                   "alternative": CUT_IN_ALTERNATIVE}
    else:
        t0, t1 = cut_in["window"]
        t1 = min(t1, steps_run)
        pos = np.asarray(cut_trail, dtype=float)
        rows_j = _window_rows(pos, Xj, Yj, Lj, t0, t1, WALKER_REACTION_RADIUS_M)
        rows_b = _window_rows(pos, Xb, Yb, Lb, t0, min(t1, len_b - 1), WALKER_REACTION_RADIUS_M)
        ev_j = sum(slowdown_onsets(Vj[:, r], t0, t1) for r in rows_j)
        ev_b = sum(slowdown_onsets(Vb[:, r], t0, min(t1, len_b - 1)) for r in rows_b)
        contacts = contact_rows(pos, Xj, Yj, Cj, Lj, t0, t1, CUT_IN_CONTACT_DIST_M)
        tracked = cut_in["tracking_max_err_m"] <= CUT_IN_MAX_TRACKING_ERR_M
        kept = tracked and ev_j >= 1 and not contacts
        cut_out = {
            "attempted": True, "placed": True,
            "mechanism": "static co-sim slot puppeteered via set_agent_states (same python stepping as walkers)",
            "lane_id": cut_in["lane_id"], "adjacent_lane_id": cut_in["adjacent_lane_id"],
            "lateral_offset_m": round(cut_in["lateral_offset_m"], 2),
            "merge_steps": cut_in["merge_steps"],
            "cruise_mps": round(cut_in["cruise_mps"], 2),
            "window_steps": [t0, t1],
            "slowdown_events_jaywalker": ev_j,
            "slowdown_events_baseline": ev_b,
            "contact_vehicle_rows": contacts,
            "tracking_max_err_m": round(float(cut_in["tracking_max_err_m"]), 4),
            "verdict": "kept" if kept else "abandoned",
            "reason": ("merge completed, followers reacted, no scripted contact"
                       if kept else
                       ("tracking error >%.2f m" % CUT_IN_MAX_TRACKING_ERR_M if not tracked
                        else ("followers produced %d contact(s) during the merge" % len(contacts)
                              if contacts else
                              "0 follower slowdown events during the merge window"))),
            "alternative": None if kept else CUT_IN_ALTERNATIVE,
        }
        ev_j_total += cut_out["slowdown_events_jaywalker"]
        ev_b_total += cut_out["slowdown_events_baseline"]

    reaction = {
        "radius_m": WALKER_REACTION_RADIUS_M,
        "definition": (f"window=[start_step, done_step]; ref = vehicle speed at window start "
                       f"(floor {SLOWDOWN_REF_FLOOR_MPS} m/s); event = >= {SLOWDOWN_SUSTAIN_STEPS} "
                       f"consecutive captures at <= {SLOWDOWN_DROP_FRAC}*ref, onset-counted; "
                       "radius anchored to the scripted walker trajectory in BOTH runs"),
        "slowdown_events_baseline_total": ev_b_total,
        "slowdown_events_jaywalker_total": ev_j_total,
        "walker_contact_vehicles_total": hit_total,
    }

    # ---- comparison artifacts --------------------------------------------------
    missing = []
    for t in COMPARE_KEYFRAMES:
        b_png = baseline_dir / f"env0_t{t:03d}.png"
        h_png = jw_dir / f"env0_t{t:03d}.png"
        if b_png.is_file() and h_png.is_file():
            targets = []
            for w in walkers:
                wpos = trails[w["row"]][min(t, len(trails[w["row"]]) - 1)]
                targets.append((f"walker{w['k']} jaywalker", "jaywalker",
                                wpos[0], wpos[1], GIF_COLORS["walker"]))
            if cut_in:
                cpt = cut_trail[min(t, len(cut_trail) - 1)]
                targets.append(("cut-in maker", "cut-in vehicle", cpt[0], cpt[1],
                                HAZARD_CLASS_COLORS["cut_in"]))
            h_annot = add_walker_callouts(Image.open(h_png), targets,
                                          full_map_world_to_px(map_corners), cjk_font)
            side_by_side(b_png, h_annot, compare_dir / f"compare_t{t:03d}.png")
        else:
            missing.append(t)
    zooms = []
    for w in walkers:
        t = w["mid_step"]
        if t not in snap_jw or t not in snap_b:
            missing.append(f"zoom_walker{w['k']}_t{t:03d}")
            continue
        ctr = w["cross_pt"]
        zoom_w2p = lambda x, y, ctr=ctr: _world_to_px(x, y, ctr, ZOOM_PX / ZOOM_WINDOW_M, ZOOM_PX)
        b_img = render_gif_frame(t, snap_b[t][0], snap_b[t][1], road_elements,
                                 traffic_elements, sdc_row_b, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                 dt, args.seed, map_bin.stem, mode_label="baseline",
                                 min_cone_px=0, walker_min_px=0)
        j_img = render_gif_frame(t, snap_jw[t][0], snap_jw[t][1], road_elements,
                                 traffic_elements, sdc_row, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                 dt, args.seed, map_bin.stem, mode_label=mode_label,
                                 static_indices=static_indices, static_class_of=class_of,
                                 min_cone_px=0, walker_indices=walker_rows,
                                 walker_trails={r: p[:t + 1] for r, p in trail_arrays.items()},
                                 walker_min_px=0)
        wpos = trails[w["row"]][min(t, len(trails[w["row"]]) - 1)]
        targets = [(f"walker{w['k']} jaywalker", "jaywalker", wpos[0], wpos[1],
                    GIF_COLORS["walker"])]
        if cut_in:
            cpt = cut_trail[min(t, len(cut_trail) - 1)]
            targets.append(("cut-in maker", "cut-in vehicle", cpt[0], cpt[1],
                            HAZARD_CLASS_COLORS["cut_in"]))
        j_img = add_walker_callouts(j_img, targets, zoom_w2p, cjk_font)
        zoom_name = f"zoom_walker{w['k']}_t{t:03d}.png"
        side_by_side(b_img, j_img, compare_dir / zoom_name)
        zooms.append(f"compare/{zoom_name}")

    pil = render_full_map(steps_run, af0, ai0, road_elements, traffic_elements, sdc_row,
                          map_corners, dt, args.seed, map_bin.stem, mode_label,
                          static_indices=static_indices, static_class_of=class_of,
                          walker_indices=walker_rows, walker_trails=trail_arrays)
    pil = add_trail_annotations(pil, walkers, trail_arrays,
                                full_map_world_to_px(map_corners), cut_in, cut_trail, cjk_font)
    pil.save(seed_dir / "trails.png")

    # pixel-existence audit: every image that should show walkers must contain
    # walker-magenta pixels (hazard keyframes + compares + crossing zooms + the
    # trails overlay); any 0 -> hard fail (same mechanism as the cone audit)
    audit_paths = (list(keyframe_paths)
                   + [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                      if t not in missing]
                   + zooms + ["trails.png"])
    walker_audit = audit_pixels(seed_dir, audit_paths, WALKER_RGB, "walker")
    print(f"[hazard-demo] walker pixel audit: {len(walker_audit)} images, "
          f"min count {min(walker_audit.values()) if walker_audit else 0}")

    notes = [
        "已知偏差（契约要求明录）：内核 agent type 仍为 1（VEHICLE）——行人实为 0.5×0.5m 静态槽小车，"
        "观测/碰撞按小车对待（零内核改动，不写 type 2 PEDESTRIAN）；set_agent_states 只写 vec->envs[0]"
        "（cosim 单 env 假设）→ 本模式强制 --envs 1。",
        "步进机制：walker 为 cosim_partner_slots 静态槽，每步先 set_agent_states 写新位姿再 env.step()"
        "（车流决策基于上一步观测，collision/TTC 看到当前位姿）；c_step 不移动静态槽；"
        "脚本位姿 vs 内核回读逐步最大误差在 walkers[].tracking_max_err_m（<1e-3 m 量级）。",
        "mid-block 口径：直车道（逐段航向变化 <0.35 rad）且距任何斑马线（type 31）≥12 m——"
        "map bin 无 type 5-9 路口元素，斑马线距离即避路口判据；起点 = 近侧路缘外 1.2 m"
        "（上限 5 m，不做 off-road 检查——jaywalker 本就违例），但起点/终点 OBB 与动态车辆无重叠"
        "（动态重叠检查保留）。",
        "度量口径：横穿窗口 = [start_step, done_step]（未完成用终点步）；反应半径 15 m，"
        "锚定 walker 脚本轨迹（baseline 用同一轨迹锚点对比本 run 自己的车辆）；"
        "明显减速 ref = 该车窗口首步速度（≥2 m/s 底线），事件 = 连续 ≥3 步速度 ≤0.5*ref，"
        "按 onset 去重；最小距离 = walker 与车辆中心距，含锁停未移除车辆（removed 排除）。",
        "车辆撞到 walker → 该车 latch stopped（collision_behavior=stop）；窗口内 3 m 内接触车辆"
        "记入 walker_hit_vehicle_rows。",
        "cut-in（stretch）：与 walker 同一 Python 步进机制（静态槽 + 每步 set_agent_states），"
        "相邻车道起步、横向 0.3 m/步（dt=0.3s ≈1 m/s）挤入目标车道，纵向巡航 = seed 车流中位速"
        "（截断 4-9 m/s）；无 victim 选择模型，起步点无动态重叠，接触事件如实记录；"
        "verdict 判据 = 跟踪误差 ≤0.25 m 且合并窗口 followers ≥1 次减速且零接触，"
        "不满足即 abandoned 并给替代方案（内核 per-agent IDM 参数）。",
        "渲染遵循长期标准：单独 legend.png（index.html 首项，新增 行人=品红圆点+轨迹线、"
        "cut-in=紫色 两行；baseline/static-hazards 图例逐字不变）；类别着色不按 id；"
        "walker 全图强制 ≥12px 圆点，zoom 按真实尺度；walker_pixel_counts 对所有应含 walker 的图"
        "做像素审计，任何一张为 0 即硬失败（机制沿用锥桶审计）；zoom 对准每个 walker 横穿中点时刻；"
        "trails.png = 最终帧叠加全程轨迹 + 起点/终点标记。",
        "回归约束：baseline / static-hazards 代码路径未动（渲染器新参数全部默认关闭，"
        "图例仅 jaywalker 模式增行），三 seed 数字已验证 byte-identical。",
    ]
    seed_summary = {
        "mode": mode_label,
        "kernel": kernel_provenance(),
        "seed": args.seed,
        "map": {"name": map_bin.stem, "path": str(map_bin)},
        "agents_per_env": args.agents_per_env,
        "steps_requested": args.steps,
        "dt": dt,
        "placement": {
            "walkers_placed": len(walkers),
            "reject_reasons": {k: v for k, v in walker_rejects.items() if v},
            "slots_used": len(initial),
            "readback_max_err_m": round(float(place_err), 6),
            "dynamic_overlap_violations_postcheck": int(post_overlap),
        },
        "walkers": walker_outputs,
        "cut_in": cut_out,
        "reaction": reaction,
        "simulation": {
            "window_step": min(steps_run, len_b - 1),
            "collision_agents_baseline_final": baseline_metrics["collision_agents_final"],
            "collision_agents_jaywalker_final": jw_metrics["collision_agents_final"],
            "baseline": baseline_metrics,
            "jaywalker": jw_metrics,
        },
        "render_audit": {
            "walker_color_rgb": list(WALKER_RGB),
            "walker_min_diam_px_full_map": WALKER_MIN_DIAM_PX,
            "zoom_scale": "true size (walker_min_px=0)",
            "walker_pixel_counts": walker_audit,
            "images_checked": len(walker_audit),
            "all_nonzero": True,
        },
        "artifacts": {
            "baseline_keyframes": baseline_keyframes,
            "jaywalker_keyframes": keyframe_paths,
            "jaywalker_gif": "jaywalker/jaywalker.gif" if gif_imgs else None,
            "compare": [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                        if t not in missing],
            "zooms": zooms,
            "trails_overlay": "trails.png",
        },
        "missing_artifacts": missing,
        "runtime_s": round(time.time() - t_start, 1),
        "notes": notes,
    }
    (seed_dir / "summary.json").write_text(
        json.dumps(seed_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    seed_summaries = []
    for d in sorted(root.glob("seed*/summary.json")):
        try:
            seed_summaries.append(json.loads(d.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    write_jaywalker_root(root, seed_summaries, cjk_path=cjk_font, mode_label=mode_label)
    comp = sum(1 for w in walker_outputs if w["completed"])
    print(f"[hazard-demo] artifacts -> {root} ({seed_summary['runtime_s']}s); "
          f"crossed {comp}/{len(walker_outputs)}, slowdown events "
          f"baseline={ev_b_total} vs jaywalker={ev_j_total}, "
          f"cut_in={cut_out['verdict']}")


def write_reactions_chart(root, seed_summaries):
    """Two-panel aggregate: per-walker min distance to traffic during crossing;
    slowdown events baseline vs jaywalker per seed (same windows, same anchor)."""
    import matplotlib.pyplot as plt

    seed_summaries = sorted(seed_summaries, key=lambda s: s["seed"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    palette = plt.cm.tab10(np.arange(len(seed_summaries)))
    labels, vals, bar_colors = [], [], []
    for si, s in enumerate(seed_summaries):
        for w in s["walkers"]:
            labels.append(f"s{s['seed']}w{w['k']}")
            vals.append(w["min_dist_to_vehicle_m"] if w["min_dist_to_vehicle_m"] is not None else 0.0)
            bar_colors.append(palette[si])
    ax1.bar(range(len(vals)), vals, color=bar_colors)
    ax1.axhline(2.0, color="red", linestyle="--", linewidth=1, label="2 m reference")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=45)
    ax1.set_ylabel("min distance walker-vehicle (m)")
    ax1.set_title("min distance during crossing (per walker)")
    ax1.legend(fontsize=8)
    seeds = [str(s["seed"]) for s in seed_summaries]
    b_vals = [s["reaction"]["slowdown_events_baseline_total"] for s in seed_summaries]
    j_vals = [s["reaction"]["slowdown_events_jaywalker_total"] for s in seed_summaries]
    xpos = np.arange(len(seeds))
    width = 0.35
    ax2.bar(xpos - width / 2, b_vals, width, label="baseline", color="#8c8c8c")
    ax2.bar(xpos + width / 2, j_vals, width, label="jaywalker", color="#FF00FF")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(seeds)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("obvious-slowdown events (15 m radius)")
    ax2.set_title("traffic slowdown events: baseline vs jaywalker")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "reactions.png", dpi=140)
    plt.close(fig)


def write_jaywalker_root(root, seed_summaries, cjk_path=None, mode_label="jaywalker"):
    """Aggregate legend.png / reactions.png / summary.json / index.html over seed dirs."""
    render_legend_png(root / "legend.png", mode_label, cjk_path=cjk_path)
    all_w = [w for s in seed_summaries for w in s["walkers"]]
    mind = [w["min_dist_to_vehicle_m"] for w in all_w
            if w["min_dist_to_vehicle_m"] is not None]
    completed = sum(1 for w in all_w if w["completed"])
    aggregate = {
        "walkers_total": len(all_w),
        "completed_total": completed,
        "completion_rate": round(completed / len(all_w), 3) if all_w else None,
        "min_dist_m": ({"min": round(float(np.min(mind)), 3),
                        "p50": round(float(np.percentile(mind, 50)), 3),
                        "p95": round(float(np.percentile(mind, 95)), 3)} if mind else None),
        "slowdown_events_baseline_total": sum(s["reaction"]["slowdown_events_baseline_total"]
                                              for s in seed_summaries),
        "slowdown_events_jaywalker_total": sum(s["reaction"]["slowdown_events_jaywalker_total"]
                                               for s in seed_summaries),
        "walker_contact_vehicles_total": sum(s["reaction"]["walker_contact_vehicles_total"]
                                             for s in seed_summaries),
        "cut_in_verdicts": {str(s["seed"]): s["cut_in"].get("verdict")
                            for s in sorted(seed_summaries, key=lambda x: x["seed"])},
    }
    agg = {
        "mode": mode_label,
        "kernel": kernel_provenance(),
        "seeds": [s["seed"] for s in seed_summaries],
        "aggregate": aggregate,
        "per_seed": [{"seed": s["seed"], "walkers": len(s["walkers"]),
                      "completed": sum(1 for w in s["walkers"] if w["completed"]),
                      "reaction": s["reaction"],
                      "cut_in_verdict": s["cut_in"].get("verdict"),
                      "render_audit": {"images_checked": s["render_audit"]["images_checked"],
                                       "all_nonzero": s["render_audit"]["all_nonzero"]}}
                     for s in sorted(seed_summaries, key=lambda x: x["seed"])],
        "notes": (seed_summaries[-1].get("notes") if seed_summaries else []),
    }
    (root / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not seed_summaries:
        return
    try:
        write_reactions_chart(root, seed_summaries)
    except Exception as exc:  # chart is cosmetic; never sink the run
        print(f"[hazard-demo] reactions chart skipped: {exc}")

    rows = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        md = [w["min_dist_to_vehicle_m"] for w in s["walkers"]
              if w["min_dist_to_vehicle_m"] is not None]
        md_txt = (f"min={min(md):.2f} / p50={np.percentile(md, 50):.2f} / "
                  f"p95={np.percentile(md, 95):.2f}" if md else "n/a")
        comp = sum(1 for w in s["walkers"] if w["completed"])
        r = s["reaction"]
        cells = (f"<td>{comp}/{len(s['walkers'])}</td><td>{md_txt}</td>"
                 f"<td>{r['slowdown_events_baseline_total']} → "
                 f"{r['slowdown_events_jaywalker_total']}</td>"
                 f"<td>{r['walker_contact_vehicles_total']}</td>"
                 f"<td>{s['cut_in'].get('verdict')} — {s['cut_in'].get('reason', '')}</td>")
        rows += f"<tr><td>seed {s['seed']}</td>{cells}</tr>"
    galleries = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        sd = f"seed{s['seed']}"
        frames = "".join(
            f'<figure><img src="{sd}/jaywalker/frames/{p.name}"/>'
            f"<figcaption>{p.name}</figcaption></figure>"
            for p in sorted((root / sd / "jaywalker" / "frames").glob("*.png")))
        compares = "".join(
            f'<figure><img src="{sd}/compare/{p.name}"/>'
            f"<figcaption>{p.name}（左 baseline / 右 jaywalker）</figcaption></figure>"
            for p in sorted((root / sd / "compare").glob("*.png")))
        trails_img = (f'<figure><img src="{sd}/trails.png"/>'
                      f"<figcaption>trails.png — 最终帧叠加全程轨迹</figcaption></figure>"
                      if (root / sd / "trails.png").is_file() else "")
        gif = (f'<img src="{sd}/jaywalker/jaywalker.gif" width="560"/>'
               if (root / sd / "jaywalker" / "jaywalker.gif").is_file() else "")
        galleries += (f"<h2>seed {s['seed']}</h2>"
                      f"<h3>jaywalker 全图帧</h3>{frames}"
                      f"<h3>compare / 横穿事件 zoom</h3>{compares}"
                      f"<h3>轨迹叠加（最终帧 + 全程 trail）</h3>{trails_img}"
                      f"<h3>跟车 GIF（镜头对准横穿点）</h3>{gif}\n")
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Slipstream hazard-demo · M2 jaywalker</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1, h2, h3 {{ color: #333; }} table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
  th {{ background: #eee; }} figure {{ margin: 8px; display: inline-block; }}
  img {{ border: 1px solid #ddd; max-width: 640px; }} figcaption {{ font-size: 12px; color: #666; }}
  code {{ background: #eee; padding: 2px 6px; }}
</style></head><body>
  <h1>M2 jaywalker · python-stepped 行人横穿 + 反应度量</h1>
  <p>命令: <code>PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode jaywalker --seed &lt;42|43|44&gt;</code></p>
  <h2>图例（legend.png 单独成图，index 首项；行人=品红圆点+轨迹线）</h2>
  <img src="legend.png" width="480"/>
  <img src="reactions.png" width="680"/>
  <h2>汇总</h2>
  <ul>
    <li>完成横穿 {aggregate['completed_total']}/{aggregate['walkers_total']}
        （完成率 {aggregate['completion_rate']}）；最小距离分布
        {aggregate['min_dist_m']}；减速事件 baseline {aggregate['slowdown_events_baseline_total']}
        vs jaywalker {aggregate['slowdown_events_jaywalker_total']}；
        walker 接触车辆 {aggregate['walker_contact_vehicles_total']}。</li>
    <li>已知偏差：内核 type 仍为 1 (VEHICLE)，行人按小车对待；set_agent_states 只写 envs[0] → 强制 --envs 1。</li>
    <li>像素审计：summary.json render_audit.walker_pixel_counts，任何一张应含 walker 的图品红像素为 0 即硬失败。</li>
  </ul>
  <h2>每 seed 度量</h2>
  <table><tr><th>seed</th><th>完成横穿</th><th>最小距离 (m)</th><th>减速事件 baseline → jaywalker</th><th>walker 接触车辆</th><th>cut-in verdict</th></tr>
  {rows}</table>
  {galleries}
  <h2>口径 / notes</h2><ul>{''.join(f'<li>{n}</li>' for n in agg['notes'])}</ul>
</body></html>"""
    (root / "index.html").write_text(html, encoding="utf-8")


def run_mix(args, map_bin, cfg):
    """M3: config-driven mix — every class enabled in the hazard config shares
    ONE episode. Static classes (parked/collided/cones) are co-sim slots placed
    once; jaywalker walkers and the cut-in maker are python-stepped every step
    (same mechanisms as M1/M2). A same-seed clean baseline episode provides the
    compare side and the reaction windows. Co-sim setters only reach vec env 0
    -> envs=1 (forced).

    Prototype boundary: the config knobs drive this demo's placement logic
    only — no kernel change (C reset placement / PEDESTRIAN type / per-agent
    IDM are next-phase spec-ticket scope). Regression contract: with the
    config fully disabled the mix build reserves zero co-sim slots and places
    nothing, so the episode is bit-identical to the pure gigaflow baseline."""
    if args.envs != 1:
        print("[hazard-demo] mix: co-sim setters only reach vec env 0; forcing --envs 1")
        args.envs = 1
    if args.walkers is not None:
        raise SystemExit("--walkers applies to --mode jaywalker only; mix reads the "
                         "walker count from hazard config jaywalker.n_range")
    if args.skip_cut_in:
        raise SystemExit("--skip-cut-in applies to --mode jaywalker only; mix reads "
                         "cut_in.enable from the hazard config")

    root = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    seed_dir = root / f"seed{args.seed}"
    baseline_dir = seed_dir / "baseline" / "frames"
    mix_dir = seed_dir / "mix" / "frames"
    compare_dir = seed_dir / "compare"
    for d in (baseline_dir, mix_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    enabled = [k for k in ("parked", "clusters", "cones", "jaywalker", "cut_in")
               if cfg[k]["enable"]]
    print(f"[hazard-demo] mode=mix seed={args.seed} map={map_bin} config={cfg.get('_path')}")
    print(f"[hazard-demo] global enable={cfg['enable']}; enabled classes: "
          f"{enabled if cfg['enable'] else 'NONE (pure gigaflow regression run)'}")
    cjk_font = detect_cjk_font()
    print(f"[hazard-demo] CJK font: {cjk_font or 'none -> English-only labels (no tofu)'}")

    # ---- episode 1: mix ------------------------------------------------------
    # Capacity from the config upper bound; all-disabled -> 0 slots -> the
    # build is byte-identical to the plain baseline build (regression gate b).
    slots = mix_slot_capacity(cfg)
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
    af0, ai0, mf0 = buffers["agent_f32"][0], buffers["agent_i32"][0], buffers["metrics_f32"][0]
    lanes = LaneIndex(road_elements)
    edges = EdgeIndex(road_elements)
    cw_segs = crosswalk_segments(road_elements)
    dyn = dynamic_boxes(af0, ai0)

    # ---- placement (config-driven) ------------------------------------------
    rng = hazard_rng(args.seed)
    wrng = walker_rng(args.seed)
    rejects = {r: 0 for r in REJECT_REASONS}
    walker_rejects = {r: 0 for r in WALKER_REJECT_REASONS}
    placed, walkers, cut_in = [], [], None
    plan = {"parked": 0, "clusters": 0, "cone_zones": 0}
    walkers_planned = 0
    if cfg["enable"]:
        pc, cc, kc, jc = cfg["parked"], cfg["clusters"], cfg["cones"], cfg["jaywalker"]
        plan = plan_hazard_counts(
            rng,
            parked_range=tuple(pc["n_range"]) if pc["enable"] else (0, 0),
            cluster_range=tuple(cc["n_range"]) if cc["enable"] else (0, 0),
            zone_range=tuple(kc["zones_range"]) if kc["enable"] else (0, 0),
        )
        size_table = None
        if pc["enable"]:
            size_table = tuple(zip(pc["sizes"], [VEHICLE_PRESETS[s] for s in pc["sizes"]],
                                   pc["size_weights"]))
        parked_n = place_parked(rng, lanes, edges, dyn, placed, rejects, plan["parked"],
                                size_table=size_table)
        clusters_n = place_clusters(rng, lanes, edges, dyn, placed, rejects, plan["clusters"],
                                    layouts=cc["layouts"])
        zones_n, cones_n = place_cone_zones(rng, lanes, edges, dyn, placed, rejects,
                                            plan["cone_zones"], cones=kc)
        print(f"[hazard-demo] plan={plan} placed: parked={parked_n} clusters={clusters_n} "
              f"cone_zones={zones_n} ({cones_n} cones)")
        if jc["enable"]:
            walkers_planned = int(wrng.randint(jc["n_range"][0], jc["n_range"][1] + 1))
            walkers, walker_rejects = plan_walkers(wrng, lanes, edges, cw_segs, dyn,
                                                   walkers_planned, static_placed=placed)

    for k, w in enumerate(walkers):
        w["row"] = args.agents_per_env + len(placed) + k
        w["speed_mps"] = float(wrng.uniform(*cfg["jaywalker"]["speed_range"]))
        w["start_step"] = 1 + k * WALKER_START_STAGGER_STEPS
        est = int(np.ceil(w["crossing_width_m"] / (w["speed_mps"] * args.dt)))
        w["mid_step"] = min(w["start_step"] + max(est // 2, 1), args.steps - 1)
        w["traveled"] = 0.0
        w["done_step"] = None
        w["tracking_max_err_m"] = 0.0
    valid0 = (ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (ai0[:, 5] == 0)
    med_v = float(np.median(af0[valid0, 6])) if valid0.any() else 6.0
    cruise = float(np.clip(med_v, *CUT_IN_CRUISE_RANGE)) if np.isfinite(med_v) else 6.0
    if cfg["enable"] and cfg["cut_in"]["enable"]:
        cut_in = plan_cut_in(wrng, lanes, dyn, walkers, cruise)
    if cut_in:
        cut_in["row"] = args.agents_per_env + len(placed) + len(walkers)
        cut_in["merge_steps"] = int(np.ceil(cut_in["lateral_offset_m"] / CUT_IN_DLATERAL_PER_STEP))
        cut_in["start_step"] = 1
        cut_in["window"] = (1, min(1 + cut_in["merge_steps"] + CUT_IN_TAIL_STEPS, args.steps))
        cut_in["tracking_max_err_m"] = 0.0

    initial = placed + walkers + ([cut_in] if cut_in else [])
    print(f"[hazard-demo] walkers={len(walkers)} "
          f"cut_in={'yes (merge %.1f m over %d steps)' % (cut_in['lateral_offset_m'], cut_in['merge_steps']) if cut_in else 'disabled' if not cfg['cut_in']['enable'] else 'abandoned (no usable lane pair)'} "
          f"slots={len(initial)}/{slots} cruise={cruise:.1f} m/s")
    if cfg["enable"] and not initial:
        raise RuntimeError("hazard config enables at least one class but nothing could be "
                           "placed on this map/seed; check the config or pick another seed")
    if len(initial) > slots:
        raise RuntimeError(f"mix slots exhausted: {len(initial)} > {slots}; tighten the config ranges")

    if initial:
        apply_static_hazards(env, args, initial)
        capture()
    g_state = env.get_global_agent_state(include_static=True)
    place_err = 0.0 if not initial else max(
        max(abs(float(g_state["x"][e["idx"]]) - e["x"]),
            abs(float(g_state["y"][e["idx"]]) - e["y"])) for e in initial)
    post_overlap = sum(1 for e in initial
                       if any(obb_overlap((e["x"], e["y"], e["heading"], e["length"], e["width"]),
                                          d, margin=0.05) for d in dyn))
    # placement-time legality (M1 semantics: checked against the fresh spawn
    # layout at step 0, BEFORE any vehicle drives into a hazard; walkers are
    # intentionally off-road so they stay out of the static edge/lane checks)
    if placed:
        legality = verify_static_hazards(env, placed, [e["idx"] for e in placed],
                                         buffers["agent_f32"][0], buffers["agent_i32"][0])
    else:
        legality = {"kernel_lane_assoc_failures": 0,
                    "kernel_lane_assoc_failures_detail": [],
                    "dynamic_overlap_violations_postcheck": 0,
                    "intentional_intra_cluster_overlaps": 0,
                    "world_readback_max_err_m": 0.0,
                    "static_agent_indices_head": []}
    legality = {**legality,
                "all_slots_dynamic_overlap_postcheck": int(post_overlap),
                "world_readback_max_err_all_slots_m": round(float(place_err), 6)}
    print(f"[hazard-demo] placement readback_err={place_err} dynamic_overlap_postcheck={post_overlap}")
    print(f"[hazard-demo] legality: {legality}")

    static_class_of = {e["idx"]: e["kind"] for e in initial}
    static_indices = list(static_class_of)
    walker_rows = [w["row"] for w in walkers]
    zone_centers = {}
    for e in placed:
        key = e.get("cluster_id") or e.get("zone_id")
        if key:
            zone_centers.setdefault(key, []).append((e["x"], e["y"]))
    trails = {w["row"]: [(float(af0[w["row"], 0]), float(af0[w["row"], 1]))] for w in walkers}
    cut_trail = ([(float(af0[cut_in["row"], 0]), float(af0[cut_in["row"], 1]))]
                 if cut_in else None)

    per_step = []

    def record_step():
        valid = (ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (ai0[:, 5] == 0)
        per_step.append({"x": af0[:, 0].copy(), "y": af0[:, 1].copy(),
                         "v": af0[:, 6].copy(), "valid": valid.copy(),
                         "coll": (mf0[:, COLLISION_IDX] > 0).copy()})

    record_step()
    keyframe_paths = []
    snap_mix = {}  # step -> buffer copies for true-scale zoom re-renders

    def render_keyframes(step):
        pil = render_full_map(step, af0, ai0, road_elements, traffic_elements, sdc_row,
                              map_corners, args.dt, args.seed, map_bin.stem, "mix",
                              static_indices=static_indices, static_class_of=static_class_of,
                              walker_indices=walker_rows,
                              walker_trails={r: np.asarray(p) for r, p in trails.items()})
        path = mix_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        keyframe_paths.append(str(path.relative_to(seed_dir)))
        if step in COMPARE_KEYFRAMES or any(step == w["mid_step"] for w in walkers):
            snap_mix[step] = (af0.copy(), ai0.copy())

    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    steps_run = 0
    episode_end = None
    gif_frames = []
    collision_curve = []
    if walkers:
        gif_center = (walkers[0]["cross_pt"][0], walkers[0]["cross_pt"][1])
    else:
        mask0 = env_active_mask(ai0)
        gif_center = (float(np.mean(af0[mask0, 0])), float(np.mean(af0[mask0, 1])))
    mix_keyframes = sorted(set(HAZARD_KEYFRAMES) | {w["mid_step"] for w in walkers})
    render_keyframes(0)
    gif_frames.append((0, af0.copy(), ai0.copy()))

    for step in range(1, args.steps + 1):
        idxs, xs, ys, zs, hs, vxs, vys, written = [], [], [], [], [], [], [], {}
        for w in walkers:
            if w["done_step"] is not None or step < w["start_step"]:
                continue  # standing at the kerb / already across (c_step never moves slots)
            w["traveled"] += w["speed_mps"] * args.dt
            if w["traveled"] >= w["crossing_width_m"]:
                w["traveled"] = w["crossing_width_m"]
                w["done_step"] = step
            px = w["start"][0] + w["dir"][0] * w["traveled"]
            py = w["start"][1] + w["dir"][1] * w["traveled"]
            v = w["speed_mps"] if w["done_step"] is None else 0.0
            idxs.append(w["row"])
            xs.append(px); ys.append(py); zs.append(w["z"]); hs.append(w["heading"])
            vxs.append(w["dir"][0] * v); vys.append(w["dir"][1] * v)
            written[w["row"]] = (px, py)
        if cut_in is not None:
            px, py, pz, ph = cut_in_pose(cut_in, step, args.dt)
            idxs.append(cut_in["row"])
            xs.append(px); ys.append(py); zs.append(pz); hs.append(ph)
            vxs.append(cut_in["dir"][0] * cut_in["cruise_mps"])
            vys.append(cut_in["dir"][1] * cut_in["cruise_mps"])
            written[cut_in["row"]] = (px, py)
        if idxs:
            env.set_agent_states(
                np.asarray(idxs, np.int32),
                np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                np.asarray(zs, np.float32), np.asarray(hs, np.float32),
                np.asarray(vxs, np.float32), np.asarray(vys, np.float32),
                np.zeros(len(idxs), np.float32), np.zeros(len(idxs), np.float32),
                None, np.zeros(len(idxs), np.float32),
            )
        env.step(actions)
        steps_run = step
        capture()
        record_step()
        live = (ai0[:, 3] == 1) & (ai0[:, 2] == 1)
        collision_curve.append(int(np.sum(live & (mf0[:, COLLISION_IDX] > 0))))
        for w in walkers:
            trails[w["row"]].append((float(af0[w["row"], 0]), float(af0[w["row"], 1])))
            if w["row"] in written:
                px, py = written[w["row"]]
                w["tracking_max_err_m"] = max(
                    w["tracking_max_err_m"],
                    max(abs(px - float(af0[w["row"], 0])), abs(py - float(af0[w["row"], 1]))))
        if cut_in is not None:
            cut_trail.append((float(af0[cut_in["row"], 0]), float(af0[cut_in["row"], 1])))
            if cut_in["row"] in written:
                px, py = written[cut_in["row"]]
                cut_in["tracking_max_err_m"] = max(
                    cut_in["tracking_max_err_m"],
                    max(abs(px - float(af0[cut_in["row"], 0])),
                        abs(py - float(af0[cut_in["row"], 1]))))
        lo, hi = env.agent_offsets[0], env.agent_offsets[1]
        if episode_end is None and bool(np.any(env.truncations[lo:hi])):
            episode_end = step
            print(f"[hazard-demo] env0 mix episode ended at step {step}")
            break
        if step in mix_keyframes:
            render_keyframes(step)
        if args.gif_stride > 0 and step % args.gif_stride == 0:
            gif_frames.append((step, af0.copy(), ai0.copy()))

    for w in walkers:
        w["crossing_duration_s"] = (round((w["done_step"] - w["start_step"] + 1) * args.dt, 2)
                                    if w["done_step"] is not None else None)
    mix_offroad = int(np.sum((ai0[:, 3] == 1) & (ai0[:, 2] == 1) & (mf0[:, OFFROAD_IDX] > 0)))
    mix_metrics = {
        "steps_run": steps_run,
        "ended_step": episode_end if episode_end is not None else f"ran to {steps_run}",
        "collision_agents_final": int(collision_curve[-1]) if collision_curve else 0,
        "offroad_agents_final": mix_offroad,
    }
    env.close()

    trail_arrays = {r: np.asarray(p) for r, p in trails.items()}
    gif_path = seed_dir / "mix" / "mix.gif"
    gif_imgs = [
        render_gif_frame(st, af, ai, road_elements, traffic_elements, sdc_row,
                         gif_center, HAZARD_GIF_WINDOW_M, 900, args.dt, args.seed,
                         map_bin.stem, mode_label="mix",
                         static_indices=static_indices, static_class_of=static_class_of,
                         walker_indices=walker_rows, walker_trails=trail_arrays)
        for st, af, ai in gif_frames
    ]
    if gif_imgs:
        gif_imgs[0].save(gif_path, save_all=True, append_images=gif_imgs[1:],
                         duration=int(1000 / args.gif_fps), loop=0)

    # ---- episode 2: same-seed clean baseline (comparison side) ---------------
    env_b = build_env(args, map_bin)
    env_b.reset(seed=args.seed)
    scenarios_b = env_b.get_state()
    buffers_b = capture_buffers(env_b, scenarios_b)
    sdc_row_b = int((scenarios_b[0].get("active_agent_indices") or [0])[0])

    def capture_b():
        env_b.get_obs_html_frame(
            buffers_b["agent_f32"], buffers_b["agent_i32"], buffers_b["metrics_f32"],
            buffers_b["puffer_f32"], buffers_b["traffic_i16"], buffers_b["goals_f32"],
            buffers_b["rewards_f32"], buffers_b["coefs_f32"],
        )

    capture_b()
    abf, abi, amf = buffers_b["agent_f32"][0], buffers_b["agent_i32"][0], buffers_b["metrics_f32"][0]
    per_step_b = []

    def record_step_b():
        valid = (abi[:, 3] == 1) & (abi[:, 2] == 1) & (abi[:, 5] == 0)
        per_step_b.append({"x": abf[:, 0].copy(), "y": abf[:, 1].copy(),
                           "v": abf[:, 6].copy(), "valid": valid.copy(),
                           "coll": (amf[:, COLLISION_IDX] > 0).copy()})

    record_step_b()
    baseline_keyframes = []
    snap_b = {}
    baseline_end = None

    def render_keyframes_b(step):
        pil = render_full_map(step, abf, abi, road_elements, traffic_elements,
                              sdc_row_b, map_corners, args.dt, args.seed, map_bin.stem,
                              "baseline")
        path = baseline_dir / f"env0_t{step:03d}.png"
        pil.save(path)
        baseline_keyframes.append(str(path.relative_to(seed_dir)))
        if step in COMPARE_KEYFRAMES or any(step == w["mid_step"] for w in walkers):
            snap_b[step] = (abf.copy(), abi.copy())

    render_keyframes_b(0)
    for step in range(1, args.steps + 1):
        env_b.step(actions)
        capture_b()
        record_step_b()
        lo, hi = env_b.agent_offsets[0], env_b.agent_offsets[1]
        if baseline_end is None and bool(np.any(env_b.truncations[lo:hi])):
            baseline_end = step
            print(f"[hazard-demo] env0 baseline episode ended at step {step}")
            break
        if step in sorted(set(COMPARE_KEYFRAMES) | {w["mid_step"] for w in walkers}):
            render_keyframes_b(step)
    baseline_offroad = int(np.sum((abi[:, 3] == 1) & (abi[:, 2] == 1) & (amf[:, OFFROAD_IDX] > 0)))
    baseline_metrics = {
        "steps_run": len(per_step_b) - 1,
        "ended_step": baseline_end if baseline_end is not None else f"ran to {len(per_step_b) - 1}",
        "collision_agents_final": int(np.sum((abi[:, 3] == 1) & (abi[:, 2] == 1)
                                             & (amf[:, COLLISION_IDX] > 0))),
        "offroad_agents_final": baseline_offroad,
    }
    env_b.close()

    # ---- interaction metrics (crossing / contact / slowdown) ------------------
    Xj = np.stack([r["x"] for r in per_step]); Yj = np.stack([r["y"] for r in per_step])
    Vj = np.stack([r["v"] for r in per_step]); Lj = np.stack([r["valid"] for r in per_step])
    Cj = np.stack([r["coll"] for r in per_step])
    Xb = np.stack([r["x"] for r in per_step_b]); Yb = np.stack([r["y"] for r in per_step_b])
    Vb = np.stack([r["v"] for r in per_step_b]); Lb = np.stack([r["valid"] for r in per_step_b])
    Cb = np.stack([r["coll"] for r in per_step_b])
    len_b = len(per_step_b)

    ev_j_total = ev_b_total = hit_total = 0
    walker_outputs = []
    for w in walkers:
        t0 = w["start_step"]
        t1 = w["done_step"] if w["done_step"] is not None else steps_run
        pos = trail_arrays[w["row"]].astype(float)
        rows_j = _window_rows(pos, Xj, Yj, Lj, t0, t1, WALKER_REACTION_RADIUS_M)
        rows_b = _window_rows(pos, Xb, Yb, Lb, t0, min(t1, len_b - 1), WALKER_REACTION_RADIUS_M)
        ev_j = sum(slowdown_onsets(Vj[:, r], t0, t1) for r in rows_j)
        ev_b = sum(slowdown_onsets(Vb[:, r], t0, min(t1, len_b - 1)) for r in rows_b)
        min_d, arg = _window_min_dist(pos, Xj, Yj, Lj, t0, t1)
        contacts = contact_rows(pos, Xj, Yj, Cj, Lj, t0, t1, WALKER_HIT_DIST_M)
        ev_j_total += ev_j
        ev_b_total += ev_b
        hit_total += len(contacts)
        walker_outputs.append({
            "k": w["k"], "kernel_row": int(w["row"]), "lane_id": w["lane_id"],
            "s": round(w["s"], 2),
            "start_m": [round(v, 2) for v in w["start"]],
            "end_m": [round(v, 2) for v in w["end"]],
            "crossing_width_m": round(w["crossing_width_m"], 2),
            "speed_mps": round(w["speed_mps"], 3),
            "start_step": w["start_step"],
            "completed": w["done_step"] is not None,
            "completed_step": w["done_step"],
            "crossing_duration_s": w["crossing_duration_s"],
            "min_dist_to_vehicle_m": round(min_d, 3) if min_d is not None else None,
            "min_dist_capture_and_row": list(arg) if arg else None,
            "vehicles_in_radius": len(rows_j),
            "slowdown_events_mix": ev_j,
            "slowdown_events_baseline": ev_b,
            "walker_hit_vehicle_rows": contacts,
            "tracking_max_err_m": round(float(w["tracking_max_err_m"]), 4),
        })

    if not cfg["cut_in"]["enable"]:
        cut_out = {"attempted": False, "placed": False, "verdict": "disabled",
                   "reason": "hazard config cut_in.enable: false (default; M2 three-seed "
                             "all abandoned — kernel per-agent IDM is the recorded alternative)",
                   "alternative": CUT_IN_ALTERNATIVE}
    elif cut_in is None:
        cut_out = {"attempted": True, "placed": False, "verdict": "abandoned",
                   "reason": "no straight near-parallel lane pair with a collision-free start",
                   "alternative": CUT_IN_ALTERNATIVE}
    else:
        t0, t1 = cut_in["window"]
        t1 = min(t1, steps_run)
        pos = np.asarray(cut_trail, dtype=float)
        rows_j = _window_rows(pos, Xj, Yj, Lj, t0, t1, WALKER_REACTION_RADIUS_M)
        rows_b = _window_rows(pos, Xb, Yb, Lb, t0, min(t1, len_b - 1), WALKER_REACTION_RADIUS_M)
        ev_j = sum(slowdown_onsets(Vj[:, r], t0, t1) for r in rows_j)
        ev_b = sum(slowdown_onsets(Vb[:, r], t0, min(t1, len_b - 1)) for r in rows_b)
        contacts = contact_rows(pos, Xj, Yj, Cj, Lj, t0, t1, CUT_IN_CONTACT_DIST_M)
        tracked = cut_in["tracking_max_err_m"] <= CUT_IN_MAX_TRACKING_ERR_M
        kept = tracked and ev_j >= 1 and not contacts
        cut_out = {
            "attempted": True, "placed": True,
            "mechanism": "static co-sim slot puppeteered via set_agent_states (same python stepping as walkers)",
            "lane_id": cut_in["lane_id"], "adjacent_lane_id": cut_in["adjacent_lane_id"],
            "lateral_offset_m": round(cut_in["lateral_offset_m"], 2),
            "merge_steps": cut_in["merge_steps"],
            "cruise_mps": round(cut_in["cruise_mps"], 2),
            "window_steps": [t0, t1],
            "slowdown_events_mix": ev_j,
            "slowdown_events_baseline": ev_b,
            "contact_vehicle_rows": contacts,
            "tracking_max_err_m": round(float(cut_in["tracking_max_err_m"]), 4),
            "verdict": "kept" if kept else "abandoned",
            "reason": ("merge completed, followers reacted, no scripted contact"
                       if kept else
                       ("tracking error >%.2f m" % CUT_IN_MAX_TRACKING_ERR_M if not tracked
                        else ("followers produced %d contact(s) during the merge" % len(contacts)
                              if contacts else
                              "0 follower slowdown events during the merge window"))),
            "alternative": None if kept else CUT_IN_ALTERNATIVE,
        }
        ev_j_total += cut_out["slowdown_events_mix"]
        ev_b_total += cut_out["slowdown_events_baseline"]

    interaction = {
        "radius_m": WALKER_REACTION_RADIUS_M,
        "definition": (f"window=[start_step, done_step]; ref = vehicle speed at window start "
                       f"(floor {SLOWDOWN_REF_FLOOR_MPS} m/s); event = >= {SLOWDOWN_SUSTAIN_STEPS} "
                       f"consecutive captures at <= {SLOWDOWN_DROP_FRAC}*ref, onset-counted; "
                       "radius anchored to the scripted walker trajectory in BOTH runs"),
        "walkers_completed": sum(1 for w in walker_outputs if w["completed"]),
        "walkers_total": len(walker_outputs),
        "slowdown_events_baseline_total": ev_b_total,
        "slowdown_events_mix_total": ev_j_total,
        "walker_contact_vehicles_total": hit_total,
        "cut_in_verdict": cut_out["verdict"],
    }

    # ---- comparison artifacts --------------------------------------------------
    missing = []
    for t in COMPARE_KEYFRAMES:
        b_png = baseline_dir / f"env0_t{t:03d}.png"
        h_png = mix_dir / f"env0_t{t:03d}.png"
        if b_png.is_file() and h_png.is_file():
            h_img = add_hazard_callouts(Image.open(h_png), placed,
                                        full_map_world_to_px(map_corners), cjk_font)
            targets = []
            for w in walkers:
                wpos = trails[w["row"]][min(t, len(trails[w["row"]]) - 1)]
                targets.append((f"walker{w['k']} jaywalker", "jaywalker",
                                wpos[0], wpos[1], GIF_COLORS["walker"]))
            if cut_in:
                cpt = cut_trail[min(t, len(cut_trail) - 1)]
                targets.append(("cut-in maker", "cut-in vehicle", cpt[0], cpt[1],
                                HAZARD_CLASS_COLORS["cut_in"]))
            if targets:
                h_img = add_walker_callouts(h_img, targets,
                                            full_map_world_to_px(map_corners), cjk_font)
            side_by_side(b_png, h_img, compare_dir / f"compare_t{t:03d}.png",
                         labels=("baseline", "mix"))
        else:
            missing.append(t)

    zooms = []
    zoom_cone_paths, zoom_walker_paths = [], []
    cone_zone_keys = {e.get("zone_id") for e in placed if e["kind"] == "cone"}
    for key, pts in zone_centers.items():
        ctr = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        zoom_w2p = lambda x, y, ctr=ctr: _world_to_px(
            x, y, ctr, ZOOM_PX / ZOOM_WINDOW_M, ZOOM_PX)
        for t in (0, 40):
            if t not in snap_mix or t not in snap_b:
                continue
            b_img = render_gif_frame(t, snap_b[t][0], snap_b[t][1], road_elements,
                                     traffic_elements, sdc_row_b, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                     args.dt, args.seed, map_bin.stem, mode_label="baseline",
                                     min_cone_px=0, walker_min_px=0)
            m_img = render_gif_frame(t, snap_mix[t][0], snap_mix[t][1], road_elements,
                                     traffic_elements, sdc_row, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                     args.dt, args.seed, map_bin.stem, mode_label="mix",
                                     static_indices=static_indices,
                                     static_class_of=static_class_of,
                                     walker_indices=walker_rows,
                                     walker_trails={r: p[:t + 1] for r, p in trail_arrays.items()},
                                     min_cone_px=0, walker_min_px=0)
            m_img = add_hazard_callouts(m_img, hazards_in_window(placed, ctr, ZOOM_WINDOW_M),
                                        zoom_w2p, cjk_font)
            zoom_name = f"zoom_{key}_t{t:03d}.png"
            side_by_side(b_img, m_img, compare_dir / zoom_name, labels=("baseline", "mix"))
            zooms.append(f"compare/{zoom_name}")
            if key in cone_zone_keys:
                zoom_cone_paths.append(f"compare/{zoom_name}")
    for w in walkers:
        t = w["mid_step"]
        if t not in snap_mix or t not in snap_b:
            missing.append(f"zoom_walker{w['k']}_t{t:03d}")
            continue
        ctr = w["cross_pt"]
        zoom_w2p = lambda x, y, ctr=ctr: _world_to_px(x, y, ctr, ZOOM_PX / ZOOM_WINDOW_M, ZOOM_PX)
        b_img = render_gif_frame(t, snap_b[t][0], snap_b[t][1], road_elements,
                                 traffic_elements, sdc_row_b, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                 args.dt, args.seed, map_bin.stem, mode_label="baseline",
                                 min_cone_px=0, walker_min_px=0)
        m_img = render_gif_frame(t, snap_mix[t][0], snap_mix[t][1], road_elements,
                                 traffic_elements, sdc_row, ctr, ZOOM_WINDOW_M, ZOOM_PX,
                                 args.dt, args.seed, map_bin.stem, mode_label="mix",
                                 static_indices=static_indices, static_class_of=static_class_of,
                                 walker_indices=walker_rows,
                                 walker_trails={r: p[:t + 1] for r, p in trail_arrays.items()},
                                 min_cone_px=0, walker_min_px=0)
        wpos = trails[w["row"]][min(t, len(trails[w["row"]]) - 1)]
        targets = [(f"walker{w['k']} jaywalker", "jaywalker", wpos[0], wpos[1],
                    GIF_COLORS["walker"])]
        m_img = add_walker_callouts(m_img, targets, zoom_w2p, cjk_font)
        zoom_name = f"zoom_walker{w['k']}_t{t:03d}.png"
        side_by_side(b_img, m_img, compare_dir / zoom_name, labels=("baseline", "mix"))
        zooms.append(f"compare/{zoom_name}")
        zoom_walker_paths.append(f"compare/{zoom_name}")

    trails_path = None
    if walkers:
        pil = render_full_map(steps_run, af0, ai0, road_elements, traffic_elements, sdc_row,
                              map_corners, args.dt, args.seed, map_bin.stem, "mix",
                              static_indices=static_indices, static_class_of=static_class_of,
                              walker_indices=walker_rows, walker_trails=trail_arrays)
        pil = add_trail_annotations(pil, walkers, trail_arrays,
                                    full_map_world_to_px(map_corners), cut_in, cut_trail, cjk_font)
        pil.save(seed_dir / "trails.png")
        trails_path = "trails.png"

    # pixel-existence audits: cone-orange and walker-magenta pixels in every
    # image that should show them; any 0 -> hard fail (M1/M2 mechanism)
    cone_audit_paths = (list(keyframe_paths)
                        + [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                           if t not in missing]
                        + zoom_cone_paths) if cone_zone_keys else []
    cone_audit = audit_pixels(seed_dir, cone_audit_paths, CONE_RGB, "cone") \
        if cone_audit_paths else {}
    walker_audit_paths = (list(keyframe_paths)
                          + [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                             if t not in missing]
                          + zoom_walker_paths
                          + (["trails.png"] if trails_path else [])) if walkers else []
    walker_audit = audit_pixels(seed_dir, walker_audit_paths, WALKER_RGB, "walker") \
        if walker_audit_paths else {}
    print(f"[hazard-demo] pixel audits: cones={len(cone_audit)} walker={len(walker_audit)} images")

    notes = [
        "边界声明（M3）：本配置面是原型级（Python 侧）——旋钮只驱动 demo 的 co-sim 放置逻辑；"
        "真正的内核子系统（C reset 内生放置、PEDESTRIAN type、per-agent IDM）属下一阶段规格票，本票零内核改动。",
        "配置来源 " + str(cfg.get("_path")) + "，init 前经 validate_hazard_config fail-fast 校验"
        "（范围/类型/概率界外立即报错，校验器独立可测：tests/test_hazard_demo.py）。",
        "回归契约：enable: false（全部类型禁用）→ 0 个 co-sim 静态槽 + 零放置 → episode 与纯 gigaflow "
        "baseline 逐位一致（小规模逐位测试在 tests/test_hazard_demo.py；全规模双门在 M3 交付说明）。",
        "static-hazards / jaywalker 模式未动：放置函数新参数全部默认回退到冻结的 M1/M2 常量，"
        "同 seed 数字逐位不变。",
        "mix 机制：静态三类一次性放置（M1 路径），walker/cut-in 每步 set_agent_states 步进（M2 路径），"
        "两类共用 co-sim 静态槽；set_agent_states 只写 vec->envs[0] → 强制 --envs 1。",
        "mix 槽容量由配置上界推导（mix_slot_capacity）：parked 上界 + 2×cluster 上界 + "
        "zones 上界×每区锥数上界 + walker 上界 + cut_in 1；未用槽在内核侧 sim_valid=0，不参与碰撞/观测。",
        "walker 在 mix 中额外拒绝与已放置静态 hazard 重叠（plan_walkers static_placed 参数）；"
        "jaywalker 模式不传该参数，行为与 M2 逐位一致。",
        "交互度量口径同 M2：横穿窗口 = [start_step, done_step]；反应半径 15 m；"
        "明显减速 ref = 窗口首步速度（≥2 m/s 底线），连续 ≥3 步 ≤0.5*ref，onset 去重；"
        "接触 = 窗口内 3 m 内 latch 碰撞车辆；cut-in 接触距离 6 m。",
        "渲染遵循长期标准：单独 legend.png（index 首项，mix 图例仅列出本 run 实际 spawn 的类别——"
        "cut_in.enable=false 时不列 cut-in 行）；类别着色；锥桶/行人全图强制 ≥12px 标记；"
        "compare/zoom 有文字+箭头指认；CJK 缺字体时英文-only（无豆腐块）；"
        "锥桶/行人像素审计任何一张为 0 即硬失败。",
        "kernel_lane_assoc_failures 口径同 M1：贴边静态槽在内核侧 current_lane_idx=-1 属在道证据缺失，"
        "无 gameplay 影响，如实记录不修正；walker 刻意起点在路缘外，不进该检查。",
    ]
    seed_summary = {
        "mode": "mix",
        "kernel": kernel_provenance(),
        "hazard_config": {k: (v if k != "_path" else v) for k, v in cfg.items()},
        "seed": args.seed,
        "map": {"name": map_bin.stem, "path": str(map_bin)},
        "agents_per_env": args.agents_per_env,
        "steps_requested": args.steps,
        "dt": args.dt,
        "placement": {
            "parked": {"planned": plan["parked"],
                       "placed": sum(1 for e in placed if e["kind"] == "parked")},
            "collided": {"planned": plan["clusters"],
                         "placed": len({e.get("cluster_id") for e in placed
                                        if e.get("cluster_id")})},
            "cones": {"planned": plan["cone_zones"],
                      "placed": len({e.get("zone_id") for e in placed
                                     if e.get("zone_id")}),
                      "cones_placed": sum(1 for e in placed if e["kind"] == "cone")},
            "walkers": {"planned": walkers_planned, "placed": len(walkers)},
            "reject_reasons": {**{f"static:{k}": v for k, v in rejects.items() if v},
                               **{f"walker:{k}": v for k, v in walker_rejects.items() if v}},
            "slots_used": len(initial),
            "slots_reserved": slots,
        },
        "legality": legality,
        "walkers": walker_outputs,
        "cut_in": cut_out,
        "interaction": interaction,
        "simulation": {
            "window_step": min(steps_run, len_b - 1),
            "collision_agents_baseline_final": baseline_metrics["collision_agents_final"],
            "collision_agents_mix_final": mix_metrics["collision_agents_final"],
            "baseline": baseline_metrics,
            "mix": mix_metrics,
        },
        "render_audit": {
            "cone_color_rgb": list(CONE_RGB),
            "walker_color_rgb": list(WALKER_RGB),
            "cone_pixel_counts": cone_audit,
            "walker_pixel_counts": walker_audit,
            "images_checked": len(cone_audit) + len(walker_audit),
            "all_nonzero": True,
        },
        "artifacts": {
            "baseline_keyframes": baseline_keyframes,
            "mix_keyframes": keyframe_paths,
            "mix_gif": "mix/mix.gif" if gif_imgs else None,
            "compare": [f"compare/compare_t{t:03d}.png" for t in COMPARE_KEYFRAMES
                        if t not in missing],
            "zooms": zooms,
            "trails_overlay": trails_path,
        },
        "missing_artifacts": missing,
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
    write_mix_root(root, seed_summaries, cjk_path=cjk_font)
    comp = sum(1 for w in walker_outputs if w["completed"])
    print(f"[hazard-demo] artifacts -> {root} ({seed_summary['runtime_s']}s); "
          f"crossed {comp}/{len(walker_outputs)}, slowdown events "
          f"baseline={ev_b_total} vs mix={ev_j_total}, contacts={hit_total}, "
          f"collisions baseline={baseline_metrics['collision_agents_final']} vs "
          f"mix={mix_metrics['collision_agents_final']}, cut_in={cut_out['verdict']}")


def write_mix_counts_histogram(root, seed_summaries):
    """Per-seed placed counts for the four mix classes."""
    import matplotlib.pyplot as plt

    seed_summaries = sorted(seed_summaries, key=lambda s: s["seed"])
    classes = ("parked", "collided clusters", "cones", "walkers")
    counts = {c: [] for c in classes}
    for s in seed_summaries:
        p = s["placement"]
        counts["parked"].append(p["parked"]["placed"])
        counts["collided clusters"].append(p["collided"]["placed"])
        counts["cones"].append(p["cones"]["cones_placed"])
        counts["walkers"].append(p["walkers"]["placed"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(classes))
    width = 0.8 / max(len(seed_summaries), 1)
    for i, s in enumerate(seed_summaries):
        vals = [counts[c][i] for c in classes]
        ax.bar(xpos + i * width, vals, width, label=f"seed {s['seed']} (placed)")
    ax.set_xticks(xpos + width * (len(seed_summaries) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("count per env")
    ax.set_title("M3 mix placement counts (config-driven, per seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "counts_histogram.png", dpi=140)
    plt.close(fig)


def write_mix_root(root, seed_summaries, cjk_path=None):
    """Aggregate legend.png / counts_histogram.png / summary.json / index.html
    over mix seed dirs. The root legend lists only classes that spawned in at
    least one seed (owner rule: never list un-spawned categories)."""
    spawned = set()
    for s in seed_summaries:
        p = s["placement"]
        if p["parked"]["placed"]:
            spawned.add("parked")
        if p["collided"]["placed"]:
            spawned.add("collided")
        if p["cones"]["placed"]:
            spawned.add("cone")
        if p["walkers"]["placed"]:
            spawned.add("walker")
        if s.get("cut_in", {}).get("placed"):
            spawned.add("cut_in")
    render_legend_png(root / "legend.png", "mix", cjk_path=cjk_path,
                      spawned=spawned or None)
    all_w = [w for s in seed_summaries for s_walkers in [s.get("walkers", [])] for w in s_walkers]
    mind = [w["min_dist_to_vehicle_m"] for w in all_w
            if w.get("min_dist_to_vehicle_m") is not None]
    completed = sum(1 for w in all_w if w.get("completed"))
    mix_coll = sum(s["simulation"]["mix"]["collision_agents_final"] for s in seed_summaries)
    base_coll = sum(s["simulation"]["baseline"]["collision_agents_final"] for s in seed_summaries)
    mix_off = sum(s["simulation"]["mix"]["offroad_agents_final"] for s in seed_summaries)
    agg = {
        "mode": "mix",
        "kernel": kernel_provenance(),
        "hazard_config": seed_summaries[0].get("hazard_config") if seed_summaries else None,
        "seeds": [s["seed"] for s in seed_summaries],
        "aggregate": {
            "parked_placed_total": sum(s["placement"]["parked"]["placed"] for s in seed_summaries),
            "clusters_placed_total": sum(s["placement"]["collided"]["placed"] for s in seed_summaries),
            "cones_placed_total": sum(s["placement"]["cones"]["cones_placed"] for s in seed_summaries),
            "walkers_placed_total": sum(s["placement"]["walkers"]["placed"] for s in seed_summaries),
            "walkers_completed_total": completed,
            "completion_rate": round(completed / len(all_w), 3) if all_w else None,
            "walker_min_dist_m": ({"min": round(float(np.min(mind)), 3),
                                   "p50": round(float(np.percentile(mind, 50)), 3),
                                   "p95": round(float(np.percentile(mind, 95)), 3)}
                                  if mind else None),
            "slowdown_events_baseline_total": sum(
                s["interaction"]["slowdown_events_baseline_total"] for s in seed_summaries),
            "slowdown_events_mix_total": sum(
                s["interaction"]["slowdown_events_mix_total"] for s in seed_summaries),
            "walker_contact_vehicles_total": sum(
                s["interaction"]["walker_contact_vehicles_total"] for s in seed_summaries),
            "collision_agents_total": {"baseline": base_coll, "mix": mix_coll},
            "offroad_agents_mix_total": mix_off,
            "dynamic_overlap_violations_total": sum(
                s["legality"]["all_slots_dynamic_overlap_postcheck"] for s in seed_summaries),
            "cut_in_verdicts": {str(s["seed"]): s["cut_in"].get("verdict")
                                for s in sorted(seed_summaries, key=lambda x: x["seed"])},
        },
        "per_seed": [{
            "seed": s["seed"],
            "placement": {k: s["placement"][k] for k in
                          ("parked", "collided", "cones", "walkers", "slots_used")},
            "legality": {k: s["legality"][k] for k in
                         ("kernel_lane_assoc_failures",
                          "dynamic_overlap_violations_postcheck",
                          "intentional_intra_cluster_overlaps",
                          "all_slots_dynamic_overlap_postcheck")},
            "interaction": {k: s["interaction"][k] for k in
                            ("walkers_completed", "walkers_total",
                             "slowdown_events_baseline_total", "slowdown_events_mix_total",
                             "walker_contact_vehicles_total", "cut_in_verdict")},
            "simulation": {"baseline": s["simulation"]["baseline"],
                           "mix": s["simulation"]["mix"]},
            "render_audit": {"images_checked": s["render_audit"]["images_checked"],
                             "all_nonzero": s["render_audit"]["all_nonzero"]},
        } for s in sorted(seed_summaries, key=lambda x: x["seed"])],
        "notes": (seed_summaries[-1].get("notes") if seed_summaries else []),
    }
    (root / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not seed_summaries:
        return
    try:
        write_mix_counts_histogram(root, seed_summaries)
    except Exception as exc:  # chart is cosmetic; never sink the run
        print(f"[hazard-demo] mix counts histogram skipped: {exc}")

    rows = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        p, inter, sim = s["placement"], s["interaction"], s["simulation"]
        cells = (f"<td>{p['parked']['placed']}</td><td>{p['collided']['placed']}</td>"
                 f"<td>{p['cones']['placed']} ({p['cones']['cones_placed']} cones)</td>"
                 f"<td>{p['walkers']['placed']}</td>"
                 f"<td>{inter['walkers_completed']}/{inter['walkers_total']}</td>"
                 f"<td>{inter['slowdown_events_baseline_total']} → "
                 f"{inter['slowdown_events_mix_total']}</td>"
                 f"<td>{inter['walker_contact_vehicles_total']}</td>"
                 f"<td>{sim['baseline']['collision_agents_final']} → "
                 f"{sim['mix']['collision_agents_final']}</td>"
                 f"<td>{sim['mix']['offroad_agents_final']}</td>"
                 f"<td>{inter['cut_in_verdict']}</td>")
        rows += f"<tr><td>seed {s['seed']}</td>{cells}</tr>"
    galleries = ""
    for s in sorted(seed_summaries, key=lambda x: x["seed"]):
        sd = f"seed{s['seed']}"
        frames = "".join(
            f'<figure><img src="{sd}/mix/frames/{p.name}"/>'
            f"<figcaption>{p.name}</figcaption></figure>"
            for p in sorted((root / sd / "mix" / "frames").glob("*.png")))
        compares = "".join(
            f'<figure><img src="{sd}/compare/{p.name}"/>'
            f"<figcaption>{p.name}（左 baseline / 右 mix）</figcaption></figure>"
            for p in sorted((root / sd / "compare").glob("*.png")))
        trails_img = (f'<figure><img src="{sd}/trails.png"/>'
                      f"<figcaption>trails.png — 最终帧叠加全程轨迹</figcaption></figure>"
                      if (root / sd / "trails.png").is_file() else "")
        gif = (f'<img src="{sd}/mix/mix.gif" width="560"/>'
               if (root / sd / "mix" / "mix.gif").is_file() else "")
        galleries += (f"<h2>seed {s['seed']}</h2>"
                      f"<h3>mix 全图帧</h3>{frames}"
                      f"<h3>compare / 事件 zoom</h3>{compares}"
                      f"<h3>轨迹叠加（最终帧 + 全程 trail）</h3>{trails_img}"
                      f"<h3>跟车 GIF</h3>{gif}\n")
    a = agg["aggregate"]
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>Slipstream hazard-demo · M3 mix</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  h1, h2, h3 {{ color: #333; }} table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
  th {{ background: #eee; }} figure {{ margin: 8px; display: inline-block; }}
  img {{ border: 1px solid #ddd; max-width: 640px; }} figcaption {{ font-size: 12px; color: #666; }}
  code {{ background: #eee; padding: 2px 6px; }}
</style></head><body>
  <h1>M3 mix · 配置驱动的全类型同场演示</h1>
  <p>命令: <code>PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/hazard_demo.py --mode mix --seed &lt;42|43|44&gt;</code>
     （配置: <code>{(agg['hazard_config'] or {}).get('_path', 'config/hazard_demo.yaml')}</code>）</p>
  <h2>图例（legend.png 单独成图，index 首项；仅列实际 spawn 的类别）</h2>
  <img src="legend.png" width="520"/>
  <img src="counts_histogram.png" width="680"/>
  <h2>汇总</h2>
  <ul>
    <li>parked {a['parked_placed_total']} · 碰撞簇 {a['clusters_placed_total']} · 锥桶 {a['cones_placed_total']} ·
        walker {a['walkers_placed_total']}（完成横穿 {a['walkers_completed_total']}，"
        完成率 {a['completion_rate']}）。</li>
    <li>减速事件 baseline {a['slowdown_events_baseline_total']} vs mix {a['slowdown_events_mix_total']}；
        walker 接触车辆 {a['walker_contact_vehicles_total']}；碰撞 baseline {a['collision_agents_total']['baseline']}
        vs mix {a['collision_agents_total']['mix']}；mix offroad {a['offroad_agents_mix_total']}；
        非有意动态重叠 {a['dynamic_overlap_violations_total']}。</li>
    <li>边界声明：配置面为原型级（Python 侧旋钮驱动 demo 放置）；内核子系统（C reset 内生放置、
        PEDESTRIAN type、per-agent IDM）属下一阶段规格票，本票零内核改动。</li>
    <li>回归契约：enable: false → 与纯 gigaflow baseline 逐位一致。</li>
    <li>像素审计：summary.json render_audit，锥桶/行人任一应含像素图为 0 即硬失败。</li>
  </ul>
  <h2>每 seed 度量</h2>
  <table><tr><th>seed</th><th>parked</th><th>碰撞簇</th><th>锥桶区</th><th>walker</th><th>完成横穿</th>
  <th>减速事件 baseline → mix</th><th>walker 接触</th><th>碰撞 baseline → mix</th><th>mix offroad</th><th>cut-in</th></tr>
  {rows}</table>
  {galleries}
  <h2>口径 / notes</h2><ul>{''.join(f'<li>{n}</li>' for n in agg['notes'])}</ul>
</body></html>"""
    (root / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    verify_in_repo_kernel()
    if args.placement == "kernel" and args.mode not in ("static-hazards", "jaywalker"):
        raise SystemExit("--placement kernel only applies to --mode static-hazards or --mode jaywalker")
    if args.out is None:
        out_default = {"baseline": "artifacts/baseline",
                       "static-hazards": "artifacts/static-hazards",
                       "jaywalker": "artifacts/jaywalker",
                       "mix": "artifacts/mix"}[args.mode]
        if args.mode == "static-hazards" and args.placement == "kernel":
            out_default = "artifacts/kernel-static"
        if args.mode == "jaywalker" and args.placement == "kernel":
            out_default = "artifacts/kernel-jaywalker"
        args.out = out_default
    if args.hazard_config is not None and args.mode != "mix" \
            and not (args.mode == "static-hazards" and args.placement == "kernel") \
            and not (args.mode == "jaywalker" and args.placement == "kernel"):
        raise SystemExit("--hazard-config only applies to --mode mix (or kernel placement)")
    if args.mode == "mix":
        out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        map_bin = resolve_map_bin(args.map_dir, args.map)
        cfg = load_hazard_config(args.hazard_config)  # init 前校验, fail-fast
        run_mix(args, map_bin, cfg)
        return
    if args.mode == "static-hazards":
        out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        map_bin = resolve_map_bin(args.map_dir, args.map)
        run_static_hazards(args, map_bin)
        return
    if args.mode == "jaywalker":
        out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        map_bin = resolve_map_bin(args.map_dir, args.map)
        run_jaywalker(args, map_bin)
        return
    t_start = time.time()
    out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    map_bin = resolve_map_bin(args.map_dir, args.map)
    print(f"[hazard-demo] mode={args.mode} seed={args.seed} map={map_bin}")
    cjk_font = detect_cjk_font()
    print(f"[hazard-demo] CJK font: {cjk_font or 'none -> English-only labels (no tofu)'}")

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
            pil = render_full_map(step, buffers["agent_f32"][e], buffers["agent_i32"][e],
                                  road_elements, traffic_elements, sdc_rows[e],
                                  map_corners, args.dt, args.seed, map_bin.stem, "baseline")
            path = frames_dir / f"env{e}_t{step:03d}.png"
            pil.save(path)
            keyframe_paths[e].append(str(path.relative_to(out_dir)))

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
    render_legend_png(out_dir / "legend.png", "baseline", cjk_path=cjk_font)
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
        "kernel": kernel_provenance(),
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
