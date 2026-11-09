#!/usr/bin/env python3
"""K3 enforcement-knob evidence runner (scripts/hazard_demo.py machinery, one config per run).

Produces the K3 numeric comparison artifact set under ``artifacts/k3-patches/evidence/``:
a legacy (all knobs at commit defaults) vs each knob turned on table on the same seed —
latched red-light / stop-sign / collision / offroad agent counts, episode-flow divergence
(first pose-stream divergence + latched-set diffs), truth-timing fallback disclosure for
geometry-only CARLA bins, and true-scale zoom frames for footprint-sweep-only red-light
catches ("擦边越线"), rendered by the long-term standard renderer (shared pillow renderer,
CJK font, standalone legend pointer).

Knob face (pufferlib/ocean/drive, K3 commit):
  - traffic_light_truth_timing (K4a default off): bin-provided signal programs in gigaflow;
    inert on geometry-only bins (CARLA Town05: 54 lights, 0 with bin states) — every light
    takes the documented procedural fallback.
  - traffic_light_footprint_sweep (K4a default ON): red light = swept full footprint vs the
    stop bar (begin-contact edge trigger) instead of the legacy centre-point crossing.
  - stop_sign_dwell_range (K4a default [0.5,1.5]): per-sign sampled hold-below-speed dwell in
    seconds; [0,0] = legacy single-tick completion. Sampling consumes env RNG, so a live
    range shifts the episode stream by design.
  - pedestrian_traffic_exempt (K4a default on): pedestrians skip the road-exit/red-light/
    stop-sign entries. Inert today (no active PEDESTRIAN on gigaflow; K2 jaywalkers are
    static slots) — C unit tests tests/drive/test_drive_pedestrian_exempt.c carry the
    per-class evidence; no run numbers exist or are needed.

Since K4a the python defaults carry the new semantics, so every config below pins the
pre-K4a legacy face explicitly (LEGACY_KNOBS) and layers one knob on top — the table stays
a knob-isolation comparison regardless of drive.py defaults.

Run: PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/k3_knob_evidence.py --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hazard_demo as hd  # noqa: E402  (repo scripts dir; kernel pinning enforced below)

# metrics_array layout: pufferlib/ocean/drive/constants.h (COLLISION_IDX 0 / OFFROAD_IDX 1 /
# RED_LIGHT_IDX 2 / STOP_SIGN_IDX 3); the demo pins the first two, K3 adds the right-of-way pair.
COLLISION_IDX = 0
OFFROAD_IDX = 1
RED_LIGHT_IDX = 2
STOP_SIGN_IDX = 3

ZOOM_WINDOW_M = 90.0  # hazard_demo ZOOM_WINDOW_M: true-scale zoom renders
ZOOM_PX = 900

# (name, extra Drive kwargs, one-line semantics for the report table)
# Pre-K4a legacy face, pinned explicitly: K4a flipped the python defaults, so an empty
# kwargs dict no longer means legacy. Every config inherits these pins and layers its delta.
LEGACY_KNOBS = {
    "traffic_light_truth_timing": False,
    "traffic_light_footprint_sweep": False,
    "stop_sign_dwell_range": (0.0, 0.0),
}

K3_CONFIGS = (
    ("legacy", dict(LEGACY_KNOBS),
     "pre-K4a face: truth off / sweep off / dwell [0,0] / ped exempt on (inert)"),
    ("truth-timing", {**LEGACY_KNOBS, "traffic_light_truth_timing": True},
     "bin-provided signal programs in gigaflow; no-op here — CARLA bins carry no states"),
    ("footprint-sweep", {**LEGACY_KNOBS, "traffic_light_footprint_sweep": True},
     "red light = swept footprint vs stop bar (begin-contact edge trigger)"),
    ("dwell-0.5-1.5", {**LEGACY_KNOBS, "stop_sign_dwell_range": (0.5, 1.5)},
     "per-sign sampled dwell 0.5-1.5 s (the K4a default range; consumes RNG by design)"),
    ("all-on", {**LEGACY_KNOBS, "traffic_light_truth_timing": True,
                "traffic_light_footprint_sweep": True,
                "stop_sign_dwell_range": (0.5, 1.5)},
     "all three active knobs together (ped gate stays at its inert default-on)"),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.3)
    p.add_argument("--goal-source", choices=["route", "map"], default="route")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--agents-per-env", type=int, default=150)
    p.add_argument("--map-dir", default="../PufferDrive/pufferlib/resources/drive/binaries/carla")
    p.add_argument("--map", default="Town05")
    p.add_argument("--out", default="artifacts/k3-patches/evidence")
    return p.parse_args()


def latched_set(mask, metrics, idx):
    return frozenset(np.flatnonzero(mask & (metrics[:, idx] > 0)).tolist())


def run_config(name, knobs, args, map_bin, keep_frames=False):
    """Step one env; return metrics + per-step pose stream + red-light latch events.

    keep_frames archives the full html-frame buffers at every red-light latch step
    (used by the legacy/sweep runs so zoom pairs can render the same step from both)."""
    env = hd.build_env(args, map_bin, extra_env_kwargs=dict(knobs))
    env.reset(seed=args.seed)
    scenarios = env.get_state()
    buffers = hd.capture_buffers(env, scenarios)
    road_elements = scenarios[0].get("road_elements") or []
    traffic_elements = scenarios[0].get("traffic_elements") or []
    sdc_row = int((scenarios[0].get("active_agent_indices") or [0])[0])

    def capture():
        env.get_obs_html_frame(
            buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
            buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
            buffers["rewards_f32"], buffers["coefs_f32"],
        )

    capture()
    af0, ai0, mf0 = buffers["agent_f32"][0], buffers["agent_i32"][0], buffers["metrics_f32"][0]

    pose_stream = []       # per step: (live x, live y) float32 snapshot for divergence diff
    archived = {}          # latch step -> (af copy, ai copy), only with keep_frames
    fire_steps = {k: None for k in ("collision", "offroad", "red_light", "stop_sign")}
    fire_idx = {k: () for k in fire_steps}
    red_events = []        # (step, [agent_idx], af copy, ai copy) on each red-light latch transition
    prev_red = latched_set(hd.env_active_mask(ai0), mf0, RED_LIGHT_IDX)
    steps_run = 0
    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    for step in range(1, args.steps + 1):
        env.step(actions)
        steps_run = step
        capture()
        mask = hd.env_active_mask(ai0)
        pose_stream.append(af0[mask, :2].copy())
        for key, idx in (("collision", COLLISION_IDX), ("offroad", OFFROAD_IDX),
                         ("red_light", RED_LIGHT_IDX), ("stop_sign", STOP_SIGN_IDX)):
            cur = latched_set(mask, mf0, idx)
            if fire_steps[key] is None and cur:
                fire_steps[key] = step
                fire_idx[key] = tuple(sorted(cur))
            if key == "red_light":
                new = cur - prev_red
                if new:
                    red_events.append((step, sorted(new), af0.copy(), ai0.copy()))
                    if keep_frames:
                        archived[step] = (af0.copy(), ai0.copy())
                prev_red = cur
    out = {
        "steps_run": steps_run,
        "fire_steps": fire_steps,
        "fire_idx": fire_idx,
        "red_events": red_events,
        "archived": archived,
        "pose_stream": pose_stream,
        "final": {
            "collision_agents": sorted(latched_set(hd.env_active_mask(ai0), mf0, COLLISION_IDX)),
            "offroad_agents": sorted(latched_set(hd.env_active_mask(ai0), mf0, OFFROAD_IDX)),
            "red_light_agents": sorted(latched_set(hd.env_active_mask(ai0), mf0, RED_LIGHT_IDX)),
            "stop_sign_agents": sorted(latched_set(hd.env_active_mask(ai0), mf0, STOP_SIGN_IDX)),
            "active_final": int(np.sum(hd.env_active_mask(ai0))),
        },
        "road_elements": road_elements,
        "traffic_elements": traffic_elements,
        "sdc_row": sdc_row,
    }
    env.close()
    return out


def first_divergence(a, b):
    """First step where the live pose streams differ (episode-flow divergence)."""
    for step, (pa, pb) in enumerate(zip(a, b), start=1):
        if pa.shape != pb.shape or not np.array_equal(pa, pb):
            return step
    return None


def main():
    args = parse_args()
    hd.verify_in_repo_kernel()
    out_dir = (REPO_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_bin = hd.resolve_map_bin(args.map_dir, args.map)
    cjk = hd.detect_cjk_font()
    print(f"[k3-evidence] map={map_bin} seed={args.seed} steps={args.steps} "
          f"envs={args.envs}x{args.agents_per_env} cjk={'yes' if cjk else 'no'}")

    runs = {}
    for name, knobs, _ in K3_CONFIGS:
        print(f"[k3-evidence] run {name} knobs={knobs or '{}'}")
        runs[name] = run_config(name, knobs, args, map_bin,
                                keep_frames=name in ("legacy", "footprint-sweep"))
        print(f"[k3-evidence]   {name}: steps={runs[name]['steps_run']} "
              f"red={len(runs[name]['final']['red_light_agents'])} "
              f"stop={len(runs[name]['final']['stop_sign_agents'])} "
              f"coll={len(runs[name]['final']['collision_agents'])}")

    legacy = runs["legacy"]
    table = []
    for name, knobs, semantics in K3_CONFIGS:
        run = runs[name]
        row = {
            "name": name,
            "knobs": {k: list(v) if isinstance(v, tuple) else v for k, v in knobs.items()},
            "semantics": semantics,
            "steps_run": run["steps_run"],
            "counts": {k: len(v) for k, v in run["final"].items() if isinstance(v, list)},
            "agents": run["final"],
            "first_fire_step": run["fire_steps"],
            "first_pose_divergence_vs_legacy": (
                None if name == "legacy" else first_divergence(legacy["pose_stream"], run["pose_stream"])
            ),
            "latched_diff_vs_legacy": {
                key: {
                    "new": sorted(set(run["final"][key]) - set(legacy["final"][key])),
                    "gone": sorted(set(legacy["final"][key]) - set(run["final"][key])),
                }
                for key in ("collision_agents", "offroad_agents", "red_light_agents", "stop_sign_agents")
            },
        }
        table.append(row)

    # footprint-sweep zoom pair: an agent the swept-footprint rule catches that the legacy
    # centre-point rule never latched, at the step its red-light latch fires; the legacy
    # frame at the same step shows the same crossing going unflagged under the old rule
    zooms = []
    for step, idxs, af, ai in runs["footprint-sweep"]["red_events"]:
        for idx in idxs:
            if idx in legacy["final"]["red_light_agents"]:
                continue
            zooms.append((step, idx, af, ai))
    zoom_files = []
    for step, idx, af, ai in zooms[:2]:  # keep it to the first two catches
        x, y = float(af[idx, 0]), float(af[idx, 1])
        sweep_img = hd.render_gif_frame(step, af, ai, runs["footprint-sweep"]["road_elements"],
                                        runs["footprint-sweep"]["traffic_elements"],
                                        runs["footprint-sweep"]["sdc_row"], (x, y), ZOOM_WINDOW_M,
                                        ZOOM_PX, args.dt, args.seed, map_bin.stem,
                                        mode_label="kernel-sweep-zoom", min_cone_px=0)
        if legacy["archived"].get(step) is not None:
            laf, lai = legacy["archived"][step]
            legacy_img = hd.render_gif_frame(step, laf, lai, legacy["road_elements"],
                                             legacy["traffic_elements"], legacy["sdc_row"],
                                             (x, y), ZOOM_WINDOW_M, ZOOM_PX, args.dt, args.seed,
                                             map_bin.stem, mode_label="legacy-zoom", min_cone_px=0)
            p_sweep = out_dir / f"zoom_footprint_agent{idx}_t{step:03d}_sweep.png"
            p_leg = out_dir / f"zoom_footprint_agent{idx}_t{step:03d}_legacy.png"
            sweep_img.save(p_sweep)
            legacy_img.save(p_leg)
            zoom_files.append({"agent_idx": idx, "step": step,
                               "frames": [str(p_sweep.name), str(p_leg.name)],
                               "caption": f"agent {idx}: sweep latch t={step} (左) vs 同步 legacy 帧未旗标 (右)"})
        else:
            path = out_dir / f"zoom_footprint_agent{idx}_t{step:03d}_sweeponly.png"
            sweep_img.save(path)
            zoom_files.append({"agent_idx": idx, "step": step, "frames": [str(path.name)],
                               "caption": f"agent {idx}: swept footprint latched red at t={step}; "
                                          f"legacy centre-point rule never latched this agent"})
    if not zoom_files:
        print("[k3-evidence] no sweep-only catch on this seed/map: no zoom pair (honest zero)")

    summary = {
        "kernel": hd.kernel_provenance(),
        "command": f"PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python scripts/k3_knob_evidence.py "
                   f"--seed {args.seed} --steps {args.steps} --dt {args.dt} --map {args.map}",
        "map": str(map_bin),
        "seed": args.seed,
        "steps_requested": args.steps,
        "dt": args.dt,
        "envs": f"{args.envs} x {args.agents_per_env}",
        "table": table,
        "zooms": zoom_files,
        "notes": [
            "truth timing on CARLA Town05: the bin carries 54 lights / 0 with stored states, so every "
            "light takes the documented procedural fallback (truth_timing_fallback_lights); replay "
            "semantics are pinned by C tests tests/drive/test_drive_traffic_lights.c "
            "(test_truth_timing_preserves_bin_states / _off_overwrites_bin_states / _off_matches_bin_free_env "
            "/ _fallback_counts_geometry_only_light). Numbers identical to legacy here are the expected no-op.",
            "footprint sweep keeps the env RNG stream untouched: divergence starts at the first "
            "sweep-only red-light latch (behavior=stop cascade), visible in first_pose_divergence_vs_legacy.",
            "dwell sampling consumes env RNG (one uniform per stop sign at reset, 17 signs on Town05), "
            "so a live range shifts the spawn stream by design; the bit-identity contract is an "
            "all-defaults contract (tests/drive/test_drive_pedestrian_exempt.c iron-rule test).",
            "pedestrian gate is inert on this map (no active PEDESTRIAN; K2 jaywalkers are static slots) — "
            "per-class evidence lives in tests/drive/test_drive_pedestrian_exempt.c, not in run numbers.",
            "counts are latched metrics_array flags over the final live agent mask (demo口径); "
            "latched_diff_vs_legacy lists agent indices newly/gone flagged per category.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = ""
    for row in table:
        c = row["counts"]
        div = row["first_pose_divergence_vs_legacy"]
        rows += (f"<tr><td>{row['name']}</td><td>{row['semantics']}</td>"
                 f"<td>{c['red_light_agents']}</td><td>{c['stop_sign_agents']}</td>"
                 f"<td>{c['collision_agents']}</td><td>{c['offroad_agents']}</td>"
                 f"<td>{row['steps_run']}</td><td>{'—' if div is None else div}</td></tr>\n")
    zoom_html = ""
    for z in zoom_files:
        imgs = "".join(f'<figure><img src="{f}"/><figcaption>{f}</figcaption></figure>' for f in z["frames"])
        zoom_html += f"  <h2>footprint-sweep zoom · agent {z['agent_idx']} @ t={z['step']}</h2>\n{imgs}\n<p>{z['caption']}</p>\n"
    if not zoom_html:
        zoom_html = "  <p>无 sweep-only 擦边捕获（诚实零）——本 seed/map 上 footprint 规则与中心点规则没有分歧捕获。</p>\n"
    notes = "".join(f"      <li>{n}</li>\n" for n in summary["notes"])
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Slipstream K3 enforcement knobs · legacy vs on (seed {args.seed})</title>
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
  <h1>K3 执法旋钮对照 · legacy vs on · seed {args.seed}</h1>
  <p>命令: <code>{summary['command']}</code></p>
  <h2>kernel 溯源</h2>
  <ul>
    <li>pufferlib: <code>{summary['kernel']['pufferlib_path']}</code></li>
    <li>binding .so: <code>{summary['kernel']['binding_so']}</code></li>
    <li>.so sha256: <code>{summary['kernel']['binding_sha256']}</code></li>
    <li>git sha: <code>{summary['kernel']['git_sha']}</code> · 构建: <code>{summary['kernel']['built_by']}</code></li>
  </ul>
  <h2>数字对照（latched agents, demo 口径）</h2>
  <table>
    <tr><th>config</th><th>语义</th><th>红灯</th><th>停牌</th><th>碰撞</th><th>offroad</th>
        <th>steps</th><th>流分歧首步 vs legacy</th></tr>
    {rows}
  </table>
  {zoom_html}
  <h2>口径 / 注记</h2>
  <ul>
{notes}
  </ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[k3-evidence] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
