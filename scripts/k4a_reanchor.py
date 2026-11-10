#!/usr/bin/env python3
"""K4a re-anchor runner: rebuild the simulation anchors under the flipped TerraZero
enforcement defaults, with the old anchors side by side.

K4a flipped the enforcement defaults (drive.py / puffer_drive.yaml):
  - traffic_light_footprint_sweep: False -> True   (TerraZero swept-path red light)
  - stop_sign_dwell_range: [0,0] -> [0.5, 1.5]     (sampled dwell; first calibration,
    must be re-calibrated with experiment data; sampling consumes env RNG, so every
    episode stream shifts by design)
  - traffic_light_truth_timing stays False (ODD-fidelity opt-in; no-op on CARLA bins)
  - pedestrian_traffic_exempt stays True (inert until active PEDESTRIAN agents exist)

Produces, under ``artifacts/reanchor-k4/``:
  - baseline/seed{42,43,44}/      four-mode demo re-runs (200-step anchor window):
    kernel-static/, kernel-jaywalker/, mix/   (same demo pipeline as the old anchors,
    now with the new default face; summaries carry red-light/stop-sign latches)
  - legacy-escape-hatch.json      seed-42 baseline re-run with the explicit pre-K4a
    legacy pins (sweep off + dwell [0,0], truth off) asserted against the frozen
    old-anchor numbers (artifacts/baseline @ 45c9329: 200 steps / 31 collisions / 0
    offroad) - the escape hatch reproduces the old behavior end to end
  - comparison.md                 old vs new anchor table (steps / collisions /
    offroad / violation rates per mode)

The old artifact dirs are left untouched; everything lands under --out.

Run: PYTHONPATH=$PWD .venv/bin/python scripts/k4a_reanchor.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hazard_demo as hd  # noqa: E402  (repo scripts dir; kernel pinning enforced below)

# metrics_array layout (pufferlib/ocean/drive/constants.h), mirrored by the demo counters
COLLISION_IDX = 0
OFFROAD_IDX = 1
RED_LIGHT_IDX = 2
STOP_SIGN_IDX = 3

SEEDS = (42, 43, 44)

# Pre-K4a legacy face, pinned explicitly (K4a flipped the python defaults, so an
# empty kwargs dict no longer means legacy).
LEGACY_KNOBS = {
    "traffic_light_truth_timing": False,
    "traffic_light_footprint_sweep": False,
    "stop_sign_dwell_range": (0.0, 0.0),
}

# Frozen old anchor for the escape-hatch assertion: artifacts/baseline/summary.json
# (kernel git 45c9329, .so sha256 443d302f..., 2 envs x 150 agents, seed 42).
OLD_BASELINE_ANCHOR = {"steps_run": 200, "collision_agents": 31, "offroad_agents": 0}

# Old-anchor sources for comparison.md (paths + the commit each was anchored at).
OLD_ANCHOR_SOURCES = {
    "baseline": ("artifacts/baseline", "45c9329", "STATE 锚：200 步/31 碰撞/0 offroad（seed42）"),
    "static-hazards": ("artifacts/static-hazards", "baf5e4f", "STATE 锚：碰撞增量 s42 +116.7% / s43 +133.3% / s44 +16.0%"),
    "kernel-static": ("artifacts/kernel-static", "be8318a", "K1 内核放置锚"),
    "kernel-jaywalker": ("artifacts/kernel-jaywalker-k2fix", "ec37906", "K2 hotfix 锚"),
    "mix": ("artifacts/mix", "45c9329", "M3 原型锚"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.3)
    p.add_argument("--goal-source", choices=["route", "map"], default="route")
    p.add_argument("--envs", type=int, default=2)
    p.add_argument("--agents-per-env", type=int, default=150)
    p.add_argument("--map-dir", default="pufferlib/resources/drive/binaries/carla")
    p.add_argument("--map", default="Town05")
    p.add_argument("--out", default="artifacts/reanchor-k4")
    p.add_argument("--skip-demo", action="store_true", help="reuse existing demo summaries, skip the runs")
    return p.parse_args()


def run_demo(args, seed, mode_args, out_dir):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "hazard_demo.py"),
        "--seed", str(seed), "--steps", str(args.steps), "--dt", str(args.dt),
        "--goal-source", args.goal_source, "--envs", str(args.envs),
        "--agents-per-env", str(args.agents_per_env),
        "--map-dir", args.map_dir, "--map", args.map, "--out", str(out_dir),
        *mode_args,
    ]
    print(f"[k4a-reanchor] run: {' '.join(cmd[len(cmd)-8:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise SystemExit(f"[k4a-reanchor] demo run failed ({mode_args}, seed {seed})")
    print(f"[k4a-reanchor]   done in {time.time() - t0:.0f}s")
    return proc.stdout


def verify_legacy_escape_hatch(args, map_bin, out_dir):
    """One explicit-legacy-pins baseline run (seed 42, demo-default env face) asserted
    against the frozen old anchor: 200 steps / 31 collisions / 0 offroad."""
    args_dict = argparse.Namespace(
        seed=42, steps=args.steps, dt=args.dt, goal_source=args.goal_source,
        envs=args.envs, agents_per_env=args.agents_per_env,
    )
    env = hd.build_env(args_dict, map_bin, extra_env_kwargs=dict(LEGACY_KNOBS))
    env.reset(seed=args_dict.seed)
    scenarios = env.get_state()
    buffers = hd.capture_buffers(env, scenarios)
    af0, ai0, mf0 = buffers["agent_f32"][0], buffers["agent_i32"][0], buffers["metrics_f32"][0]

    def capture():
        env.get_obs_html_frame(
            buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
            buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
            buffers["rewards_f32"], buffers["coefs_f32"],
        )

    capture()
    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    steps_run = 0
    for step in range(1, args_dict.steps + 1):
        env.step(actions)
        steps_run = step
        capture()
    total = {"collision_agents": 0, "offroad_agents": 0, "red_light_agents": 0, "stop_sign_agents": 0}
    for e in range(env.num_envs):
        ai0 = buffers["agent_i32"][e]
        mf0 = buffers["metrics_f32"][e]
        live = hd.env_active_mask(ai0)
        total["collision_agents"] += int(np.sum(live & (mf0[:, COLLISION_IDX] > 0)))
        total["offroad_agents"] += int(np.sum(live & (mf0[:, OFFROAD_IDX] > 0)))
        total["red_light_agents"] += int(np.sum(live & (mf0[:, RED_LIGHT_IDX] > 0)))
        total["stop_sign_agents"] += int(np.sum(live & (mf0[:, STOP_SIGN_IDX] > 0)))
    env.close()

    matches = (
        steps_run == OLD_BASELINE_ANCHOR["steps_run"]
        and total["collision_agents"] == OLD_BASELINE_ANCHOR["collision_agents"]
        and total["offroad_agents"] == OLD_BASELINE_ANCHOR["offroad_agents"]
    )
    result = {
        "claim": "显式 legacy 旋钮（sweep off + dwell [0,0] + truth off）复现 K4a 前旧行为",
        "seed": 42,
        "face": "2 envs x 150 agents, Town05, 200 steps, zero actions (demo baseline face)",
        "legacy_knobs": {k: list(v) if isinstance(v, tuple) else v for k, v in LEGACY_KNOBS.items()},
        "measured": {"steps_run": steps_run, **total},
        "old_anchor": {"source": "artifacts/baseline/summary.json", **OLD_BASELINE_ANCHOR},
        "match": bool(matches),
        "notes": [
            "窗口口径（200 步 latched 总数）与旧锚一致；旧锚 artifact 未记红灯/停牌 latch，"
            "故 legacy 面的违规 latch 数在此首次入档（窗口口径）。",
            "流级逐位一致性由 C 测试 test_explicit_legacy_knobs_keep_episode_stream_bit_identical 钉死。",
        ],
    }
    (out_dir / "legacy-escape-hatch.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[k4a-reanchor] legacy escape hatch: {result['measured']} match={matches}")
    return result


def read_demo_summaries(reanchor_root):
    out = {}
    base_root = reanchor_root / "baseline"
    out["baseline"] = []
    for seed in SEEDS:
        p = base_root / f"seed{seed}" / "summary.json"
        if p.is_file():
            out["baseline"].append(json.loads(p.read_text(encoding="utf-8")))
    for key, dirname in (("kernel-static", "kernel-static"),
                         ("kernel-jaywalker", "kernel-jaywalker"),
                         ("mix", "mix")):
        root = reanchor_root / dirname
        per_seed = []
        for seed in SEEDS:
            p = root / f"seed{seed}" / "summary.json"
            if p.is_file():
                per_seed.append(json.loads(p.read_text(encoding="utf-8")))
        out[key] = per_seed
    return out


def fmt_rate(count, agents):
    return f"{count / agents:.5f}" if agents else "-"


def build_comparison_md(reanchor_root, summaries, legacy_result, kernel, args):
    total_agents = args.envs * args.agents_per_env
    one_env_agents = args.agents_per_env  # kernel-static / kernel-jaywalker / mix force --envs 1

    def baseline_rows(per_seed):
        rows = []
        for s in per_seed:
            total = s["total"]
            rows.append(
                f"| {s['seed']} | {s['steps_run']} | {total['collision_agents']} | {total['offroad_agents']} "
                f"| {total.get('red_light_agents', '-')} ({fmt_rate(total.get('red_light_agents', 0), total_agents)}) "
                f"| {total.get('stop_sign_agents', '-')} ({fmt_rate(total.get('stop_sign_agents', 0), total_agents)}) |")
        return "\n".join(rows) or "| - | - | - | - | - | - |"

    def episode_row(sim):
        m = sim.get("baseline", {})
        h = sim.get(sim_key_for(sim), {})
        return (m, h)

    def sim_key_for(sim):
        for key in ("hazard", "jaywalker", "mix"):
            if key in sim:
                return key
        return "hazard"

    def mode_rows(per_seed, agents):
        rows = []
        for s in per_seed:
            sim = s["simulation"]
            b, h = episode_row(sim)
            key = sim_key_for(sim)
            inc = sim.get("collision_rate_change_pct", "-")
            rows.append(
                f"| {s['seed']} | {sim.get('window_step', '-')} | {b.get('collision_agents_final', '-')} | "
                f"{b.get('offroad_agents_final', '-')} | {b.get('red_light_agents_final', '-')} "
                f"({fmt_rate(b.get('red_light_agents_final', 0), agents)}) | {b.get('stop_sign_agents_final', '-')} "
                f"({fmt_rate(b.get('stop_sign_agents_final', 0), agents)}) "
                f"| {h.get('collision_agents_final', '-')} | {h.get('offroad_agents_final', '-')} "
                f"| {h.get('red_light_agents_final', '-')} ({fmt_rate(h.get('red_light_agents_final', 0), agents)}) "
                f"| {h.get('stop_sign_agents_final', '-')} ({fmt_rate(h.get('stop_sign_agents_final', 0), agents)}) "
                f"| {inc}% |")
        return "\n".join(rows) or "| - | - | - | - | - | - | - | - | - | - | - |"

    old_static = json.loads((REPO_ROOT / OLD_ANCHOR_SOURCES["kernel-static"][0] / "summary.json")
                            .read_text(encoding="utf-8"))
    old_py_static = json.loads((REPO_ROOT / OLD_ANCHOR_SOURCES["static-hazards"][0] / "summary.json")
                               .read_text(encoding="utf-8"))
    old_jw_root = REPO_ROOT / OLD_ANCHOR_SOURCES["kernel-jaywalker"][0]
    old_jw = [json.loads((old_jw_root / f"seed{s}" / "summary.json").read_text(encoding="utf-8"))
              for s in SEEDS if (old_jw_root / f"seed{s}" / "summary.json").is_file()]
    old_mix = json.loads((REPO_ROOT / OLD_ANCHOR_SOURCES["mix"][0] / "summary.json").read_text(encoding="utf-8"))
    old_base = json.loads((REPO_ROOT / OLD_ANCHOR_SOURCES["baseline"][0] / "summary.json").read_text(encoding="utf-8"))

    old_static_rows = "\n".join(
        f"| {s['seed']} | {s['simulation']['window_step']} | {s['simulation']['collision_agents_baseline_at_window']} | "
        f"{s['simulation']['collision_agents_hazard_at_window']} | {s['simulation']['collision_rate_change_pct']}% |"
        for s in old_static["per_seed"])
    old_py_static_rows = "\n".join(
        f"| {s['seed']} | {s['simulation']['window_step']} | {s['simulation']['collision_agents_baseline_at_window']} | "
        f"{s['simulation']['collision_agents_hazard_at_window']} | {s['simulation']['collision_rate_change_pct']}% |"
        for s in old_py_static["per_seed"])
    old_jw_rows = "\n".join(
        f"| {s['seed']} | {s['simulation']['window_step']} | {s['simulation']['collision_agents_baseline_final']} | "
        f"{s['simulation']['collision_agents_jaywalker_final']} | {s['reaction']['slowdown_events_baseline_total']}→"
        f"{s['reaction']['slowdown_events_jaywalker_total']} | {s['reaction']['walker_contact_vehicles_total']} | "
        f"{len(s.get('walkers', []))} |"
        for s in old_jw)
    old_mix_rows = "\n".join(
        f"| {s['seed']} | {s['simulation']['baseline']['steps_run']} | "
        f"{s['simulation']['baseline']['collision_agents_final']} | {s['simulation']['mix']['collision_agents_final']} | "
        f"{s['simulation']['mix']['offroad_agents_final']} | {s.get('interaction', {}).get('walkers_completed', '-')} |"
        for s in old_mix["per_seed"])

    new_static_rows = mode_rows(summaries.get("kernel-static", []), one_env_agents)
    new_jw_rows = mode_rows(summaries.get("kernel-jaywalker", []), one_env_agents)
    new_mix_rows = mode_rows(summaries.get("mix", []), one_env_agents)
    new_base_rows = baseline_rows(summaries.get("baseline", []))

    agg = {}
    for s in summaries.get("kernel-jaywalker", []):
        r = s.get("reaction", {})
        agg.setdefault("slowdown", []).append((r.get("slowdown_events_baseline_total"), r.get("slowdown_events_jaywalker_total")))
        agg.setdefault("contact", []).append(r.get("walker_contact_vehicles_total"))
        agg.setdefault("walkers", []).append(len(s.get("walkers", [])))
    jw_agg_line = (f"新锚 jaywalker：walker {sum(agg.get('walkers', []))}，"
                   f"减速事件 baseline {sum(a for a, _ in agg.get('slowdown', []))} → jaywalker {sum(b for _, b in agg.get('slowdown', []))}，"
                   f"接触 {sum(agg.get('contact', []))}")

    match_line = "✅ 一致" if legacy_result["match"] else "❌ 不一致（见 legacy-escape-hatch.json）"

    return f"""# K4a 重定锚：统一切默认后的新基线（与旧锚并排）

## 裁决与语义

- `traffic_light_footprint_sweep`: False → **True**（红灯 = 全身扫越停止线，TerraZero swept-path 口径）
- `stop_sign_dwell_range`: [0,0] → **[0.5, 1.5]**（每牌采样驻停时长；分布未披露，首版标定值，需实验再标定）
- `traffic_light_truth_timing`: 保持 False（ODD 保真场景手动开；CARLA/无 states bin 本就 no-op）
- 行人豁免 gate：保持 on（当前流中无 active PEDESTRIAN，惯性生效）
- **流位移披露**：dwell 采样每 episode 对每块停牌抽一个均匀数（Town05 = 17 块），reset 期消耗 RNG，
  下游 spawn/灯态流整体位移——这是设计内行为，不是回归。K3 evidence 已录同一效应。

## kernel 溯源

- binding .so: `{kernel['binding_so']}`
- .so sha256: `{kernel['binding_sha256']}`
- git sha: `{kernel['git_sha']}` · 构建: `{kernel['built_by']}`
- 命令: `PYTHONPATH=$PWD .venv/bin/python scripts/k4a_reanchor.py --seeds {','.join(map(str, SEEDS))}`

## 口径

- 200 步锚窗口、IDM 全员、零动作步进（demo 各模式默认面）。
- 违规率 = latched agents / episode 内 agent 总数（baseline 2×150=300；kernel-static/jaywalker/mix 单 env 150）。
- 红灯/停牌 latch 计数为本次新增的 demo summary 字段（K4a 起每模式入档）；旧锚 artifact 未记该字段，
  旧违规率仅有 K3 全 episode 口径（`artifacts/k3-patches/comparison.md`，episode_length 307，rate 0.01333/0.18），
  与 200 步窗口不可直接比——如实分列，不做换算。

## 1. baseline（纯 gigaflow，四模式共同参照）

**旧锚**（artifacts/baseline @ 45c9329，seed42；seed43/44 旧锚无独立 artifact，static-hazards 内部基线可参看下表）：
STATE 锚 = 200 步 / 31 碰撞 / 0 offroad。

| seed | 步数 | 碰撞 | offroad | 红灯 latch（率） | 停牌 latch（率） |
|---|---|---|---|---|---|
{new_base_rows}

## 2. static-hazards（kernel 放置，K1 面）

**旧锚**（kernel-static @ be8318a，legacy 执法面）：

| seed | 窗口 | 基线碰撞 | hazard 碰撞 | 增量 |
|---|---|---|---|---|
{old_static_rows}

（STATE 锚 +116.7%/+133.3%/+16.0% 出自 python 放置 M1 面 artifacts/static-hazards @ baf5e4f：

| seed | 窗口 | 基线碰撞 | hazard 碰撞 | 增量 |
|---|---|---|---|---|
{old_py_static_rows}
）

**新锚**（新执法面，K1 放置不变）：

| seed | 窗口 | 基线碰撞 | 基线 offroad | 基线红灯（率） | 基线停牌（率） | hz 碰撞 | hz offroad | hz 红灯（率） | hz 停牌（率） | 碰撞增量 |
|---|---|---|---|---|---|---|---|---|---|---|
{new_static_rows}

## 3. jaywalker（kernel 行人，K2 面）

**旧锚**（kernel-jaywalker-k2fix @ ec37906，legacy 执法面）：

| seed | 窗口 | 基线碰撞 | jaywalker 碰撞 | 减速事件 base→jw | 接触 | walkers |
|---|---|---|---|---|---|---|
{old_jw_rows}

**新锚**（新执法面，K2 步进不变）：

| seed | 窗口 | 基线碰撞 | 基线 offroad | 基线红灯（率） | 基线停牌（率） | jw 碰撞 | jw offroad | jw 红灯（率） | jw 停牌（率） | 碰撞增量 |
|---|---|---|---|---|---|---|---|---|---|---|
{new_jw_rows}

{jw_agg_line}（旧锚 aggregate：6 walker / 完成 6 / 减速 0→1 / 接触 1）

## 4. mix（M3 配置驱动全类型）

**旧锚**（mix @ 45c9329，legacy 执法面）：aggregate parked 19 / 簇 5 / 锥桶 52 / walker 4 完成 4 /
碰撞 base 46 → mix 81 / offroad 0 / 非有意动态重叠 0。

| seed | 步数 | 基线碰撞 | mix 碰撞 | mix offroad | 完成 walker |
|---|---|---|---|---|---|
{old_mix_rows}

**新锚**（新执法面）：

| seed | 窗口 | 基线碰撞 | 基线 offroad | 基线红灯（率） | 基线停牌（率） | mix 碰撞 | mix offroad | mix 红灯（率） | mix 停牌（率） | 碰撞增量 |
|---|---|---|---|---|---|---|---|---|---|---|
{new_mix_rows}

## 5. legacy 逃生门验证（一次）

- 配置：baseline 面 + 显式 legacy 旋钮（sweep off + dwell [0,0] + truth off），seed42，2×150。
- 断言：{OLD_BASELINE_ANCHOR['steps_run']} 步 / {OLD_BASELINE_ANCHOR['collision_agents']} 碰撞 / {OLD_BASELINE_ANCHOR['offroad_agents']} offroad
  （= artifacts/baseline @ 45c9329 旧锚）→ **{match_line}**
- 实测：{legacy_result['measured']}
- 明细：`legacy-escape-hatch.json`（含新口径下 legacy 面的红灯/停牌 latch 首次入档）。
- 流级逐位一致性由 C 测试 `test_explicit_legacy_knobs_keep_episode_stream_bit_identical` 钉死。

## 6. 训练入口就绪（K4b）

- 主配置 `pufferlib/config/puffer_drive.yaml` env 段新增 hazard 块（K1/K2 放置旋钮 + K3/K4a 三执法旋钮），
  带 K4a 默认值与注释；`env` 段 1:1 映射 `Drive` kwargs（`pufferl.py` train 链路
  `load_config -> normalize_puffer_drive_config -> vector.make(env_kwargs=args["env"])`）。
- schema/校验：`config_schema.DriveEnvConfig` 增设全部 hazard 字段 + cross-field 镜像校验
  （dwell 序、gigaflow 依赖、权重和、walker 槽帽等），坏配置在 load 期即报 `APIUsageError`。
- 单测：`tests/unit_tests/test_config_schema_hazard.py`（yaml→schema→kwargs 链路 + 失败路径）；
  `tests/unit_tests/test_hazard_kernel_config.py`（K4a 默认面 + legacy 逃生门 kwargs 面）。
- K4b 只差 PPO smoke（1e6 步 reward 曲线，hazard 开/关对比）：训练入口 + 配置路径已接线，本票零训练。
"""


def main():
    args = parse_args()
    hd.verify_in_repo_kernel()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    global SEEDS
    SEEDS = seeds

    reanchor_root = (REPO_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    reanchor_root.mkdir(parents=True, exist_ok=True)
    map_bin = hd.resolve_map_bin(args.map_dir, args.map)

    if not args.skip_demo:
        for seed in seeds:
            run_demo(args, seed, ["--mode", "baseline"], reanchor_root / "baseline" / f"seed{seed}")
        for seed in seeds:
            run_demo(args, seed, ["--mode", "static-hazards", "--placement", "kernel"], reanchor_root / "kernel-static")
        for seed in seeds:
            run_demo(args, seed, ["--mode", "jaywalker", "--placement", "kernel"], reanchor_root / "kernel-jaywalker")
        for seed in seeds:
            run_demo(args, seed, ["--mode", "mix"], reanchor_root / "mix")

    legacy_result = verify_legacy_escape_hatch(args, map_bin, reanchor_root)
    summaries = read_demo_summaries(reanchor_root)
    kernel = hd.kernel_provenance()
    md = build_comparison_md(reanchor_root, summaries, legacy_result, kernel, args)
    (reanchor_root / "comparison.md").write_text(md, encoding="utf-8")
    print(f"[k4a-reanchor] artifacts -> {reanchor_root} (comparison.md + legacy-escape-hatch.json)")


if __name__ == "__main__":
    main()
