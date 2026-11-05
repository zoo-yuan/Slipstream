"""Ticket 07 M3: hazard config face + mix mode tests.

Two layers:

- Config validation (validate_hazard_config / load_hazard_config /
  mix_slot_capacity): pure Python, runs anywhere, fail-fast contract —
  out-of-range / wrong-typed knobs must raise HazardConfigError with the
  offending key path before the env is built.
- Mix episode harness (20 agents / 50 steps, no rendering): deterministic
  double-run bit-identity, all-disabled == pure gigaflow (the M3 regression
  contract), and placement legality invariants (off-road violations = 0,
  non-intentional dynamic overlap = 0). Skipped unless the import chain
  resolves to THIS repo's kernel and the Town05 map bin is available.

Run with the pinned in-repo kernel (repo pufferlib shadows the sibling venv):

    PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python -m pytest tests/test_hazard_demo.py -v
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import hazard_demo as hd  # noqa: E402  (repo-root path inserted above)


# --- guards -------------------------------------------------------------------

def _kernel_probe():
    try:
        import pufferlib
        import pufferlib.ocean.drive.binding as binding
    except Exception as exc:  # noqa: BLE001 - any import failure is a skip reason
        return False, f"pufferlib import failed ({exc})"
    root = Path(hd.REPO_ROOT)
    ok = (Path(pufferlib.__file__).resolve().is_relative_to(root)
          and Path(binding.__file__).resolve().is_relative_to(root))
    reason = ("repo kernel not on the import path — run with "
              "PYTHONPATH=$PWD ../PufferDrive/.venv/bin/python -m pytest so the "
              "in-repo kernel shadows the sibling venv copy")
    return ok, reason


_KERNEL_OK, _KERNEL_SKIP_REASON = _kernel_probe()
requires_kernel = pytest.mark.skipif(not _KERNEL_OK, reason=_KERNEL_SKIP_REASON)


@pytest.fixture(scope="module")
def map_bin():
    if not _KERNEL_OK:
        pytest.skip(_KERNEL_SKIP_REASON)
    try:
        return hd.resolve_map_bin(hd.DEFAULT_MAP_DIR, hd.DEFAULT_MAP)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


# --- config validation (pure Python, no kernel) --------------------------------

class TestHazardConfigValidation:
    def test_repo_default_config_validates_and_matches_frozen_defaults(self):
        cfg = hd.load_hazard_config(None)
        assert cfg["enable"] is True
        assert cfg["parked"] == {"enable": True, "n_range": [6, 10],
                                 "sizes": ["car", "bus", "truck"],
                                 "size_weights": [0.6, 0.2, 0.2]}
        assert cfg["clusters"] == {"enable": True, "n_range": [1, 2],
                                   "layouts": ["rear_end", "t_bone"]}
        assert cfg["cones"] == {"enable": True, "zones_range": [1, 2],
                                "taper_cones": 8, "block_rows": 2, "block_cols": 7,
                                "spacing_m": 1.5, "taper_prob": 0.5}
        assert cfg["jaywalker"] == {"enable": True, "n_range": [1, 2],
                                    "speed_range": [1.2, 1.6]}
        assert cfg["cut_in"] == {"enable": False}  # three-seed abandoned record

    def test_full_valid_dict_normalizes(self):
        raw = {
            "enable": True,
            "parked": {"enable": True, "n_range": [3, 7], "sizes": ["bus", "car"],
                       "size_weights": [0.25, 0.75]},
            "clusters": {"enable": True, "n_range": [0, 1], "layouts": ["t_bone"]},
            "cones": {"enable": True, "zones_range": [2, 4], "taper_cones": 6,
                      "block_rows": 1, "block_cols": 5, "spacing_m": 2.0,
                      "taper_prob": 1.0},
            "jaywalker": {"enable": True, "n_range": [1, 3],
                          "speed_range": [0.8, 2.5]},
            "cut_in": {"enable": True},
        }
        cfg = hd.validate_hazard_config(raw)
        assert cfg["parked"]["n_range"] == [3, 7]
        assert cfg["clusters"]["layouts"] == ["t_bone"]
        assert cfg["cones"]["spacing_m"] == 2.0
        assert cfg["jaywalker"]["speed_range"] == [0.8, 2.5]
        assert cfg["cut_in"]["enable"] is True
        assert "_path" not in cfg  # only load_hazard_config attaches provenance

    def test_partial_config_merges_defaults(self):
        cfg = hd.validate_hazard_config({"cones": {"taper_cones": 10}})
        assert cfg["cones"]["taper_cones"] == 10
        assert cfg["cones"]["block_rows"] == 2  # untouched default
        assert cfg["parked"]["n_range"] == [6, 10]
        assert cfg["cut_in"] == {"enable": False}

    def test_all_disabled_passes_and_capacity_zero(self):
        cfg = hd.validate_hazard_config({
            "enable": False, "parked": {"enable": False},
            "clusters": {"enable": False}, "cones": {"enable": False},
            "jaywalker": {"enable": False}, "cut_in": {"enable": False}})
        assert hd.mix_slot_capacity(cfg) == 0

    def test_sizes_override_without_weights_gets_uniform(self):
        cfg = hd.validate_hazard_config({"parked": {"sizes": ["car", "truck"]}})
        assert cfg["parked"]["size_weights"] == [0.5, 0.5]

    @pytest.mark.parametrize("raw", [
        {"parked": {"n_range": [5]}, "x": "wrong length"},
        {"parked": {"n_range": [1.5, 3]}},
        {"parked": {"n_range": [True, 3]}},
        {"parked": {"n_range": [3, 2]}},
        {"parked": {"n_range": [-1, 2]}},
    ])
    def test_bad_n_range_rejected(self, raw):
        with pytest.raises(hd.HazardConfigError, match="parked"):
            hd.validate_hazard_config(raw)

    def test_unknown_keys_rejected(self):
        with pytest.raises(hd.HazardConfigError, match="unknown key"):
            hd.validate_hazard_config({"totally_bogus": 1})
        with pytest.raises(hd.HazardConfigError, match="unknown key"):
            hd.validate_hazard_config({"parked": {"n_rang": [1, 2]}})

    def test_enable_must_be_bool(self):
        with pytest.raises(hd.HazardConfigError, match="bool"):
            hd.validate_hazard_config({"enable": 1})
        with pytest.raises(hd.HazardConfigError, match="bool"):
            hd.validate_hazard_config({"cut_in": {"enable": "yes"}})

    def test_parked_sizes_rejected(self):
        with pytest.raises(hd.HazardConfigError, match="sizes"):
            hd.validate_hazard_config({"parked": {"sizes": ["moto"]}})
        with pytest.raises(hd.HazardConfigError, match="sizes"):
            hd.validate_hazard_config({"parked": {"sizes": []}})
        with pytest.raises(hd.HazardConfigError, match="size_weights"):
            hd.validate_hazard_config({"parked": {"size_weights": [0.5, 0.5]}})
        with pytest.raises(hd.HazardConfigError, match="sum to 1.0"):
            hd.validate_hazard_config(
                {"parked": {"sizes": ["car", "bus"], "size_weights": [0.5, 0.2]}})
        with pytest.raises(hd.HazardConfigError, match="non-negative"):
            hd.validate_hazard_config(
                {"parked": {"sizes": ["car", "bus"], "size_weights": [1.5, -0.5]}})

    def test_cluster_layouts_rejected(self):
        with pytest.raises(hd.HazardConfigError, match="layouts"):
            hd.validate_hazard_config({"clusters": {"layouts": ["side_swipe"]}})
        with pytest.raises(hd.HazardConfigError, match="layouts"):
            hd.validate_hazard_config({"clusters": {"layouts": []}})

    @pytest.mark.parametrize("raw", [
        {"cones": {"taper_cones": 1}},
        {"cones": {"block_rows": 0}},
        {"cones": {"block_cols": 1}},
        {"cones": {"spacing_m": 0}},
        {"cones": {"taper_prob": -0.1}},
        {"cones": {"taper_prob": 1.5}},
        {"cones": {"zones_range": [2, 1]}},
    ])
    def test_cones_rejected(self, raw):
        with pytest.raises(hd.HazardConfigError, match="cones"):
            hd.validate_hazard_config(raw)

    @pytest.mark.parametrize("raw", [
        {"jaywalker": {"speed_range": [0, 1.6]}},
        {"jaywalker": {"speed_range": [1.6, 1.2]}},
        {"jaywalker": {"speed_range": [1.0]}},
        {"jaywalker": {"speed_range": ["a", 2.0]}},
        {"jaywalker": {"n_range": [-1, 1]}},
    ])
    def test_jaywalker_rejected(self, raw):
        with pytest.raises(hd.HazardConfigError, match="jaywalker"):
            hd.validate_hazard_config(raw)

    def test_root_must_be_mapping(self):
        with pytest.raises(hd.HazardConfigError, match="mapping"):
            hd.validate_hazard_config([1, 2])

    def test_load_errors(self, tmp_path):
        with pytest.raises(hd.HazardConfigError, match="not found"):
            hd.load_hazard_config(tmp_path / "missing.yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("parked: [unclosed", encoding="utf-8")
        with pytest.raises(hd.HazardConfigError, match="valid YAML"):
            hd.load_hazard_config(bad)

    def test_mix_slot_capacity_bounds(self):
        cfg = hd.validate_hazard_config({
            "parked": {"n_range": [0, 10]}, "clusters": {"n_range": [1, 2]},
            "cones": {"zones_range": [1, 2]}, "jaywalker": {"n_range": [1, 2]},
            "cut_in": {"enable": True}})
        # parked hi + 2*clusters hi + zones hi*max(taper, rows*cols) + walkers hi + cut 1
        assert hd.mix_slot_capacity(cfg) == 10 + 4 + 2 * 14 + 2 + 1
        assert hd.mix_slot_capacity(hd.validate_hazard_config({"enable": False})) == 0


# --- mix episode harness (20 agents / 50 steps, no rendering) ------------------

SMALL_CFG = hd.validate_hazard_config({
    "parked": {"n_range": [2, 2]},
    "clusters": {"n_range": [1, 1]},
    "cones": {"zones_range": [1, 1]},
    "jaywalker": {"n_range": [1, 1], "speed_range": [1.2, 1.4]},
    "cut_in": {"enable": False},
})
ALL_OFF_CFG = hd.validate_hazard_config({
    "enable": False, "parked": {"enable": False}, "clusters": {"enable": False},
    "cones": {"enable": False}, "jaywalker": {"enable": False},
    "cut_in": {"enable": False}})


def _harness_args(seed, agents_per_env=20):
    return argparse.Namespace(envs=1, agents_per_env=agents_per_env, seed=seed,
                              dt=0.3, goal_source="route", steps=50)


def run_mix_episode(map_bin, seed, cfg, steps=50, agents_per_env=20):
    """Render-free copy of the run_mix placement/stepping core: static classes
    placed once, walker slots stepped every step via set_agent_states BEFORE
    env.step. Returns (placed, walkers, captures) where captures is a list of
    per-step (agent_f32, agent_i32, metrics_f32) snapshots."""
    args = _harness_args(seed, agents_per_env)
    args.steps = steps
    slots = hd.mix_slot_capacity(cfg)
    env = hd.build_env(args, map_bin, cosim_partner_slots=slots)
    try:
        env.reset(seed=seed)
        scenarios = env.get_state()
        buffers = hd.capture_buffers(env, scenarios, extra_agent_slots=slots)

        def capture():
            env.get_obs_html_frame(
                buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
                buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
                buffers["rewards_f32"], buffers["coefs_f32"])

        capture()
        af0, ai0 = buffers["agent_f32"][0], buffers["agent_i32"][0]
        lanes = hd.LaneIndex(scenarios[0].get("road_elements") or [])
        edges = hd.EdgeIndex(scenarios[0].get("road_elements") or [])
        cw_segs = hd.crosswalk_segments(scenarios[0].get("road_elements") or [])
        dyn = hd.dynamic_boxes(af0, ai0)

        rng, wrng = hd.hazard_rng(seed), hd.walker_rng(seed)
        rejects = {r: 0 for r in hd.REJECT_REASONS}
        placed, walkers = [], []
        if cfg["enable"]:
            pc, cc, kc, jc = cfg["parked"], cfg["clusters"], cfg["cones"], cfg["jaywalker"]
            plan = hd.plan_hazard_counts(
                rng,
                parked_range=tuple(pc["n_range"]) if pc["enable"] else (0, 0),
                cluster_range=tuple(cc["n_range"]) if cc["enable"] else (0, 0),
                zone_range=tuple(kc["zones_range"]) if kc["enable"] else (0, 0))
            size_table = None
            if pc["enable"]:
                size_table = tuple(zip(pc["sizes"],
                                       [hd.VEHICLE_PRESETS[s] for s in pc["sizes"]],
                                       pc["size_weights"]))
            hd.place_parked(rng, lanes, edges, dyn, placed, rejects, plan["parked"],
                            size_table=size_table)
            hd.place_clusters(rng, lanes, edges, dyn, placed, rejects,
                              plan["clusters"], layouts=cc["layouts"])
            hd.place_cone_zones(rng, lanes, edges, dyn, placed, rejects,
                                plan["cone_zones"], cones=kc)
            if jc["enable"]:
                n_w = int(wrng.randint(jc["n_range"][0], jc["n_range"][1] + 1))
                walkers, _rej = hd.plan_walkers(wrng, lanes, edges, cw_segs, dyn, n_w,
                                                static_placed=placed)
        for k, w in enumerate(walkers):
            w["row"] = args.agents_per_env + len(placed) + k
            w["speed_mps"] = float(wrng.uniform(*cfg["jaywalker"]["speed_range"]))
            w["start_step"] = 1 + k * hd.WALKER_START_STAGGER_STEPS
            est = int(np.ceil(w["crossing_width_m"] / (w["speed_mps"] * args.dt)))
            w["mid_step"] = min(w["start_step"] + max(est // 2, 1), steps - 1)
            w["traveled"] = 0.0
            w["done_step"] = None
        initial = placed + walkers
        if initial:
            hd.apply_static_hazards(env, args, initial)
            capture()

        actions = np.zeros((env.num_agents, 2), dtype=np.float32)
        captures = []
        for step in range(1, steps + 1):
            idxs, xs, ys, zs, hs, vxs, vys = [], [], [], [], [], [], []
            for w in walkers:
                if w["done_step"] is not None or step < w["start_step"]:
                    continue
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
            if idxs:
                env.set_agent_states(
                    np.asarray(idxs, np.int32),
                    np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                    np.asarray(zs, np.float32), np.asarray(hs, np.float32),
                    np.asarray(vxs, np.float32), np.asarray(vys, np.float32),
                    np.zeros(len(idxs), np.float32), np.zeros(len(idxs), np.float32),
                    None, np.zeros(len(idxs), np.float32))
            env.step(actions)
            capture()
            captures.append((buffers["agent_f32"][0].copy(),
                             buffers["agent_i32"][0].copy(),
                             buffers["metrics_f32"][0].copy()))
        legality = (hd.verify_static_hazards(env, placed, [e["idx"] for e in placed],
                                             captures[-1][0], captures[-1][1])
                    if placed else None)
        return placed, walkers, captures, legality
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            env.close()


def _placement_key(placed, walkers):
    return (
        [(e["kind"], repr(e["x"]), repr(e["y"]), repr(e["heading"]),
          e["length"], e["width"]) for e in placed],
        [(w["k"], repr(w["start"][0]), repr(w["start"][1]), repr(w["speed_mps"]))
         for w in walkers],
    )


@requires_kernel
class TestMixEpisodes:
    def test_mix_episode_deterministic(self, map_bin):
        """20 agents / 50 steps, same seed double run: placement and every
        captured frame bit-identical."""
        a = run_mix_episode(map_bin, seed=1234, cfg=SMALL_CFG)
        b = run_mix_episode(map_bin, seed=1234, cfg=SMALL_CFG)
        assert _placement_key(a[0], a[1]) == _placement_key(b[0], b[1])
        assert len(a[2]) == len(b[2]) == 50
        for (fa, ia, ma), (fb, ib, mb) in zip(a[2], b[2]):
            assert np.array_equal(fa, fb)
            assert np.array_equal(ia, ib)
            assert np.array_equal(ma, mb)
        assert a[0], "small mix config placed nothing"
        assert a[1], "small mix config placed no walker"

    def test_mix_disabled_equals_pure_gigaflow(self, map_bin):
        """Regression gate (b) at harness scale: all-disabled config reserves
        zero co-sim slots and places nothing, so the episode must be
        bit-identical to the plain baseline build."""
        placed, walkers, captures, _legality = run_mix_episode(map_bin, seed=7, cfg=ALL_OFF_CFG)
        assert placed == [] and walkers == []
        args = _harness_args(7)
        env = hd.build_env(args, map_bin)
        try:
            env.reset(seed=7)
            scenarios = env.get_state()
            buffers = hd.capture_buffers(env, scenarios)

            def capture():
                env.get_obs_html_frame(
                    buffers["agent_f32"], buffers["agent_i32"], buffers["metrics_f32"],
                    buffers["puffer_f32"], buffers["traffic_i16"], buffers["goals_f32"],
                    buffers["rewards_f32"], buffers["coefs_f32"])

            capture()
            actions = np.zeros((env.num_agents, 2), dtype=np.float32)
            for step, (fm, im, mm) in enumerate(captures, start=1):
                env.step(actions)
                capture()
                assert np.array_equal(buffers["agent_f32"][0], fm), f"step {step} drift"
                assert np.array_equal(buffers["agent_i32"][0], im), f"step {step} drift"
                assert np.array_equal(buffers["metrics_f32"][0], mm), f"step {step} drift"
        finally:
            with contextlib.redirect_stdout(io.StringIO()):
                env.close()

    def test_placement_legality_invariants(self, map_bin):
        """Off-road violations = 0 (every static pose on-road) and
        non-intentional dynamic overlap = 0 (only intra-cluster overlaps are
        allowed), checked on the placed poses, the kernel postcheck, and the
        dynamic agents' offroad metrics over the episode."""
        placed, walkers, captures, legality = run_mix_episode(map_bin, seed=99, cfg=SMALL_CFG)
        road = None
        assert placed and walkers
        args = _harness_args(99)
        env = None
        try:
            scenarios = None
            # rebuild geometry from a fresh env (read-only map access)
            env = hd.build_env(args, map_bin)
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset(seed=99)
            scenarios = env.get_state()
        finally:
            if env is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    env.close()
        road = scenarios[0].get("road_elements") or []
        edges = hd.EdgeIndex(road)

        # off-road invariant: no placed static pose crosses a road edge
        for e in placed:
            pose = (e["x"], e["y"], e["heading"], e["length"], e["width"])
            assert not edges.blocked(pose), f"off-road placement: {e}"

        # overlap invariant: only intentional intra-cluster overlaps
        initial_f32, initial_i32 = captures[0][0], captures[0][1]
        dyn = hd.dynamic_boxes(initial_f32, initial_i32)
        for e in placed:
            pose = (e["x"], e["y"], e["heading"], e["length"], e["width"])
            overlaps_dyn = sum(1 for d in dyn if hd.obb_overlap(pose, d, margin=0.05))
            assert overlaps_dyn == 0, f"non-intentional dynamic overlap: {e}"
        by_cluster = {}
        for e in placed:
            if e.get("cluster_id"):
                by_cluster.setdefault(e["cluster_id"], []).append(e)
        for pair in by_cluster.values():
            assert len(pair) == 2 and hd.obb_overlap(
                (pair[0]["x"], pair[0]["y"], pair[0]["heading"],
                 pair[0]["length"], pair[0]["width"]),
                (pair[1]["x"], pair[1]["y"], pair[1]["heading"],
                 pair[1]["length"], pair[1]["width"]), margin=0.0), \
                "cluster pair must intentionally overlap"

        # kernel postcheck + walker placement evidence
        assert legality is not None
        assert legality["dynamic_overlap_violations_postcheck"] == 0
        assert legality["world_readback_max_err_m"] < 1e-3
        for w in walkers:
            assert w["start_step"] >= 1 and w["crossing_width_m"] > 0

        # dynamic agents never go off-road during the mix episode
        for f32, i32, m32 in captures:
            live = (i32[:, 3] == 1) & (i32[:, 2] == 1)
            assert int(np.sum(live & (m32[:, hd.OFFROAD_IDX] > 0))) == 0
