#!/usr/bin/env python3
"""K4a hazard section in the main training config (pufferlib/config/puffer_drive.yaml):
schema admission, K4a default values, Drive-kwarg parity, and the fail-fast rules that
mirror the kernel's init-time validation.

Standalone on purpose (like test_hazard_kernel_config.py): it must run in the
demo-borrowed sibling venv, and pufferl.py imports pufferlib._C (the PPO advantage
kernel), which the repo's NO_TRAIN=.so pinning discipline does not build. The config
chain under test (yaml -> OmegaConf schema -> normalized dict) is _C-free, so these
tests compose it directly via hydra - pufferl.load_config minus its argv plumbing.

Run: PYTHONPATH=$PWD python -m unittest tests.unit_tests.test_config_schema_hazard
"""

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pufferlib
from pufferlib.config_schema import (
    normalize_puffer_drive_config,
    puffer_drive_constructor_keys,
    validate_puffer_drive_config,
)


def compose_puffer_drive_config(overrides=()):
    """Compose the repo yaml via Hydra and run the full config chain
    (normalize + validate), mirroring the training entry: pufferl.load_config then
    validate_puffer_drive_config(args, "training")."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(pufferlib.__file__)))
    config_dir = os.path.join(repo_root, "pufferlib", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="puffer_drive", overrides=list(overrides))
    args = defaultdict(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    args["train"]["use_rnn"] = args["rnn_name"] is not None  # derived by load_config from rnn_name
    args = defaultdict(dict, normalize_puffer_drive_config(args, "test"))
    validate_puffer_drive_config(args, "test")
    return args


# The K4a hazard face: K1/K2 hazard subsystem + K3 enforcement knobs, one key per
# Drive constructor kwarg, all under env: (the env section maps 1:1 onto env_kwargs).
HAZARD_ENV_KEYS = (
    "hazard_enable",
    "hazard_parked_enable", "hazard_parked_n_lo", "hazard_parked_n_hi",
    "hazard_parked_car_weight", "hazard_parked_bus_weight", "hazard_parked_truck_weight",
    "hazard_clusters_enable", "hazard_clusters_n_lo", "hazard_clusters_n_hi",
    "hazard_clusters_layout",
    "hazard_cones_enable", "hazard_cones_zones_lo", "hazard_cones_zones_hi",
    "hazard_cones_taper_cones", "hazard_cones_block_rows", "hazard_cones_block_cols",
    "hazard_cones_spacing_m", "hazard_cones_taper_prob",
    "hazard_walkers_enable", "hazard_walkers_n_lo", "hazard_walkers_n_hi",
    "hazard_walkers_speed_lo_mps", "hazard_walkers_speed_hi_mps",
    "traffic_light_truth_timing", "traffic_light_footprint_sweep",
    "stop_sign_dwell_range", "pedestrian_traffic_exempt",
)


class TestHazardConfigSection(unittest.TestCase):
    def test_hazard_section_loads_with_k4a_defaults(self):
        """The main config carries the hazard face end to end: schema admits every
        key, the composed yaml carries the K4a defaults, and every key is a Drive
        constructor kwarg (the env section maps 1:1 onto env_kwargs)."""
        args = compose_puffer_drive_config()
        constructor_keys = puffer_drive_constructor_keys()
        env = args["env"]
        for key in HAZARD_ENV_KEYS:
            self.assertIn(key, env, f"env.{key} missing from composed config")
            self.assertIn(key, constructor_keys, f"env.{key} is not a Drive constructor kwarg")
        self.assertIs(env["hazard_enable"], False)
        self.assertIs(env["hazard_parked_enable"], True)
        self.assertIs(env["hazard_walkers_enable"], False)
        self.assertIs(env["traffic_light_footprint_sweep"], True)
        self.assertIs(env["traffic_light_truth_timing"], False)
        self.assertIs(env["pedestrian_traffic_exempt"], True)
        self.assertEqual(env["stop_sign_dwell_range"], [0.5, 1.5])

    def test_inverted_stop_sign_dwell_range_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "stop_sign_dwell_range"):
            compose_puffer_drive_config(["env.stop_sign_dwell_range=[1.5,0.5]"])

    def test_negative_stop_sign_dwell_range_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "stop_sign_dwell_range"):
            compose_puffer_drive_config(["env.stop_sign_dwell_range=[-0.1,1.0]"])

    def test_hazard_enable_requires_gigaflow_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "hazard_enable"):
            compose_puffer_drive_config(["env.hazard_enable=true", "env.simulation_mode=replay"])

    def test_unknown_hazard_cluster_layout_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "hazard_clusters_layout"):
            compose_puffer_drive_config(["env.hazard_enable=true", "env.hazard_clusters_layout=side_swipe"])

    def test_inverted_walker_speed_range_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "hazard_walkers_speed_lo_mps"):
            compose_puffer_drive_config(
                ["env.hazard_enable=true", "env.hazard_walkers_enable=true",
                 "env.hazard_walkers_speed_lo_mps=2.0", "env.hazard_walkers_speed_hi_mps=1.6"]
            )

    def test_legacy_override_flags_compose(self):
        """The documented legacy escape hatch must be expressible as plain Hydra
        overrides on the training entry."""
        env = compose_puffer_drive_config(
            ["env.traffic_light_footprint_sweep=false", "env.stop_sign_dwell_range=[0.0,0.0]"]
        )["env"]
        self.assertIs(env["traffic_light_footprint_sweep"], False)
        self.assertEqual(env["stop_sign_dwell_range"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
