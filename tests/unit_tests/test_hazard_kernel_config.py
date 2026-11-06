#!/usr/bin/env python3
"""K1 in-kernel static hazard knobs (pufferlib/ocean/drive): fail-fast validation
BEFORE env init, the planned-count-high slot budget contract, and the kwargs
surface that reaches the C binding.

Standalone on purpose: it must run in the demo-borrowed sibling venv (no torch),
so it imports only pufferlib.ocean.drive.drive (repo pufferlib shadows the venv
copy via PYTHONPATH=$PWD).

Run: PYTHONPATH=$PWD python -m unittest tests.unit_tests.test_hazard_kernel_config
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.drive.drive import Drive


class TestHazardKernelConfig(unittest.TestCase):
    def test_defaults_are_off_and_stream_neutral(self):
        env = Drive(config_only=True)
        kwargs = env._env_init_kwargs(map_file="x.bin", max_agents=32)
        self.assertEqual(kwargs["hazard_enable"], 0)
        # defaults mirror the frozen M1 prototype numbers (config/hazard_demo.yaml)
        self.assertEqual(kwargs["hazard_parked_n_lo"], 6)
        self.assertEqual(kwargs["hazard_parked_n_hi"], 10)
        self.assertEqual(kwargs["hazard_parked_car_weight"], 0.6)
        self.assertEqual(kwargs["hazard_clusters_layout"], "both")
        self.assertEqual(kwargs["hazard_cones_taper_cones"], 8)
        self.assertEqual(kwargs["hazard_cones_spacing_m"], 1.5)

    def test_requires_gigaflow_and_slot_budget(self):
        with self.assertRaisesRegex(ValueError, "gigaflow"):
            Drive(config_only=True, hazard_enable=True, cosim_partner_slots=48, simulation_mode="replay")
        with self.assertRaisesRegex(ValueError, r"cosim_partner_slots >= 42"):
            Drive(config_only=True, hazard_enable=True, cosim_partner_slots=0)
        # planned-count highs: 10 parked + 2*2 cluster cars + 2 zones*14 cones = 42
        env = Drive(config_only=True, hazard_enable=True, cosim_partner_slots=48)
        self.assertEqual(env.hazard_slot_capacity(), 42)
        Drive(config_only=True, hazard_enable=True, cosim_partner_slots=42)  # exact budget passes

    def test_parked_knob_validation(self):
        with self.assertRaisesRegex(ValueError, "hazard_parked_n_lo/hi"):
            Drive(
                config_only=True,
                hazard_enable=True,
                cosim_partner_slots=52,
                hazard_parked_n_lo=8,
                hazard_parked_n_hi=6,
            )
        with self.assertRaisesRegex(ValueError, "hazard_parked_n_lo/hi"):
            Drive(config_only=True, hazard_enable=True, cosim_partner_slots=52, hazard_parked_n_lo=-1)
        with self.assertRaisesRegex(ValueError, "hazard_parked_._weight"):
            Drive(config_only=True, hazard_enable=True, cosim_partner_slots=52, hazard_parked_car_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "hazard_parked_._weight"):
            Drive(
                config_only=True,
                hazard_enable=True,
                cosim_partner_slots=52,
                hazard_parked_car_weight=0.0,
                hazard_parked_bus_weight=0.0,
                hazard_parked_truck_weight=0.0,
            )

    def test_cluster_knob_validation(self):
        with self.assertRaisesRegex(ValueError, "hazard_clusters_n_lo/hi"):
            Drive(
                config_only=True,
                hazard_enable=True,
                cosim_partner_slots=52,
                hazard_clusters_n_lo=2,
                hazard_clusters_n_hi=1,
            )
        with self.assertRaisesRegex(ValueError, "hazard_clusters_layout"):
            Drive(config_only=True, hazard_enable=True, cosim_partner_slots=52, hazard_clusters_layout="side")

    def test_cone_knob_validation(self):
        base = dict(hazard_enable=True, cosim_partner_slots=52)
        with self.assertRaisesRegex(ValueError, "hazard_cones_zones_lo/hi"):
            Drive(config_only=True, hazard_cones_zones_lo=3, hazard_cones_zones_hi=1, **base)
        with self.assertRaisesRegex(ValueError, "hazard_cones_taper_cones"):
            Drive(config_only=True, hazard_cones_taper_cones=1, **base)
        with self.assertRaisesRegex(ValueError, "hazard_cones_block_rows"):
            Drive(config_only=True, hazard_cones_block_rows=0, **base)
        with self.assertRaisesRegex(ValueError, "hazard_cones_block_cols"):
            Drive(config_only=True, hazard_cones_block_cols=1, **base)
        with self.assertRaisesRegex(ValueError, "hazard_cones_spacing_m"):
            Drive(config_only=True, hazard_cones_spacing_m=0.0, **base)
        with self.assertRaisesRegex(ValueError, "hazard_cones_taper_prob"):
            Drive(config_only=True, hazard_cones_taper_prob=1.5, **base)

    def test_walker_knob_validation(self):
        base = dict(hazard_enable=True, hazard_walkers_enable=True, cosim_partner_slots=52)
        with self.assertRaisesRegex(ValueError, "hazard_walkers_n_lo/hi"):
            Drive(config_only=True, hazard_walkers_n_lo=0, hazard_walkers_n_hi=2, **base)
        with self.assertRaisesRegex(ValueError, "hazard_walkers_n_lo/hi"):
            Drive(config_only=True, hazard_walkers_n_lo=3, hazard_walkers_n_hi=2, **base)
        with self.assertRaisesRegex(ValueError, "walker slot cap"):
            Drive(config_only=True, hazard_walkers_n_hi=9, **base)
        with self.assertRaisesRegex(ValueError, "hazard_walkers_speed_lo/hi"):
            Drive(config_only=True, hazard_walkers_speed_lo_mps=0.0, **base)
        with self.assertRaisesRegex(ValueError, "hazard_walkers_speed_lo/hi"):
            Drive(config_only=True, hazard_walkers_speed_lo_mps=1.8, hazard_walkers_speed_hi_mps=1.6, **base)

    def test_walker_slot_budget_contract(self):
        # walkers own the tail slots on top of the K1 head classes: 10 parked +
        # 2*2 cluster cars + 2 zones*14 cones + 2 walkers = 44
        env = Drive(
            config_only=True,
            hazard_enable=True,
            cosim_partner_slots=44,
            hazard_walkers_enable=True,
            hazard_walkers_n_lo=1,
            hazard_walkers_n_hi=2,
        )
        self.assertEqual(env.hazard_slot_capacity(), 44)
        with self.assertRaisesRegex(ValueError, r"cosim_partner_slots >= 44"):
            Drive(
                config_only=True,
                hazard_enable=True,
                cosim_partner_slots=43,
                hazard_walkers_enable=True,
                hazard_walkers_n_hi=2,
            )

    def test_walker_defaults_off_and_kwargs_reach_binding(self):
        env = Drive(config_only=True)
        kwargs = env._env_init_kwargs(map_file="x.bin", max_agents=32)
        self.assertEqual(kwargs["hazard_walkers_enable"], 0)
        self.assertEqual(kwargs["hazard_walkers_n_lo"], 1)
        self.assertEqual(kwargs["hazard_walkers_n_hi"], 2)
        self.assertEqual(kwargs["hazard_walkers_speed_lo_mps"], 1.2)
        self.assertEqual(kwargs["hazard_walkers_speed_hi_mps"], 1.6)

    def test_binding_rejects_bad_config_at_env_init(self):
        # Defense-in-depth: even if Python-side validation were bypassed, the C
        # binding must fail-fast on malformed hazard config before the env runs.
        base = dict(
            map_dir="pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin",
            num_maps=1,
            num_agents=32,
            min_agents_per_env=32,
            max_agents_per_env=32,
            scenario_length=100,
            simulation_mode="gigaflow",
            cosim_partner_slots=48,
            hazard_enable=True,
        )
        with self.assertRaises(Exception):
            Drive(**base, hazard_parked_car_weight=-0.5)
        with self.assertRaises(Exception):
            Drive(**base, hazard_clusters_layout="side_swipe")
        with self.assertRaises(Exception):
            Drive(**base, hazard_walkers_enable=True, hazard_walkers_n_lo=2, hazard_walkers_n_hi=1)
        with self.assertRaises(Exception):
            Drive(**base, hazard_walkers_enable=True, hazard_walkers_n_hi=99)


if __name__ == "__main__":
    unittest.main()
