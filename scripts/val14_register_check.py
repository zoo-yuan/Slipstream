#!/usr/bin/env python3
"""P1a exit self-check: register the lab-built city bin against its GPKG (residual < 0.5 m).

Mirrors PufferDrivePlanner._city_bin_origin (pufferlib/ocean/cosim/nuplan/planner.py)
step for step -- coarse stop-line translation vote, then translation-only ICP of the
bin's lane points onto the GPKG baseline paths, residual < 0.5 m hard gate -- so the
only thing skipped is the Scenario/planner initialization (which needs a scenario db).
On success it writes the same `.origin.json` sidecar the planner would, so the real
run later skips registration; on residual > 0.5 m it exits 1 without writing anything.

Usage (lab42):
  PYTHONPATH=<repo> .venv/bin/python scripts/val14_register_check.py \
      --bin artifacts/val14/bins/nuplan__us-nv-las-vegas-strip.bin \
      --map us-nv-las-vegas-strip --maps-root $NUPLAN_MAPS_ROOT
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pufferlib.ocean.cosim import nuplan_bridge as nb

RESIDUAL_LIMIT_M = 0.5  # planner.py hard stop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", required=True, help="nuplan__<map>.bin to register")
    parser.add_argument("--map", required=True, help="map name, e.g. us-nv-las-vegas-strip")
    parser.add_argument("--maps-root", required=True, help="NUPLAN_MAPS_ROOT")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--overwrite", action="store_true", help="re-register even if a sidecar exists")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    sidecar = bin_path.with_suffix(".origin.json")
    if sidecar.exists() and not args.overwrite:
        cached = json.loads(sidecar.read_text())
        print(f"sidecar exists: origin={cached['origin']} residual_m={cached['residual_m']} (skipping; --overwrite to redo)")
        return 0

    t0 = time.time()
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

    maps_db = GPKGMapsDB(map_root=str(args.maps_root), map_version=str(args.map_version))
    stops = maps_db.load_vector_layer(args.map, "stop_polygons").geometry.centroid
    baselines = maps_db.load_vector_layer(args.map, "baseline_paths").geometry

    geo = nb.read_bin_geometry(bin_path)
    if not len(geo["stop_line_centers"]):
        print(f"FAIL: {bin_path} has no traffic elements to register with", file=sys.stderr)
        return 1
    ref_stops = np.stack([stops.x.to_numpy(), stops.y.to_numpy()], axis=1)
    init = nb.coarse_translation_vote(geo["stop_line_centers"], ref_stops)
    import shapely

    lane_utm = shapely.get_coordinates(baselines.values)
    translation, resid = nb.fit_translation(nb.read_bin_lane_points(bin_path), lane_utm, init=init)
    print(f"[{time.time()-t0:.1f}s] vote init=({init[0]:.1f}, {init[1]:.1f}) -> "
          f"origin=({translation[0]:.3f}, {translation[1]:.3f}) residual={resid:.4f} m (limit {RESIDUAL_LIMIT_M})")
    if resid > RESIDUAL_LIMIT_M:
        print(f"FAIL: residual {resid:.3f} m > {RESIDUAL_LIMIT_M} m — bin does not match the GPKG", file=sys.stderr)
        return 1
    payload = json.dumps({"origin": [float(translation[0]), float(translation[1])], "residual_m": float(resid)})
    sidecar.write_text(payload)
    print(f"ok -> {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
