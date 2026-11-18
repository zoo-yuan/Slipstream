#!/usr/bin/env python3
"""Audit per-5m-cell entity density of the vegas city bin road elements (gate-2 follow-up).

Answers: of the over-cap cells (e.g. cell 9398=276 entities), how many are lanes
(non-droppable graph nodes) vs non-lane (boundary3 / line / crosswalk, droppable)?
Pure numpy rasterization approximating the kernel's grid (5 m cells, bbox + diagonal
margin) -- an audit tool, not the loader.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.val14_build_city_bins import gpkg_extra_roads, gpkg_lanes, load_gpkg_layers

CELL = 5.0


def cell_span(xy):
    ix = (xy[:, 0] // CELL).astype(int)
    iy = (xy[:, 1] // CELL).astype(int)
    return ix.min(), ix.max(), iy.min(), iy.max()


def main(maps_root, map_name):
    layers = load_gpkg_layers(maps_root, "nuplan-maps-v1.0", map_name)
    lanes, _, _ = gpkg_lanes(layers)
    extra = gpkg_extra_roads(layers)
    per_cell = {}
    for kind, items in (("lane", lanes), ("extra", extra)):
        for it in items:
            xy = np.asarray(it["xy"], dtype=np.float64)
            x0, x1, y0, y1 = cell_span(xy)
            for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    key = (ix, iy)
                    d = per_cell.setdefault(key, Counter())
                    d[kind] += 1
    total = {k: sum(c[k] for c in per_cell.values()) for k in ("lane", "extra")}
    worst = sorted(per_cell.items(), key=lambda kv: -(kv[1]["lane"] + kv[1]["extra"]))[:12]
    print(f"entities: lanes={len(lanes)} extra={len(extra)} (of which types: "
          f"{Counter(e['kind'] for e in extra).most_common()})")
    print(f"cells populated: {len(per_cell)}")
    over64 = [(k, c) for k, c in per_cell.items() if c["lane"] + c["extra"] > 64]
    print(f"cells over 64: {len(over64)}; over 256: {sum(1 for _, c in per_cell.items() if c['lane']+c['extra']>256)}")
    lane_only_over = [(k, c) for k, c in per_cell.items() if c["lane"] > 64]
    print(f"cells with lanes alone > 64 (unsolvable by thinning extras): {len(lane_only_over)}")
    for k, c in lane_only_over[:5]:
        print("   LANE-HEAVY", k, dict(c))
    print("worst cells (lane+extra):")
    for k, c in worst:
        print(f"   cell {k}: lane={c['lane']} extra={c['extra']} total={c['lane']+c['extra']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
