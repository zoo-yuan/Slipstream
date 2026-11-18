#!/usr/bin/env python3
"""Build whole-city map-only PufferDrive bins (nuplan__<map>.bin) from a nuPlan GPKG map.

Gate-2 P1a of the val14 zero-shot campaign (.scratch/proposals/val14-zeroshot-proposal.md r1):
the deployed upstream city bins are unreachable, and nuplan_bridge.write_drive_bin wrote an
empty lane graph, which silently zeroes the GPS goal-lane-distance obs columns. This script
rebuilds city bins with a lane graph per the 123Drive semantics documented in
pufferlib/ocean/cosim/nuplan/README.md: "directed scipy Dijkstra over lane entry/exit
connectivity" (Vegas 4583 lanes = 2039 lanes_polygons + 2544 lane_connectors baseline paths,
matching the README's 84 MB n x n float32 matrix).

Bin-frame contract (consumers pin it, see tests/test_val14_bins.py):
  - lanes: every baseline_paths row (lane or lane_connector), road type LANE_SURFACE_STREET,
    per-lane speed_limit from the GPKG (speed_limit_mps), -1.0 where the map carries none
    (the kernel's designed 15 m/s fallback, drive/constants.h:319);
  - entry/exit lanes: element INDICES via the connector layer (lane ->connector-> lane);
  - lane graph: one node per lane element, edge into a lane costing that lane's full arc
    length, directed shortest paths, +inf for unreachable, FILE layout row=from / column=to
    (drive/map_data.h:1090 transposes at load so drive.h:5999 reads [goal * n + from]);
  - no centroid is stored (like the deployed bins): the planner registers the origin against
    the GPKG at first use (_city_bin_origin, residual < 0.5 m hard stop) and caches a
    <bin>.origin.json sidecar.

Boundary types are data-verified, not enum-guessed: in the vegas GPKG every boundary
referenced by a lane's left/right_boundary_fid is type 0 or 2 (lane markings) and type 3 is
never referenced (physical road edge), so 3 -> ROAD_EDGE_BOUNDARY (drives boundary obs rows
and the offroad geometry) and 0/2 -> ROAD_LINE_UNKNOWN (inert in the current kernel: road
lines are neither grid candidates nor observed, datatypes.h:25 -- kept for map fidelity).

Pure layer (no GPKG/devkit imports) is testable without geopandas; the GPKG layer is a thin
devkit GPKGMapsDB reader, validated by the real build stats plus the planner registration.

Usage (lab42):
  PYTHONPATH=<repo> .venv/bin/python scripts/val14_build_city_bins.py \
      --map us-nv-las-vegas-strip --maps-root $NUPLAN_MAPS_ROOT \
      --map-version nuplan-maps-v1.0 --output-dir artifacts/val14/bins
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pufferlib.ocean.cosim import nuplan_bridge as nb
from pufferlib.ocean.cosim.nuplan_bridge import LANE_SURFACE_STREET, ROAD_EDGE_BOUNDARY, TRAFFIC_TYPE_LIGHT

ROAD_LINE_UNKNOWN = 10  # PufferDrive road-line decade, drive/constants.h
MISC_CROSSWALK = 31
TRAFFIC_TYPE_STOP_SIGN = 2
TRAFFIC_TYPE_YIELD_SIGN = 3
LANE_SPEED_LIMIT_MISSING = -1.0
MIN_LANE_WIDTH_M = 0.5
MAX_LANE_WIDTH_M = 50.0  # drive/constants.h MAX_LANE_WIDTH_M; the loader rejects wider
FALLBACK_LANE_WIDTH_M = 3.7  # drive/constants.h LANE_WIDTH, the no-widths-section default

# GPKG stop_polygon_type_fid (devkit StopLineType) -> PufferDrive traffic control type.
# TURN_STOP zones export as stop signs, matching the planner config's comment on them.
STOP_POLYGON_TYPE_TO_TRAFFIC = {1: TRAFFIC_TYPE_STOP_SIGN, 2: TRAFFIC_TYPE_LIGHT, 3: TRAFFIC_TYPE_STOP_SIGN, 4: TRAFFIC_TYPE_YIELD_SIGN}


# ---------------------------------------------------------------------------
# Pure layer: hand-buildable micro maps and the lane graph (tests live here)
# ---------------------------------------------------------------------------
def _lane_lengths(xy: np.ndarray) -> float:
    """Arc length of an (N, 2) polyline."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return 0.0
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def build_lane_graph(lanes, node_ids=None):
    """Directed shortest-path lane graph in the bin's FILE layout.

    lanes: list of dicts {key, xy, exits: [downstream keys], ...} in graph order;
    node_ids: the road-ELEMENT index of each lane (defaults to range(n), i.e. lanes
    occupying the first road slots) -- lane_graph.lane_ids in the bin are these
    subscripts, not graph positions and not map ids.
    Returns (lane_ids, distances):
      lane_ids  tuple of element indices, one per lane, in `lanes` order;
      distances row-major n x n float32, row = from, column = to, the directed
                shortest-path cost (entering a lane costs its arc length), +inf
                where unreachable, 0 on the diagonal.

    map_data.h:1090 ("File stores [from][to]") transposes this matrix at load time so
    drive.h:5999 reads one contiguous goal row per ego: distances[goal * n + from].
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    index = {lane["key"]: i for i, lane in enumerate(lanes)}
    rows, cols, weights = [], [], []
    for i, lane in enumerate(lanes):
        for out_key in lane.get("exits", []):
            j = index[out_key]
            rows.append(i)
            cols.append(j)
            weights.append(_lane_lengths(lanes[j]["xy"]))  # entering j costs j's arc length
    n = len(lanes)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    distances = dijkstra(graph, directed=True, indices=None)
    ids = tuple(range(n)) if node_ids is None else tuple(int(v) for v in node_ids)
    return ids, np.asarray(distances, dtype=np.float32)


def _traffic_entry(traffic_id, traffic_type, stop_line_xy, heading, controlled_lanes=()):
    """One traffic element in mirror_map_bin's dict schema (map-only: no state program)."""
    (x1, y1), (x2, y2) = stop_line_xy
    return {
        "id": traffic_id,
        "type": int(traffic_type),
        "stop_line": (float(x1), float(y1), 0.0, float(x2), float(y2), 0.0),
        "heading": float(heading),
        "states": (),
        "controlled_lanes": tuple(int(c) for c in controlled_lanes),
    }


def decimate_polyline(xy, min_seg_m=5.0, turn_deg=15.0):
    """Vertex-decimate a polyline: drop interior points closer than `min_seg_m` to the
    kept predecessor while the local turn is under `turn_deg` degrees. Endpoints kept.

    Gate-2 follow-up: the kernel's grid counts SEGMENTS-per-5m-cell (map_data.h
    init_grid_map: one count per segment, binned at its midpoint), and GPKG baselines
    carry ~1 m vertices, so whole-city bins stack hundreds of segments per cell at
    interchanges (vegas smoke: 276 > 64). CARLA town bins never hit this because their
    polylines are vertex-sparse. Decimating to >=5 m segments (with turn preservation)
    matches the CARLA bin vertex density semantics and loses no obs fidelity at the
    kernel's 5 m cell resolution.
    Pure function; returns a new (n, 2) float64 array with n <= len(xy).
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) <= 2:
        return xy
    keep = [xy[0]]
    i = 0
    for j in range(1, len(xy) - 1):
        seg = xy[j] - keep[-1]
        nxt = xy[j + 1] - xy[j]
        seg_len = float(np.hypot(*seg))
        nxt_len = float(np.hypot(*nxt))
        turn = 0.0
        if seg_len > 1e-9 and nxt_len > 1e-9:
            cosang = float(np.clip(np.dot(seg, nxt) / (seg_len * nxt_len), -1.0, 1.0))
            turn = math.degrees(math.acos(cosang))
        if seg_len >= min_seg_m or turn >= turn_deg:
            keep.append(xy[j])
    keep.append(xy[-1])
    return np.asarray(keep)


def cell_segment_load(road_elements, cell_m=5.0):
    """Kernel-true per-cell segment load (audit/thinning helper): count every
    consecutive segment into the 5 m cell of its midpoint, exactly like
    map_data.h init_grid_map's first pass (is_road_grid_candidate types only).
    Returns {(cx, cy): count} over candidate elements."""
    from pufferlib.ocean.cosim.nuplan_bridge import LANE_SURFACE_STREET as _LANE_T
    from pufferlib.ocean.cosim.nuplan_bridge import ROAD_EDGE_BOUNDARY as _EDGE_T

    load = {}
    for e in road_elements:
        t = _LANE_T if "exits" in e else int(e.get("type", -1))
        if not (t == _LANE_T or t == _EDGE_T):  # datatypes.h: lanes + edges only
            continue
        xy = np.asarray(e["xy"], dtype=np.float64).reshape(-1, 2)
        mid = (xy[:-1] + xy[1:]) / 2.0
        cs = np.floor(mid / cell_m).astype(int)
        for cx, cy in zip(cs[:, 0], cs[:, 1]):
            load[(int(cx), int(cy))] = load.get((int(cx), int(cy)), 0) + 1
    return load


def thin_extras_to_cell_cap(road_elements, cap=64, cell_m=5.0, max_rounds=10):
    """Second-stage guarantee: after decimation, any cell still over `cap` loses
    whole NON-LANE entities (boundaries/lines/crosswalks -- lanes are graph nodes and
    never dropped), highest per-cell segment contributor first. Returns (elements,
    stats). Raises if a cell is over cap with lanes alone (unsolvable by thinning).
    Mirrors the CARLA bins' effective sparsity semantics: dropping a redundant
    physical-edge polyline costs one boundary obs row there, like a sparser map."""
    from pufferlib.ocean.cosim.nuplan_bridge import LANE_SURFACE_STREET as _LANE_T, ROAD_EDGE_BOUNDARY as _EDGE_T

    els = [dict(e) for e in road_elements]
    dropped = {"total": 0}
    for _ in range(max_rounds):
        load = cell_segment_load(els, cell_m)
        hot = {c: n for c, n in load.items() if n > cap}
        if not hot:
            return els, {"dropped": dropped, "max_cell_load": max(load.values()) if load else 0}
        # segment-contributions of each non-lane element per hot cell
        contrib = {}
        for idx, e in enumerate(els):
            if "exits" in e:
                continue
            t = int(e.get("type", -1))
            if not (t == _EDGE_T or t == _LANE_T):
                continue
            xy = np.asarray(e["xy"], dtype=np.float64).reshape(-1, 2)
            mid = (xy[:-1] + xy[1:]) / 2.0
            cs = np.floor(mid / cell_m).astype(int)
            for cx, cy in zip(cs[:, 0], cs[:, 1]):
                if (int(cx), int(cy)) in hot:
                    contrib[idx] = contrib.get(idx, 0) + 1
        if not contrib:
            lane_load = cell_segment_load([e for e in els if "exits" in e], cell_m)
            worst = max(lane_load.values()) if lane_load else 0
            raise ValueError(
                f"cell cap {cap} exceeded with lanes alone (worst lane-only cell={worst}); "
                "thinning cannot fix this -- reduce lane count or raise the cap consciously")
        victim = max(contrib, key=lambda k: contrib[k])
        kind = els[victim].get("kind", els[victim].get("type"))
        dropped[str(kind)] = dropped.get(str(kind), 0) + 1
        dropped["total"] += 1
        del els[victim]
    raise RuntimeError("thin_extras_to_cell_cap did not converge")


def build_city_bin(road_elements, traffic, out_path, scenario_id, origin=None, decimate=True,
                   cell_cap=64):
    """Assemble and write one city bin from pure-layer inputs.

    road_elements: ordered list; each entry is either a lane ("key", "xy", "speed_limit",
    "exits" [downstream lane keys], optional "width") or a plain road ("type", "xy").
    The element index is the list position, and lane_graph.lane_ids are exactly those
    positions of the lane entries (subscripts, never graph positions or map ids).
    traffic: entries from _traffic_entry (controlled_lanes use element indices).
    Coordinates are translated by `origin` (the lane-point mean when None) to keep the
    float32 bin frame near zero; the origin is deliberately NOT stored (the planner
    registers it against the GPKG and caches a sidecar, like the deployed bins).
    """
    lane_positions = [i for i, e in enumerate(road_elements) if "exits" in e]
    if decimate:
        for e in road_elements:
            e["xy"] = decimate_polyline(e["xy"])
    if cell_cap:
        road_elements, thin_stats = thin_extras_to_cell_cap(road_elements, cap=cell_cap)
        print(f"[thin] cap={cell_cap} stats={thin_stats}")
        lane_positions = [i for i, e in enumerate(road_elements) if "exits" in e]
    if origin is None:
        origin = np.concatenate(
            [np.asarray(road_elements[i]["xy"], dtype=np.float64).reshape(-1, 2) for i in lane_positions]
        ).mean(axis=0)
    origin = np.asarray(origin, dtype=np.float64).reshape(2)

    roads = []
    for i, element in enumerate(road_elements):
        xy = np.asarray(element["xy"], dtype=np.float64).reshape(-1, 2) - origin
        if "exits" not in element:
            roads.append(nb._road_entry(i, int(element["type"]), xy))
            continue
        position_of_key = {road_elements[j]["key"]: j for j in lane_positions}
        upstream = [j for j in lane_positions if element["key"] in road_elements[j].get("exits", [])]
        width = float(element.get("width", FALLBACK_LANE_WIDTH_M))
        width = min(max(width, MIN_LANE_WIDTH_M), MAX_LANE_WIDTH_M)
        entry = nb._road_entry(
            i,
            LANE_SURFACE_STREET,
            xy,
            entry=sorted(upstream),
            exit_=sorted(position_of_key[k] for k in element.get("exits", [])),
            speed_limit=(
                float(element["speed_limit"]) if element.get("speed_limit") is not None else LANE_SPEED_LIMIT_MISSING
            ),
        )
        entry["widths"] = tuple([width] * len(xy))
        roads.append(entry)

    graph_lanes = [dict(road_elements[i]) for i in lane_positions]
    lane_ids, distances = build_lane_graph(graph_lanes, node_ids=lane_positions)
    traffic = [dict(t, id=i) for i, t in enumerate(traffic)]
    nb.write_drive_bin(
        roads,
        traffic,
        Path(out_path),
        scenario_id=str(scenario_id),
        lane_graph={
            "n": len(lane_ids),
            "lane_ids": lane_ids,
            "distances": tuple(np.asarray(distances).ravel().tolist()),
        },
    )
    return {
        "num_roads": len(roads),
        "num_lanes": len(lane_ids),
        "num_traffic": len(traffic),
        "origin": [float(origin[0]), float(origin[1])],
    }


# ---------------------------------------------------------------------------
# GPKG layer: nuPlan vector layers -> the pure layer's inputs (devkit reader)
# ---------------------------------------------------------------------------
def _fid_column(df):
    """GPKGMapsDB.load_vector_layer casts fids to strings; normalize to int64."""
    return df["fid"].astype("int64")


def load_gpkg_layers(maps_root, map_version, map_name):
    """Whole-city vector layers through the devkit's GPKGMapsDB (UTM GeoDataFrames)."""
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

    maps_db = GPKGMapsDB(map_root=str(maps_root), map_version=str(map_version))
    return {layer: maps_db.load_vector_layer(map_name, layer) for layer in
            ("baseline_paths", "lanes_polygons", "lane_connectors", "boundaries", "stop_polygons", "crosswalks")}


def gpkg_lanes(layers):
    """baseline_paths rows (lanes + lane_connectors) -> pure-layer lane dicts.

    Speed limits join lane_fid -> lanes_polygons and lane_connector_fid -> lane_connectors
    (both carry speed_limit_mps); a missing join or NaN writes -1.0 (missing). Widths come
    from the lane polygon area over its baseline length (the width column is a constant 12
    everywhere in vegas -- feet, not meters, and useless); connectors take their entry
    lane's width. Connectivity: connector elements sit between their entry and exit lanes.
    """
    baselines = layers["baseline_paths"]
    lanes_poly = layers["lanes_polygons"]
    connectors = layers["lane_connectors"]

    lane_limit = dict(zip(_fid_column(lanes_poly), lanes_poly["speed_limit_mps"].astype("float64")))
    conn_limit = dict(zip(_fid_column(connectors), connectors["speed_limit_mps"].astype("float64")))
    lane_width = {}
    for fid, width in zip(_fid_column(lanes_poly), lanes_poly.geometry.area / lanes_poly.geometry.length):
        lane_width[fid] = float(width) if np.isfinite(width) else FALLBACK_LANE_WIDTH_M
    entry_of_conn = {int(r.fid): int(r.entry_lane_fid) for r in connectors.itertuples() if np.isfinite(r.entry_lane_fid)}
    exit_of_conn = {int(r.fid): int(r.exit_lane_fid) for r in connectors.itertuples() if np.isfinite(r.exit_lane_fid)}

    lanes = []
    lane_index_by_fid = {}
    conn_index_by_fid = {}
    for row in baselines.itertuples():
        key = f"bp{row.Index}"
        lane_fid = row.lane_fid if hasattr(row, "lane_fid") else None
        conn_fid = row.lane_connector_fid if hasattr(row, "lane_connector_fid") else None
        xy = np.asarray(row.geometry.coords, dtype=np.float64)
        if conn_fid is not None and not (isinstance(conn_fid, float) and np.isnan(conn_fid)):
            fid = int(conn_fid)
            conn_index_by_fid[fid] = len(lanes)
            limit = conn_limit.get(fid)
            width = lane_width.get(entry_of_conn.get(fid, -1), FALLBACK_LANE_WIDTH_M)
            lanes.append({"key": key, "fid": ("conn", fid), "xy": xy,
                          "speed_limit": None if limit is None or not np.isfinite(limit) else float(limit),
                          "width": width, "exits": [], "entries": []})
        elif lane_fid is not None and not (isinstance(lane_fid, float) and np.isnan(lane_fid)):
            fid = int(lane_fid)
            lane_index_by_fid[fid] = len(lanes)
            limit = lane_limit.get(fid)
            lanes.append({"key": key, "fid": ("lane", fid), "xy": xy,
                          "speed_limit": None if limit is None or not np.isfinite(limit) else float(limit),
                          "width": lane_width.get(fid, FALLBACK_LANE_WIDTH_M), "exits": [], "entries": []})
        else:
            raise ValueError(f"baseline_paths row {row.Index} carries neither lane_fid nor lane_connector_fid")

    # Connectivity: lane -> its connectors -> the downstream lane, connector -> both lanes.
    for conn_fid, entry_lane_fid in entry_of_conn.items():
        if conn_fid not in conn_index_by_fid or entry_lane_fid not in lane_index_by_fid:
            continue
        conn = lanes[conn_index_by_fid[conn_fid]]
        entry = lanes[lane_index_by_fid[entry_lane_fid]]
        entry["exits"].append(conn["key"])
        conn["entries"].append(entry["key"])
    for conn_fid, exit_lane_fid in exit_of_conn.items():
        if conn_fid not in conn_index_by_fid or exit_lane_fid not in lane_index_by_fid:
            continue
        conn = lanes[conn_index_by_fid[conn_fid]]
        exit_ = lanes[lane_index_by_fid[exit_lane_fid]]
        conn["exits"].append(exit_["key"])
        exit_["entries"].append(conn["key"])

    # Pure layer takes `exits` only; build_city_bin derives `entries` from exits.
    for lane in lanes:
        lane["exits"] = sorted(set(lane["exits"]))
    return lanes, lane_index_by_fid, conn_index_by_fid


def gpkg_extra_roads(layers):
    """Boundaries and crosswalks -> non-lane road elements.

    Data-verified boundary split (vegas): types 0/2 are the lane-referenced markings
    (inert as ROAD_LINE_UNKNOWN in this kernel), type 3 is never lane-referenced and is
    the physical road edge (ROAD_EDGE_BOUNDARY: boundary obs + offroad geometry).
    Crosswalk polygons export as their exterior ring (MISC_CROSSWALK).
    """
    extra = []
    boundaries = layers["boundaries"]
    for row in boundaries.itertuples():
        btype = int(row.boundary_type_fid)
        road_type = ROAD_EDGE_BOUNDARY if btype == 3 else ROAD_LINE_UNKNOWN
        xy = np.asarray(row.geometry.coords, dtype=np.float64)
        extra.append({"xy": xy, "type": road_type, "kind": f"boundary_type_{btype}"})
    for row in layers["crosswalks"].itertuples():
        ring = row.geometry.exterior
        if ring is None:
            continue
        extra.append({"xy": np.asarray(ring.coords, dtype=np.float64), "type": MISC_CROSSWALK, "kind": "crosswalk"})
    return extra


def _stop_line_segment_and_heading(polygon, approach_xy=None):
    """Stop polygon -> ((x1,y1),(x2,y2)) of the painted line + the travel heading.

    The painted line is the LONG edge of the polygon's minimum rotated rectangle (a stop
    polygon is the thin quadrilateral across the approach); the heading is the travel
    direction INTO the line, taken from the approach lane's baseline heading at its nearest
    point, else the rectangle edge normal's direction along the polygon's short side.
    """
    rect = polygon.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords, dtype=np.float64)
    edges = np.hypot(*np.diff(coords, axis=0).T)
    long_i = int(np.argmax(edges[:4]))
    p1, p2 = coords[long_i], coords[long_i + 1]
    if approach_xy is not None and len(approach_xy) >= 2:
        pts = np.asarray(approach_xy, dtype=np.float64)
        mid = 0.5 * (p1 + p2)
        k = int(np.argmin(np.hypot(*(pts - mid).T)))
        k2 = min(k + 1, len(pts) - 1)
        heading = float(np.arctan2(pts[k2, 1] - pts[k, 1], pts[k2, 0] - pts[k, 0]))
    else:
        # fall back to the perpendicular of the long edge, pointing into the rectangle
        p3 = coords[(long_i + 2) % 4]
        normal = p3 - 0.5 * (p1 + p2)
        heading = float(np.arctan2(normal[1], normal[0]))
    return (p1, p2), heading


def gpkg_traffic(layers, lanes, lane_index_by_fid, conn_index_by_fid):
    """stop_polygons -> traffic elements; controlled lanes are approach lane element indices."""
    baselines = layers["baseline_paths"]
    approach_xy = {}
    for row in baselines.itertuples():
        lane_fid = getattr(row, "lane_fid", None)
        conn_fid = getattr(row, "lane_connector_fid", None)
        if lane_fid is not None and not (isinstance(lane_fid, float) and np.isnan(lane_fid)):
            approach_xy[("lane", int(lane_fid))] = np.asarray(row.geometry.coords, dtype=np.float64)
        elif conn_fid is not None and not (isinstance(conn_fid, float) and np.isnan(conn_fid)):
            approach_xy[("conn", int(conn_fid))] = np.asarray(row.geometry.coords, dtype=np.float64)

    traffic = []
    stats = {"stop_types": {}, "no_approach": 0}
    for row in layers["stop_polygons"].itertuples():
        stop_type = int(row.stop_polygon_type_fid)
        traffic_type = STOP_POLYGON_TYPE_TO_TRAFFIC.get(stop_type)
        if traffic_type is None:  # 0 = PED_CROSSING: crosswalks export as road elements instead
            stats["stop_types"][f"skipped_{stop_type}"] = stats["stop_types"].get(f"skipped_{stop_type}", 0) + 1
            continue
        controlled = []
        for col, table in (("lane_fids", lane_index_by_fid), ("lane_connector_fids", conn_index_by_fid)):
            raw = getattr(row, col, None)
            if raw is None or not isinstance(raw, str) or not raw.strip():
                continue
            for fid_s in raw.split(","):
                try:
                    fid = int(fid_s)
                except ValueError:
                    continue
                if fid in table:
                    controlled.append(table[fid])
        approach = next((approach_xy[lanes[k]["fid"]] for k in controlled if lanes[k]["fid"] in approach_xy), None)
        if approach is None:
            stats["no_approach"] += 1
        (p1, p2), heading = _stop_line_segment_and_heading(row.geometry, approach)
        traffic.append(_traffic_entry(len(traffic), traffic_type, (p1, p2), heading, controlled_lanes=controlled))
        stats["stop_types"][stop_type] = stats["stop_types"].get(stop_type, 0) + 1
    return traffic, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="map name, e.g. us-nv-las-vegas-strip")
    parser.add_argument("--maps-root", required=True, help="NUPLAN_MAPS_ROOT (holds nuplan-maps-v1.0.json)")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fingerprint", default=None, help="where to write the build fingerprint json (default: <bin>.fingerprint.json)")
    args = parser.parse_args()

    t0 = time.time()
    layers = load_gpkg_layers(args.maps_root, args.map_version, args.map)
    print(f"[{time.time()-t0:.1f}s] layers loaded: " + ", ".join(f"{k}={len(v)}" for k, v in layers.items()))

    lanes, lane_index_by_fid, conn_index_by_fid = gpkg_lanes(layers)
    extra_roads = gpkg_extra_roads(layers)
    traffic, stop_stats = gpkg_traffic(layers, lanes, lane_index_by_fid, conn_index_by_fid)

    with_limits = sum(1 for l in lanes if l["speed_limit"] is not None)
    print(
        f"lanes={len(lanes)} (lane rows={len(lane_index_by_fid)}, connector rows={len(conn_index_by_fid)}) "
        f"speed_limit: {with_limits}/{len(lanes)} from gpkg, {len(lanes)-with_limits} -> -1"
    )
    hist = {}
    for l in lanes:
        v = round(l["speed_limit"], 1) if l["speed_limit"] is not None else -1.0
        hist[v] = hist.get(v, 0) + 1
    print(f"speed_limit histogram (m/s, -1=missing): {dict(sorted(hist.items()))}")
    orphans = sum(1 for l in lanes if not l["exits"])
    print(f"lanes with no exit: {orphans}/{len(lanes)}; traffic elements: {len(traffic)} (stop types {stop_stats})")
    edge_types = {}
    for r in extra_roads:
        edge_types[r["kind"]] = edge_types.get(r["kind"], 0) + 1
    print(f"non-lane roads: {edge_types}")

    all_points = np.concatenate([np.asarray(l["xy"], dtype=np.float64) for l in lanes])
    origin = all_points.mean(axis=0)
    out_path = Path(args.output_dir) / f"nuplan__{args.map}.bin"
    elements = list(lanes) + [{"type": r["type"], "xy": r["xy"]} for r in extra_roads]
    info = build_city_bin(elements, traffic, out_path, scenario_id=f"nuplan__{args.map}", origin=origin)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[{time.time()-t0:.1f}s] wrote {out_path} ({size_mb:.1f} MB), origin=({origin[0]:.1f}, {origin[1]:.1f}) not stored")

    import data_utils.mirror_map_bin as mbin
    check = mbin.read_bin(out_path)
    assert [r["id"] for r in check["roads"]] == list(range(len(check["roads"])))
    n = check["lane_graph"]["n"]
    assert n == len(lanes), f"lane graph n={n} != lanes {len(lanes)}"
    assert max(check["lane_graph"]["lane_ids"]) < len(check["roads"])
    d = np.array(check["lane_graph"]["distances"], dtype=np.float32).reshape(n, n)
    finite = np.isfinite(d)
    print(f"read-back ok: roads={len(check['roads'])} graph n={n} "
          f"finite pairs={int(finite.sum())}/{n*n} ({finite.sum()/n/n:.1%}) diag0={bool((np.diag(d)==0).all())}")

    fingerprint = {
        "map": args.map,
        "map_version": args.map_version,
        "bin": str(out_path),
        "bin_bytes": out_path.stat().st_size,
        "num_roads": len(check["roads"]),
        "num_lanes": n,
        "num_traffic": len(check["traffic"]),
        "speed_limit_histogram": {str(k): v for k, v in sorted(hist.items())},
        "speed_limit_missing": len(lanes) - with_limits,
        "finite_pair_fraction": float(finite.sum() / (n * n)),
        "origin_mean_xy": [float(origin[0]), float(origin[1])],
        "origin_stored": False,
        "stop_types": {str(k): v for k, v in stop_stats["stop_types"].items()},
        "edge_kinds": edge_types,
        "layers": {k: len(v) for k, v in layers.items()},
        "build_seconds": round(time.time() - t0, 1),
    }
    fp_path = Path(args.fingerprint) if args.fingerprint else out_path.with_suffix(".fingerprint.json")
    fp_path.write_text(json.dumps(fingerprint, indent=1))
    print(f"fingerprint -> {fp_path}")


if __name__ == "__main__":
    main()
