"""val14 whole-city bin builder: semantic-level bin-format tests (gate-2 P1a).

Spec source: .scratch/proposals/val14-zeroshot-proposal.md (r1) + review
.scratch/proposals/val14-zeroshot-review-deepseek.md (blocking item #1): the
five semantic assertions pin the PufferDrive bin contract the C kernel
actually consumes, so a "plausible but wrong" city bin fails here instead of
silently zeroing the GPS goal features:

  (a) road_id == array index round-trip (mirror_map_bin.read_bin already
      asserts id == idx; we additionally check the geometry survives and the
      lane graph block survives a write->read round trip);
  (b) lane_graph lane_ids are ROAD-ELEMENT INDICES (subscripts), not map ids:
      the fixture interleaves a non-lane element so lane_ids must skip it;
  (c) the distance matrix is DIRECTED (file row = from, column = to): a
      one-way street reads from->to finite but to->to unreachable. This is
      the FILE layout: pufferlib/ocean/drive/map_data.h:1090 transposes at
      load time so drive.h:5999 reads one goal row per ego in memory
      (distances[goal * n + from]);
  (d) unreachable pairs encode as +inf (non-finite, never 0 and never
      negative), matching drive.h:6000's INFINITY convention;
  (e) per-lane speed_limit: lanes whose map row carries no limit are written
      as -1.0 (which triggers the kernel's designed 15 m/s fallback,
      constants.h:319), lanes with a limit keep the true value.

TDD contract: this file was written BEFORE scripts/val14_build_city_bins.py
and the write_drive_bin lane_graph parameter existed; its first run is the
red evidence (import/classify failure), kept under artifacts/val14/test-runs/.

The fixture is a hand-built 2-3 lane micro map (no geopandas/nuplan import):
plain coordinates, explicit entry/exit connectivity, so every expected
distance is arithmetic, not a copy of the implementation's output.

Edge-weight convention pinned here (123Drive semantics as documented in
cosim/nuplan/README.md): entering lane j costs j's full arc length, so
dist(from -> to) = sum of the lengths of the lanes strictly after `from`
up to and including `to` (0 on the diagonal).
"""

import importlib.util
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402

_SCRIPT = REPO_ROOT / "scripts" / "val14_build_city_bins.py"
_spec = importlib.util.spec_from_file_location("val14_build_city_bins", _SCRIPT)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

import data_utils.mirror_map_bin as mbin  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: main street A -> B (one-way), side lane C -> A only, plus one road
# edge element interleaved at index 1 so the graph lane_ids must skip it.
#
#   C (50 m, limit 8.9) ----> A (100 m, limit 13.9) ----> B (60 m, NO limit -> -1)
#
# Graph nodes are element indices: A=0, B=2, C=3 -> lane_ids == (0, 2, 3).
# ---------------------------------------------------------------------------
def line(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y1]], dtype=np.float64)


MICRO_LANES = [
    {"key": "A", "xy": line(0, 0, 100, 0), "speed_limit": 13.9, "exits": ["B"], "width": 3.6},
    {"key": "B", "xy": line(100, 0, 160, 0), "speed_limit": None, "exits": [], "width": 3.6},
    {"key": "C", "xy": line(0, 10, 50, 10), "speed_limit": 8.9, "exits": ["A"], "width": 3.2},
]
MICRO_EDGE = {"xy": line(0, -5, 40, -5), "type": 21}  # ROAD_EDGE_BOUNDARY

EXPECTED_LANE_IDS = (0, 2, 3)  # A=0, B=2, C=3 (index 1 is the road edge)


@pytest.fixture()
def micro_bin(tmp_path):
    """Build the micro map through the builder's pure layer and write a bin.

    The road edge sits at index 1, between lanes A and B, so element indices are
    A=0, edge=1, B=2, C=3 -- lane_ids must carry exactly those subscripts.
    """
    out = tmp_path / "nuplan__micro.bin"
    elements = [dict(MICRO_LANES[0]), dict(MICRO_EDGE), dict(MICRO_LANES[1]), dict(MICRO_LANES[2])]
    builder.build_city_bin(elements, traffic=[], out_path=out, scenario_id="micro", origin=(0.0, 0.0))
    return out


def test_roundtrip_ids_geometry_and_graph(micro_bin):
    """(a) road_id == index and the geometry/graph survive write -> read."""
    data = mbin.read_bin(micro_bin)
    assert [r["id"] for r in data["roads"]] == list(range(len(data["roads"])))
    assert data["lane_graph"]["n"] == 3
    assert len(data["lane_graph"]["distances"]) == 9

    lane_a = data["roads"][0]
    np.testing.assert_allclose(lane_a["x"], (0.0, 100.0), atol=1e-4)
    np.testing.assert_allclose(lane_a["y"], (0.0, 0.0), atol=1e-4)
    assert lane_a["type"] == builder.LANE_SURFACE_STREET
    # headings of a straight +x polyline are 0
    np.testing.assert_allclose(lane_a["headings"], (0.0, 0.0), atol=1e-6)


def test_lane_ids_are_element_indices_not_map_ids(micro_bin):
    """(b) lane_graph.lane_ids are subscripts into the roads array."""
    data = mbin.read_bin(micro_bin)
    assert tuple(data["lane_graph"]["lane_ids"]) == EXPECTED_LANE_IDS
    for lane_idx in data["lane_graph"]["lane_ids"]:
        assert 0 <= lane_idx < len(data["roads"])
        assert 0 <= data["roads"][lane_idx]["type"] <= 9  # a lane decade element


def _distances(micro_bin):
    data = mbin.read_bin(micro_bin)
    n = data["lane_graph"]["n"]
    return np.array(data["lane_graph"]["distances"], dtype=np.float32).reshape(n, n)


def test_distance_matrix_is_directed_file_layout(micro_bin):
    """(c) file row = from, column = to; one-way A->B, C->A only."""
    d = _distances(micro_bin)
    a, b, c = 0, 1, 2  # graph positions of element indices 0, 2, 3
    # A -> B reachable, B -> A unreachable (one-way street)
    assert d[a, b] == pytest.approx(60.0, abs=1e-3)
    assert not math.isfinite(d[b, a])
    # entering a lane costs its full length: C -> A -> B = 100 + 60
    assert d[c, b] == pytest.approx(160.0, abs=1e-3)
    assert d[c, a] == pytest.approx(100.0, abs=1e-3)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=0.0)


def test_unreachable_pairs_are_nonfinite_infinity(micro_bin):
    """(d) unreachable = +inf (never 0, never negative), drive.h:6000 convention."""
    d = _distances(micro_bin)
    a, b, c = 0, 1, 2
    for unreachable in (d[a, c], d[b, a], d[b, c]):
        assert math.isinf(unreachable) and unreachable > 0
    finite = d[np.isfinite(d)]
    assert (finite >= 0).all()


def test_loader_transpose_goal_row_semantics(micro_bin):
    """drive.h:5999 in-memory read [goal * n + from] after map_data.h:1090's load-time
    transpose must give the forward from->goal travel distance."""
    file_d = _distances(micro_bin)
    n = file_d.shape[0]
    mem = file_d.T.copy()  # the loader swaps [from][to] -> [goal][from]

    def lane_graph_distance_to_goal(from_lane, goal_lane):
        return float(mem[goal_lane, from_lane])

    b, c = 1, 2
    assert lane_graph_distance_to_goal(c, b) == pytest.approx(160.0, abs=1e-3)
    assert math.isinf(lane_graph_distance_to_goal(b, c))


def test_speed_limit_semantics(micro_bin):
    """(e) true limits kept; missing limits written as -1.0 (designed 15 m/s fallback)."""
    data = mbin.read_bin(micro_bin)
    assert data["roads"][0]["speed_limit"] == pytest.approx(13.9)  # lane A
    assert data["roads"][2]["speed_limit"] == -1.0  # lane B: no map limit
    assert data["roads"][3]["speed_limit"] == pytest.approx(8.9)  # lane C
    # entry/exit connectivity is written as element indices
    assert tuple(data["roads"][0]["exit_lanes"]) == (2,)  # A -> B
    assert tuple(data["roads"][0]["entry_lanes"]) == (3,)  # C -> A
    assert tuple(data["roads"][3]["exit_lanes"]) == (0,)  # C -> A


def test_bin_bytes_match_reference_writer(micro_bin):
    """The builder writes through nuplan_bridge.write_drive_bin (single spelling of
    the format): its lane_graph argument must be honored, and a lane graph of n=0
    (the legacy call) still writes an empty graph block."""
    data = mbin.read_bin(micro_bin)
    assert data["scenario_id"].rstrip(b"\0") == b"micro"
    assert data["dataset_name"].rstrip(b"\0") == b"nuplan"
    # legacy call shape keeps working (empty graph, e.g. scenario bins)
    legacy = micro_bin.with_name("legacy.bin")
    nb.write_drive_bin(roads=data["roads"], traffic=data["traffic"], out_path=legacy, scenario_id="legacy")
    legacy_data = mbin.read_bin(legacy)
    assert legacy_data["lane_graph"]["n"] == 0
    assert legacy_data["lane_graph"]["distances"] == ()


def _dense_interchange(n_lines=80, points_per_line=120, span=8.0):
    """80 dense polylines crossing one 5 m cell (vegas-interchange-like load)."""
    els = []
    for k in range(n_lines):
        t = np.linspace(0.0, span, points_per_line)
        xy = np.stack([t, 0.02 * np.sin(3.0 * t) + 0.001 * k], axis=1)
        els.append({"xy": xy, "type": 2, "kind": "boundary_type_3"})  # ROAD_EDGE_BOUNDARY
    return els


class TestDecimation:
    def test_decimate_keeps_endpoints_and_drops_dense_points(self):
        xy = np.stack([np.linspace(0, 50, 100), np.zeros(100)], axis=1)
        out = builder.decimate_polyline(xy, min_seg_m=5.0)
        assert len(out) <= 12 and np.allclose(out[0], xy[0]) and np.allclose(out[-1], xy[-1])

    def test_decimate_preserves_sharp_turns(self):
        xy = np.array([[0, 0], [4.9, 0.01], [4.9, 4.9]])  # 90deg turn within <5m
        out = builder.decimate_polyline(xy, min_seg_m=5.0, turn_deg=15.0)
        assert np.allclose(out, xy)  # vertex kept by the turn rule

    def test_dense_interchange_load_under_cap_without_decimation_fails(self):
        """Red-side control: raw gpkg-density interchange blows the kernel's
        segments-per-cell cap (the 276>64 vegas failure mode)."""
        load = builder.cell_segment_load(_dense_interchange())
        assert max(load.values()) > 64

