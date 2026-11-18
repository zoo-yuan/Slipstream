"""Apollo bag four-channel coverage audit tests (G1 pre-conversion inventory).

Spec source: .scratch/proposals/apollo-bag-audit-proposal.md (r1) + review
.scratch/proposals/apollo-bag-audit-review.md (four blocking items adopted:
control threshold 25Hz, unmatched-channel list in summary, small-layer fallback
written down, CI wording + sampled-vehicle count; non-blocking suggestions
adopted where cheap: topic-match/type-mismatch bucket, nonpositive-duration ->
parse_error, resume keyed by (name,size,mtime), red-run evidence kept).

TDD contract: this file was written BEFORE scripts/apollo_bag_audit.py existed
and its first run is the red evidence (classify import fails).

Threshold note: 25/2.5/2.5/25Hz are PRESENCE-level "has" criteria (gate-1
review ruling); raw n/Hz land in bag_channels.csv so re-judging never rescans.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "apollo_bag_audit.py"
_spec = importlib.util.spec_from_file_location("apollo_bag_audit", _SCRIPT)
apollo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apollo)

# ---------------------------------------------------------------------------
# classify (pure; review blocking items #1/#2 live here)
# ---------------------------------------------------------------------------

LOC = "/apollo/localization/pose"
OBS = "/apollo/perception/obstacles"
PLN = "/apollo/planning"
CTL = "/apollo/control"
T_LOC = "apollo.localization.LocalizationEstimate"
T_OBS = "apollo.perception.PerceptionObstacles"
T_PLN = "apollo.planning.ADCTrajectory"
T_CTL = "apollo.control.ControlCommand"

MIN_HZ = {"localization": 25.0, "perception": 2.5, "planning": 2.5, "control": 25.0}


def ch(topic, mtype, n):
    return {"topic": topic, "message_type": mtype, "n": n}


def healthy_channels(T=20.0):
    # loc/ctrl 100Hz, obs 9Hz, planning 8Hz over T seconds (matches recon).
    return [
        ch(LOC, T_LOC, int(100 * T)),
        ch(OBS, T_OBS, int(9 * T)),
        ch(PLN, T_PLN, int(8 * T)),
        ch(CTL, T_CTL, int(100 * T)),
    ]


def test_classify_four_registered_channels_all_has():
    res = apollo.classify_bag(healthy_channels(), 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"] == {"localization": True, "perception": True, "planning": True, "control": True}
    assert res["missing"] == []
    assert res["matched"]["localization"]["topic"] == LOC


def test_classify_zero_messages_means_absent():
    # Index lists the channel but it has 0 actual messages -> "no" (§4 over-report guard).
    channels = healthy_channels() + [ch(CTL, T_CTL, 0)]
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"]["control"] is True  # the n>0 copy provides it
    channels = [c for c in healthy_channels()]
    channels[3] = ch(CTL, T_CTL, 0)
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"]["control"] is False
    assert "control" in res["missing"]


def test_classify_dual_threshold_message_count_floor():
    # Dual threshold: n >= min_hz * T AND hz = n/T >= min_hz (proposal §2).
    # loc @25Hz over T=20s needs n>=500: 499 fails, 500 passes (both criteria agree).
    base = healthy_channels()
    for n, expect in ((499, False), (500, True), (525, True)):
        channels = [ch(LOC, T_LOC, n)] + base[1:]
        res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
        assert res["has"]["localization"] is expect, (n, expect)


def test_classify_unmatched_typed_not_auto_passed():
    # Type matches a registered type but topic is unregistered -> unmatched_typed,
    # manually adjudicated, never auto-passed (real case: /apollo/perception/safetyobstacles).
    channels = [c for c in healthy_channels() if c["topic"] != OBS]
    channels.append(ch("/apollo/perception/safetyobstacles", T_OBS, 180))
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"]["perception"] is False
    assert {"topic": "/apollo/perception/safetyobstacles", "message_type": T_OBS, "n": 180} in res["unmatched_typed"]


def test_classify_topic_match_type_mismatch_bucket():
    # Symmetric bucket (review non-blocking suggestion): registered topic, wrong type.
    channels = healthy_channels() + [ch(CTL, "apollo.control.OtherCommand", 50)]
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"]["control"] is True  # the well-typed copy provides it
    assert {"topic": CTL, "message_type": "apollo.control.OtherCommand", "n": 50} in res["topic_type_mismatch"]


def test_classify_unknown_variant_visible():
    # Unknown topic + unknown type with n>0 must be visible, never silently "missing".
    channels = healthy_channels() + [ch("/apollo/planning2", "apollo.planning.ADCTrajectoryV2", 40)]
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["unmatched_unknown"] == [{"topic": "/apollo/planning2", "message_type": "apollo.planning.ADCTrajectoryV2", "n": 40}]


def test_classify_below_hz_threshold_not_has():
    # obs needs >=2.5Hz; 2.0Hz over 20s (n=40) fails.
    channels = [ch(LOC, T_LOC, 2000), ch(OBS, T_OBS, 40), ch(PLN, T_PLN, 160), ch(CTL, T_CTL, 2000)]
    res = apollo.classify_bag(channels, 20.0, apollo.DEFAULT_ALIASES, MIN_HZ)
    assert res["has"]["perception"] is False
    assert res["has"]["planning"] is True


def test_default_thresholds_and_aliases_are_locked_ruling():
    # Gate-1 review ruling locked 25/2.5/2.5/25 (control raised 10 -> 25).
    assert apollo.DEFAULT_MIN_HZ == {"localization": 25.0, "perception": 2.5, "planning": 2.5, "control": 25.0}
    assert apollo.DEFAULT_ALIASES["control"] == ((CTL, T_CTL),)
    assert set(apollo.DEFAULT_ALIASES) == {"localization", "perception", "planning", "control"}


# ---------------------------------------------------------------------------
# Wilson CI (pure)
# ---------------------------------------------------------------------------


def test_wilson_ci_reference_values():
    assert apollo.wilson_ci(33, 33) == pytest.approx((0.895730002457548, 1.0), abs=1e-9)
    assert apollo.wilson_ci(0, 33) == pytest.approx((0.0, 0.10426999754245206), abs=1e-9)
    assert apollo.wilson_ci(16, 20) == pytest.approx((0.5839825677481064, 0.919342337420202), abs=1e-9)
    assert apollo.wilson_ci(5, 10) == pytest.approx((0.236593090512564, 0.7634069094874361), abs=1e-9)


def test_wilson_ci_empty_n_degenerate():
    assert apollo.wilson_ci(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# manifest / layers / sampling (pure)
# ---------------------------------------------------------------------------


def test_parse_bag_name():
    row = apollo.build_manifest_row("white-rhino-052.20260120.simple.1768893035-1768893055.bag", 123, 1.5)
    assert row["pattern"] == "ok"
    assert row["vehicle"] == "white-rhino-052"
    assert row["date"] == "20260120"
    assert row["mode"] == "simple"
    assert row["month"] == "202601"
    assert row["ts1"] == 1768893035
    assert row["ts2"] == 1768893055
    assert row["size"] == 123 and row["mtime"] == 1.5

    bad = apollo.build_manifest_row("something-else.bag", 1, 0)
    assert bad["pattern"] != "ok"
    assert bad["vehicle"] is None


def test_assign_layers_merges_small_cells_deterministically():
    def rows(month, mode, count, vehicle_num):
        out = []
        for i in range(count):
            name = f"white-rhino-{vehicle_num:02d}{i:03d}.{month}01.{mode}.100000000{i}-1000000{i + 20}.bag"
            r = apollo.build_manifest_row(name, 10, 0)
            out.append(r)
        return out

    all_rows = (
        rows("202508", "small_bag", 40, 1)
        + rows("202509", "small_bag", 5, 2)    # tiny -> merges into 202508 (same mode, tie -> smaller month)
        + rows("202510", "small_bag", 40, 3)
        + rows("202606", "simple", 50, 4)
        + rows("202607", "simple", 3, 5)       # tiny -> merges into 202606
    )
    layers = apollo.assign_layers(all_rows, min_cell=33)
    assert len(layers) == 3
    by_id = {l["layer_id"]: l for l in layers}
    # merged layer keeps the dominant (target) cell id; tiny cells merge to the
    # nearest same-mode cell (tie -> smaller month)
    assert by_id["small_bag[202508-202509]"]["n_bags"] == 45
    assert by_id["small_bag[202510-202510]"]["n_bags"] == 40
    assert by_id["simple[202606-202607]"]["n_bags"] == 53
    # determinism
    assert apollo.assign_layers(all_rows, min_cell=33) == layers


def test_draw_sample_vehicle_cap_reproducible_and_fallback():
    # 6 vehicles x 5 bags in one layer, target 33 -> cap 2/vehicle -> 12 selected.
    rows = []
    for v in range(6):
        for k in range(5):
            name = f"white-rhino-90{v}.{202508 + k}05.small_bag.100000000{k}-10000002{k}.bag"
            rows.append(apollo.build_manifest_row(name, 10, 0))
    out = apollo.draw_sample(rows, per_layer=33, seed=20261118, per_vehicle_cap=2, min_cell=33)
    assert len(out["selected"]) == 12  # min(33, 2*6) fallback truncation (review blocking #3)
    layer = out["layers"][0]
    assert layer["target"] == 33 and layer["achieved"] == 12 and layer["shortfall"] == 21
    per_vehicle = {}
    for r in out["selected"]:
        per_vehicle[r["vehicle"]] = per_vehicle.get(r["vehicle"], 0) + 1
    assert all(c <= 2 for c in per_vehicle.values()) and len(per_vehicle) == 6
    # reproducible with same seed
    out2 = apollo.draw_sample(rows, per_layer=33, seed=20261118, per_vehicle_cap=2, min_cell=33)
    assert [r["name"] for r in out2["selected"]] == [r["name"] for r in out["selected"]]
    # a different seed reshuffles the draw (order and/or set)
    out3 = apollo.draw_sample(rows, per_layer=33, seed=1, per_vehicle_cap=2, min_cell=33)
    assert [r["name"] for r in out3["selected"]] != [r["name"] for r in out["selected"]]


# ---------------------------------------------------------------------------
# summarize (pure aggregation over scanned rows)
# ---------------------------------------------------------------------------


def _summarize_fixture():
    names = {
        "a": "white-rhino-001.20250801.small_bag.1000000000-1000000020.bag",
        "b": "white-rhino-001.20250802.small_bag.1000001000-1000001020.bag",
        "c": "white-rhino-002.20250803.small_bag.1000002000-1000002020.bag",
        "d": "white-rhino-003.20260601.simple.1000003000-1000003020.bag",
        "e": "white-rhino-003.20260602.simple.1000004000-1000004020.bag",
        "f": "white-rhino-004.20260603.simple.1000005000-1000005020.bag",
    }
    manifest = [apollo.build_manifest_row(n, 10, 0) for n in names.values()]
    selected = [dict(apollo.build_manifest_row(n, 10, 0)) for n in names.values()]

    def chan_rows(bag, which, T=20.0, extra=None):
        n_map = {"localization": (LOC, T_LOC, 100), "perception": (OBS, T_OBS, 9),
                 "planning": (PLN, T_PLN, 8), "control": (CTL, T_CTL, 100)}
        rows = []
        for role in which:
            topic, mtype, hz = n_map[role]
            rows.append({"bag": bag, "channel": topic, "message_type": mtype,
                         "n": int(hz * T), "duration_s": T, "hz": hz,
                         "first_offset_s": 0.0, "last_offset_s": T - 0.01, "read_path": "scan"})
        for extra_row in extra or []:
            rows.append({"bag": bag, "channel": extra_row[0], "message_type": extra_row[1],
                         "n": extra_row[2], "duration_s": T, "hz": extra_row[2] / T,
                         "first_offset_s": 0.0, "last_offset_s": T - 0.01, "read_path": "scan"})
        return rows

    channel_rows = (
        chan_rows(names["a"], ["localization", "perception", "planning", "control"])          # full, T=20
        + chan_rows(names["b"], ["localization", "perception", "planning", "control"], T=40)  # full, T=40 (seconds-weighted pull)
        + chan_rows(names["c"], ["localization", "perception", "planning"])                   # missing control
        + chan_rows(names["d"], ["localization", "perception", "planning", "control"],
                    extra=[("/apollo/perception/safetyobstacles", T_OBS, 180)])        # full + unmatched_typed variant
        # bag e: parse_error -> no channel rows, excluded from BOTH sides of the ratio
        + chan_rows(names["f"], ["localization", "control"])                                  # missing perception+planning
    )
    error_rows = [{"bag": names["e"], "size": 10, "mtime": 0, "reason": "scan_count_mismatch", "detail": "scanned=0 header=100", "elapsed_s": 0.1}]
    return names, manifest, selected, channel_rows, error_rows


def test_summarize_pure_aggregation():
    names, manifest, selected, channel_rows, error_rows = _summarize_fixture()
    summary = apollo.summarize_from_tables(
        manifest_rows=manifest, sample_rows=selected, channel_rows=channel_rows,
        error_rows=error_rows, aliases=apollo.DEFAULT_ALIASES, min_hz=MIN_HZ,
        seed=20261118, per_layer=33, per_vehicle_cap=2, min_cell=2,
        bootstrap_reps=500,
    )
    d = summary["denominators"]
    assert d["sample_selected"] == 6 and d["sample_parse_error"] == 1 and d["classified"] == 5
    # pooled: 3 full / 5 classified
    assert summary["overall"]["pooled_sample_full4"] == {"x": 3, "n": 5, "p": 0.6}
    # seconds-weighted: (20+40+20) full / (20+40+20+20+20) total = 80/120
    sw = summary["overall"]["seconds_weighted_full4"]
    assert sw == {"seconds_full4": 80.0, "seconds_total": 120.0, "p": 80.0 / 120.0}
    # missing combos: none=3, control=1, perception+planning=1
    combos = {tuple(c["missing"]): c["n"] for c in summary["missing_combinations"]}
    assert combos[("none",)] == 3 and combos[("control",)] == 1 and combos[("perception", "planning")] == 1
    # unmatched typed variant is visible with n>0 (review blocking #2)
    ut = summary["unmatched_channels"]["unmatched_typed"]
    assert ut == [{"bag": names["d"], "channel": "/apollo/perception/safetyobstacles", "message_type": T_OBS, "n": 180}]
    assert summary["unmatched_channels"]["unknown_variant"] == []
    # parse errors listed, excluded from strata denominators
    assert summary["parse_errors"][0]["bag"] == names["e"]
    assert all(l["n_classified"] + l["n_parse_error"] == l["achieved"] for l in summary["per_layer"])
    # vehicle stats recorded (review blocking #4)
    assert summary["sampling"]["vehicles_sampled"] == 4  # 001,002,003,004
    assert summary["sampling"]["max_bags_per_vehicle"] == 2
    assert "未校正" in summary["sampling"]["ci_note"] or "偏窄" in summary["sampling"]["ci_note"]
    # stratified estimate = sum over layers of (N_h/N)*p_h; 2 layers, N_h=3 each, p_h = 2/3 and 1/2
    est = summary["overall"]["stratified_full4"]["estimate"]
    assert est == pytest.approx((3 / 6) * (2 / 3) + (3 / 6) * 0.5)
    # thresholds snapshot for re-judging without rescan
    assert summary["thresholds"]["min_hz"] == MIN_HZ
    assert summary["thresholds"]["aliases"] == {k: [list(p) for p in v] for k, v in apollo.DEFAULT_ALIASES.items()}
    lo, hi = summary["overall"]["cluster_bootstrap_ci95"]
    assert 0.0 <= lo <= est <= hi <= 1.0


def test_summarize_reweighs_shortfall_layers():
    # Layer with truncated sample still carries its population weight N_h/N.
    names, manifest, selected, channel_rows, error_rows = _summarize_fixture()
    # drop 2 of layer-1 bags from selection+rows -> layer1 achieved=1, p=1/1
    keep = {"a", "d", "e", "f"}
    sel_by_key = dict(zip("abcdef", selected))
    selected = [sel_by_key[k] for k in sorted(keep)]
    keep_names = {names[k] for k in keep}
    channel_rows = [r for r in channel_rows if r["bag"] in keep_names]
    summary = apollo.summarize_from_tables(
        manifest_rows=manifest, sample_rows=selected, channel_rows=channel_rows,
        error_rows=[e for e in error_rows if e["bag"] in keep_names], aliases=apollo.DEFAULT_ALIASES,
        min_hz=MIN_HZ, seed=20261118, per_layer=33, per_vehicle_cap=2, min_cell=2,
        bootstrap_reps=0,
    )
    # N_h are still 3 and 3 from the manifest: estimate = 0.5*1.0 + 0.5*(1/2)
    assert summary["overall"]["stratified_full4"]["estimate"] == pytest.approx(0.75)
    layer1 = next(l for l in summary["per_layer"] if l["n_classified"] == 1)
    assert layer1["N_h"] == 3 and layer1["achieved"] == 1


# ---------------------------------------------------------------------------
# audit plumbing (pure parts) + import safety
# ---------------------------------------------------------------------------


def test_read_path_scan_constant_guard():
    # §4: only section_scan counts; every emitted row is tagged read_path="scan".
    assert apollo.READ_PATH_SCAN == "scan"


def test_module_imports_without_cyber_record():
    # Import-safety: the module must be loadable and its pure API callable even
    # when cyber_record is unavailable (unit tests / summarize on any host).
    code = (
        "import sys, importlib.util, pathlib\n"
        "sys.modules['cyber_record'] = None\n"
        "spec = importlib.util.spec_from_file_location('aba', pathlib.Path(sys.argv[1]))\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "r = m.classify_bag([{'topic': '/apollo/control', 'message_type': 'apollo.control.ControlCommand', 'n': 3000}], 20.0, m.DEFAULT_ALIASES, m.DEFAULT_MIN_HZ)\n"
        "assert r['has']['control'] is True and r['has']['localization'] is False\n"
    )
    subprocess.run([sys.executable, "-c", code, str(_SCRIPT)], check=True)


# ---------------------------------------------------------------------------
# integration with cyber_record (skipif absent): synthetic four-channel bag
# ---------------------------------------------------------------------------

cyber_record = pytest.importorskip("cyber_record")


def _make_proto_class(package, name):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = f"{package.replace('.', '_')}_{name.lower()}.proto"
    fdp.package = package
    fdp.message_type.add(name=name)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{package}.{name}"))


def _write_bag(path, T_s=20.0):
    from cyber_record.record import Record
    t0 = 1_700_000_000_000_000_000  # ns
    plan = [
        (LOC, "apollo.localization", "LocalizationEstimate", 100),   # 100 Hz
        (OBS, "apollo.perception", "PerceptionObstacles", 9),        # 9 Hz
        (PLN, "apollo.planning", "ADCTrajectory", 8),                # 8 Hz
        (CTL, "apollo.control", "ControlCommand", 100),              # 100 Hz
        ("/apollo/perception/safetyobstacles", "apollo.perception", "PerceptionObstacles", 9),  # unmatched_typed variant
    ]
    with Record(str(path), mode="w") as rec:
        for topic, pkg, msg_name, hz in plan:
            cls = _make_proto_class(pkg, msg_name)
            for i in range(int(hz * T_s)):
                rec.write(topic, cls(), int(t0 + i * 1e9 / hz))
    return t0, int(t0 + T_s * 1e9)


def test_audit_one_bag_synthetic(tmp_path):
    bag = tmp_path / "white-rhino-777.20260101.simple.1700000000-1700000020.bag"
    _write_bag(bag)
    out = apollo.audit_one_bag(str(bag))
    assert out["status"] == "ok", out.get("reason_detail")
    assert out["duration_s"] == pytest.approx(20.0, abs=0.2)
    assert out["header_message_number"] == sum(int(hz * 20.0) for _, _, _, hz in
                                               [(LOC, 0, 0, 100), (OBS, 0, 0, 9), (PLN, 0, 0, 8), (CTL, 0, 0, 100), ("/x", 0, 0, 9)])
    rows = {r["channel"]: r for r in out["rows"]}
    assert set(rows) == {LOC, OBS, PLN, CTL, "/apollo/perception/safetyobstacles"}
    assert all(r["read_path"] == apollo.READ_PATH_SCAN for r in out["rows"])
    assert rows[LOC]["n"] == 2000 and rows[LOC]["hz"] == pytest.approx(100.0, rel=0.01)
    assert rows[CTL]["n"] == 2000
    assert rows[OBS]["message_type"] == "apollo.perception.PerceptionObstacles"
    assert rows[LOC]["first_offset_s"] == pytest.approx(0.0, abs=0.05)
    assert rows[LOC]["last_offset_s"] > 19.0
    # classification over the real parsed rows
    res = apollo.classify_bag(
        [{"topic": r["channel"], "message_type": r["message_type"], "n": r["n"]} for r in out["rows"]],
        out["duration_s"], apollo.DEFAULT_ALIASES, apollo.DEFAULT_MIN_HZ,
    )
    assert res["has"] == {"localization": True, "perception": True, "planning": True, "control": True}
    assert res["unmatched_typed"][0]["topic"] == "/apollo/perception/safetyobstacles"


def test_audit_one_bag_corrupt_tail_is_parse_error(tmp_path):
    bag = tmp_path / "white-rhino-778.20260101.simple.1700000000-1700000020.bag"
    _write_bag(bag)
    data = bag.read_bytes()
    (tmp_path / "trunc.bag").write_bytes(data[: int(len(data) * 0.55)])  # cut mid-file
    out = apollo.audit_one_bag(str(tmp_path / "trunc.bag"))
    assert out["status"] == "parse_error"
    assert out["reason"]  # machine-readable reason string present


def test_audit_cli_end_to_end(tmp_path):
    bag_dir = tmp_path / "bags"
    bag_dir.mkdir()
    bag = bag_dir / "white-rhino-779.20260101.simple.1700000000-1700000020.bag"
    _write_bag(bag)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    m = out_dir / "manifest.jsonl"
    apollo.main(["manifest", "--bag-dir", str(bag_dir), "--out", str(m)])
    rows, as_of = apollo.load_manifest(str(m))
    assert as_of and len(rows) == 1 and rows[0]["pattern"] == "ok"
    s = out_dir / "samples.csv"
    apollo.main(["sample", "--manifest", str(m), "--per-layer", "3", "--seed", "7", "--out", str(s)])
    assert "white-rhino-779" in s.read_text()
    apollo.main(["audit", "--manifest", str(m), "--samples", str(s), "--bag-dir", str(bag_dir),
                 "--out-dir", str(out_dir)])
    csv_text = (out_dir / "bag_channels.csv").read_text()
    assert "read_path" in csv_text and ",scan" in csv_text
    apollo.main(["summarize", "--manifest", str(m), "--samples", str(s),
                 "--channels", str(out_dir / "bag_channels.csv"),
                 "--errors", str(out_dir / "audit_errors.jsonl"),
                 "--out", str(out_dir / "summary.json")])
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["denominators"]["sample_selected"] == 1
    assert summary["denominators"]["classified"] == 1
    assert summary["overall"]["pooled_sample_full4"]["x"] == 1
