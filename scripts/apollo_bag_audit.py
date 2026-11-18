#!/usr/bin/env python
"""Apollo bag four-channel coverage audit (G1 converter pre-inventory).

Route (gate 1, reviewed r1): .scratch/proposals/apollo-bag-audit-proposal.md
Review blocking items adopted: control threshold 25Hz; unmatched-channel
(n>0) lists in summary.json; small-layer fallback written down (truncate to
per-vehicle-cap draw, report shortfall, reweight by N_h); CI wording
"uncorrected, may be narrow" + sampled-vehicle count recorded.

Design invariants:
- Pure logic (name parsing, layer merge, sampling, classification, Wilson CI,
  stratified aggregation) is decoupled from cyber_record, which is imported
  lazily inside audit_one_bag() only -> the module is import-safe anywhere.
- All directories/files come from CLI parameters; no hardcoded paths.
- audit only ever counts messages via read_messages_section_scan() (dataset
  index quirk: read_messages() returns 0 on these bags) and tags every row
  read_path="scan"; index-listed channels with zero scanned messages are
  recorded with n=0 and judged absent (index over-reports, recon §0).
- Thresholds are summarize-time parameters (default 25/2.5/2.5/25Hz presence
  ruling); raw n/Hz land in bag_channels.csv so re-judging never rescans.

Outputs (out-dir): bag_channels.csv, audit_index.jsonl, audit_errors.jsonl;
manifest/sample/summarize write their own --out paths. summary.json carries
per-layer four-has rates + Wilson CI, missing combinations, parse_error list,
unmatched channel lists (n>0), sampled-vehicle stats, as-of, threshold and
alias snapshots, seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants (gate-1 review ruling: presence thresholds 25/2.5/2.5/25Hz)
# ---------------------------------------------------------------------------

ROLES = ("localization", "perception", "planning", "control")

DEFAULT_MIN_HZ = {
    "localization": 25.0,
    "perception": 2.5,
    "planning": 2.5,
    "control": 25.0,
}

# Alias table initial values = the four (topic, message_type) pairs measured on
# the real bags (recon 5/5; qualified names confirmed on 2026-11-18). New
# variants found during pilot/full runs are adjudicated manually and added via
# --aliases <json>; they are never auto-passed.
DEFAULT_ALIASES = {
    "localization": (("/apollo/localization/pose", "apollo.localization.LocalizationEstimate"),),
    "perception": (("/apollo/perception/obstacles", "apollo.perception.PerceptionObstacles"),),
    "planning": (("/apollo/planning", "apollo.planning.ADCTrajectory"),),
    "control": (("/apollo/control", "apollo.control.ControlCommand"),),
}

READ_PATH_SCAN = "scan"

BAG_NAME_RE = re.compile(
    r"^(?P<vehicle>white-rhino-\d+)\.(?P<date>\d{8})\.(?P<mode>simple|small_bag)\."
    r"(?P<ts1>\d+)-(?P<ts2>\d+)\.bag$"
)

Z95 = 1.959963984540054  # two-sided 95%

CHANNEL_COLUMNS = [
    "bag", "channel", "message_type", "n", "duration_s", "hz",
    "first_offset_s", "last_offset_s", "read_path",
]

SAMPLE_COLUMNS = ["bag", "layer", "vehicle", "date", "mode", "month", "size", "mtime", "seed"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Manifest (pure row builder + thin directory scanner)
# ---------------------------------------------------------------------------


def build_manifest_row(name: str, size: int, mtime: float) -> dict:
    m = BAG_NAME_RE.match(name)
    if m is None:
        return {
            "name": name, "size": size, "mtime": mtime, "pattern": "unrecognized",
            "vehicle": None, "date": None, "mode": None, "month": None,
            "ts1": None, "ts2": None,
        }
    d = m.groupdict()
    return {
        "name": name, "size": size, "mtime": mtime, "pattern": "ok",
        "vehicle": d["vehicle"], "date": d["date"], "mode": d["mode"],
        "month": d["date"][:6], "ts1": int(d["ts1"]), "ts2": int(d["ts2"]),
    }


def scan_bag_dir(bag_dir: str) -> list:
    """Snapshot-enumerate *.bag (non-recursive) as (name, size, mtime)."""
    out = []
    with os.scandir(bag_dir) as it:
        for entry in sorted(it, key=lambda e: e.name):
            if not entry.name.endswith(".bag") or not entry.is_file(follow_symlinks=False):
                continue
            st = entry.stat(follow_symlinks=False)
            out.append((entry.name, st.st_size, st.st_mtime))
    return out


# ---------------------------------------------------------------------------
# Stratification: month x mode cells, small cells merged deterministically
# ---------------------------------------------------------------------------


def assign_layers(rows: list, min_cell: int) -> list:
    """Group pattern-ok rows into layers.

    Rule (proposal §6): cells are (month, mode); any cell smaller than
    min_cell merges into the nearest cell -- same mode preferred, then
    nearest month, then lexicographic (month, mode). Fully deterministic.
    """
    cells = {}
    for r in rows:
        if r.get("pattern") != "ok":
            continue
        cells.setdefault((r["month"], r["mode"]), []).append(r)

    def merge_once():
        smallest_key = min(cells, key=lambda k: (len(cells[k]), k))
        candidates = [k for k in cells if k != smallest_key]
        month = int(smallest_key[0])
        target = min(
            candidates,
            key=lambda k: (0 if k[1] == smallest_key[1] else 1, abs(int(k[0]) - month), k),
        )
        cells[target] = cells[target] + cells.pop(smallest_key)

    while len(cells) > 1:
        if min(len(v) for v in cells.values()) >= min_cell:
            break
        merge_once()

    layers = []
    for (month, mode) in sorted(cells):
        bag_rows = sorted(cells[(month, mode)], key=lambda r: r["name"])
        months = sorted({r["month"] for r in bag_rows})
        modes = sorted({r["mode"] for r in bag_rows})
        label = modes[0] if len(modes) == 1 else "mixed"
        layer_id = f"{label}[{months[0]}-{months[-1]}]"
        layers.append({
            "layer_id": layer_id,
            "months": months,
            "modes": modes,
            "n_bags": len(bag_rows),
            "n_vehicles": len({r["vehicle"] for r in bag_rows}),
            "bags": [r["name"] for r in bag_rows],
            "_rows": bag_rows,
        })
    layers.sort(key=lambda l: l["layer_id"])
    return layers


def draw_sample(rows: list, per_layer: int, seed: int, per_vehicle_cap: int = 2,
                min_cell: int = 33) -> dict:
    """Stratified draw: srs within layer (seeded, per-layer RNG), each vehicle
    capped at per_vehicle_cap bags. If a layer cannot reach per_layer under the
    cap, the draw truncates and the shortfall is reported (review blocking #3);
    reweighting by N_h happens in summarize, never by relaxing the cap or
    borrowing across layers."""
    vehicle_of = {r["name"]: r["vehicle"] for r in rows if r.get("pattern") == "ok"}
    layers = assign_layers(rows, min_cell)
    selected = []
    layer_table = []
    for layer in layers:
        rng = random.Random(f"{seed}|{layer['layer_id']}")
        order = list(layer["bags"])
        rng.shuffle(order)
        vcount = {}
        take = []
        for name in order:
            if len(take) >= per_layer:
                break
            v = vehicle_of[name]
            if vcount.get(v, 0) < per_vehicle_cap:
                take.append(name)
                vcount[v] = vcount.get(v, 0) + 1
        achieved = len(take)
        layer_table.append({
            "layer_id": layer["layer_id"], "n_bags": layer["n_bags"],
            "n_vehicles": layer["n_vehicles"], "target": per_layer,
            "achieved": achieved, "shortfall": max(0, per_layer - achieved),
        })
        row_by_name = {r["name"]: r for r in layer["_rows"]}
        for name in sorted(take):
            selected.append(dict(row_by_name[name], layer=layer["layer_id"]))
    return {
        "selected": selected,
        "layers": layer_table,
        "excluded_unpatterned": sum(1 for r in rows if r.get("pattern") != "ok"),
    }


# ---------------------------------------------------------------------------
# Wilson CI + classification (pure)
# ---------------------------------------------------------------------------


def wilson_ci(x: int, n: int, z: float = Z95):
    """Wilson score interval for x successes in n trials (95% by default).
    Degenerate n=0 -> (0.0, 1.0). Note: srs-within-stratum interval; vehicle
    clustering is limited (<=2 bags/vehicle) but not eliminated."""
    if n <= 0:
        return (0.0, 1.0)
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def classify_bag(channels, duration_s: float, aliases=DEFAULT_ALIASES,
                 min_hz=DEFAULT_MIN_HZ) -> dict:
    """Judge a bag's four channels from actual scanned counts.

    ``channels``: iterable of mappings with topic / message_type / n.
    "Has" = an exactly registered (topic, message_type) pair whose message
    count n satisfies BOTH thresholds n >= min_hz * T and n / T >= min_hz
    (dual presence criteria, proposal §2). Index-only channels (n=0) never
    count. Type-registered-but-topic-unregistered channels with n>0 land in
    unmatched_typed for manual adjudication (never auto-passed); topic-
    registered-type-mismatch lands in topic_type_mismatch; fully unknown
    channels with n>0 land in unmatched_unknown so variant channels are never
    silently judged missing (review blocking #2).
    """
    registered = {}
    for role, pairs in aliases.items():
        for pair in pairs:
            registered[(pair[0], pair[1])] = role
    registered_topics = {p[0] for p in registered}
    registered_types = {p[1] for p in registered}

    best = {role: None for role in ROLES}  # role -> (n, topic, hz)
    unmatched_typed, topic_type_mismatch, unmatched_unknown = [], [], []

    for c in channels:
        topic = c.get("topic")
        mtype = c.get("message_type")
        n = int(c.get("n", 0))
        role = registered.get((topic, mtype))
        if role is not None:
            if best[role] is None or n > best[role][0]:
                hz = (n / duration_s) if duration_s and duration_s > 0 else 0.0
                best[role] = (n, topic, hz)
            continue
        if n <= 0:
            continue
        entry = {"topic": topic, "message_type": mtype, "n": n}
        type_reg = mtype in registered_types
        topic_reg = topic in registered_topics
        if type_reg:
            unmatched_typed.append(entry)
        if topic_reg:
            topic_type_mismatch.append(entry)
        if not type_reg and not topic_reg:
            unmatched_unknown.append(entry)

    has, matched, missing = {}, {}, []
    T = duration_s if duration_s and duration_s > 0 else 0.0
    for role in ROLES:
        cand = best[role]
        matched[role] = (
            {"topic": cand[1], "n": cand[0], "hz": cand[2]} if cand is not None else None
        )
        ok = False
        if cand is not None and T > 0:
            thr = min_hz[role]
            ok = cand[0] >= thr * T - 1e-9 and cand[0] / T >= thr - 1e-9
        has[role] = ok
        if not ok:
            missing.append(role)

    def entry_key(e):
        return (e["topic"], e["message_type"])

    return {
        "has": has,
        "missing": missing,
        "matched": matched,
        "unmatched_typed": sorted(unmatched_typed, key=entry_key),
        "topic_type_mismatch": sorted(topic_type_mismatch, key=entry_key),
        "unmatched_unknown": sorted(unmatched_unknown, key=entry_key),
    }


# ---------------------------------------------------------------------------
# audit: cyber_record parsing (lazy import; section_scan only)
# ---------------------------------------------------------------------------


def audit_one_bag(path: str) -> dict:
    """Parse one bag with read_messages_section_scan and return channel rows.

    Status "ok" requires: header begin/end readable with end > begin, and the
    number of scanned messages equal to header.message_number (the scan path
    silently breaks on corrupt sections, so any deficit is a parse_error --
    proposal §4 scan==0 rule tightened to any deficit, both sides excluded).
    """
    t0 = time.monotonic()

    def done(payload):
        payload["elapsed_s"] = round(time.monotonic() - t0, 3)
        return payload

    try:
        from cyber_record.record import Record  # lazy: import-safety
        with Record(path) as rec:
            header_n = int(rec.get_message_count() or 0)
            begin = rec.get_start_time()
            end = rec.get_end_time()
            channel_types = {}
            for cname, info in sorted(getattr(rec._reader, "channels", {}).items()):
                channel_types[cname] = getattr(info, "message_type", None)
            counts, first_t, last_t = {}, {}, {}
            for chan, _msg, t in rec.read_messages_section_scan():
                counts[chan] = counts.get(chan, 0) + 1
                if chan not in first_t or t < first_t[chan]:
                    first_t[chan] = t
                if chan not in last_t or t > last_t[chan]:
                    last_t[chan] = t
    except Exception as exc:  # noqa: BLE001 - any open/parse failure is a parse_error
        return done({"status": "parse_error", "reason": "open_error",
                     "detail": f"{type(exc).__name__}: {exc}", "rows": []})

    if begin is None or end is None or end <= begin:
        return done({"status": "parse_error", "reason": "bad_header_time_range",
                     "detail": f"begin={begin} end={end} header_n={header_n}", "rows": []})
    duration_s = (end - begin) / 1e9
    total = sum(counts.values())
    if header_n > 0 and total != header_n:
        reason = "scan_zero_with_header_n" if total == 0 else "scan_count_mismatch"
        return done({"status": "parse_error", "reason": reason,
                     "detail": f"scanned={total} header={header_n}", "rows": []})

    rows = []
    for chan in sorted(set(counts) | set(channel_types)):
        n = counts.get(chan, 0)
        hz = n / duration_s if duration_s > 0 else 0.0
        fo = (first_t[chan] - begin) / 1e9 if chan in first_t else ""
        lo = (last_t[chan] - begin) / 1e9 if chan in last_t else ""
        rows.append({
            "channel": chan, "message_type": channel_types.get(chan) or "",
            "n": n, "duration_s": duration_s, "hz": round(hz, 4),
            "first_offset_s": round(fo, 4) if fo != "" else "",
            "last_offset_s": round(lo, 4) if lo != "" else "",
            "read_path": READ_PATH_SCAN,
        })
    return done({
        "status": "ok", "duration_s": duration_s,
        "header_message_number": header_n, "scanned_messages": total,
        "n_channels": len(rows), "n_channels_with_messages": len(counts),
        "rows": rows,
    })


# ---------------------------------------------------------------------------
# summarize: stratified aggregation + CIs (pure over scanned tables)
# ---------------------------------------------------------------------------


def summarize_from_tables(*, manifest_rows, sample_rows, channel_rows, error_rows,
                          aliases, min_hz, seed, per_layer, per_vehicle_cap,
                          min_cell, bootstrap_reps=2000, generated_at=None,
                          manifest_as_of=None) -> dict:
    patterned = [r for r in manifest_rows if r.get("pattern") == "ok"]
    layers = assign_layers(patterned, min_cell)
    layer_of = {}
    for layer in layers:
        for name in layer["bags"]:
            layer_of[name] = layer["layer_id"]

    sampled_names = [r["name"] for r in sample_rows]
    vehicle_of = {r["name"]: r.get("vehicle") for r in sample_rows}
    rows_by_bag = {}
    for row in channel_rows:
        rows_by_bag.setdefault(row["bag"], []).append(row)
    errors_by_bag = {e["bag"]: e for e in error_rows if e["bag"] in set(sampled_names)}

    classifications, parse_errors = {}, []
    for name in sampled_names:
        if name in errors_by_bag:
            parse_errors.append({
                "bag": name, "reason": errors_by_bag[name].get("reason"),
                "detail": errors_by_bag[name].get("detail"),
            })
            continue
        bag_rows = rows_by_bag.get(name)
        if not bag_rows:
            parse_errors.append({"bag": name, "reason": "missing_scan", "detail": "no rows in bag_channels.csv"})
            continue
        duration_s = bag_rows[0]["duration_s"]
        res = classify_bag(
            [{"topic": r["channel"], "message_type": r["message_type"], "n": r["n"]}
             for r in bag_rows],
            duration_s, aliases, min_hz,
        )
        classifications[name] = (res, duration_s)

    def full4(res):
        return all(res["has"][role] for role in ROLES)

    per_layer_out = []
    combo_counts = {}
    for layer in layers:
        lid = layer["layer_id"]
        sel = [n for n in sampled_names if layer_of.get(n) == lid]
        cls = [n for n in sel if n in classifications]
        x = sum(1 for n in cls if full4(classifications[n][0]))
        n_cls = len(cls)
        p = (x / n_cls) if n_cls else None
        lo, hi = wilson_ci(x, n_cls) if n_cls else (None, None)
        for n in cls:
            combo = tuple(classifications[n][0]["missing"]) or ("none",)
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        per_layer_out.append({
            "layer_id": lid, "months": layer["months"], "modes": layer["modes"],
            "N_h": layer["n_bags"], "vehicles_in_layer": layer["n_vehicles"],
            "target": per_layer, "achieved": len(sel),
            "shortfall": max(0, per_layer - len(sel)),
            "n_classified": n_cls,
            "n_parse_error": len(sel) - n_cls,
            "vehicles_sampled": len({vehicle_of[n] for n in sel}),
            "x_full4": x, "p_full4": p,
            "wilson_lo": lo, "wilson_hi": hi,
        })

    x_total = sum(l["x_full4"] for l in per_layer_out)
    n_total = sum(l["n_classified"] for l in per_layer_out)
    N = len(manifest_rows)

    represented = [l for l in per_layer_out if l["n_classified"] > 0]
    weight_sum = sum(l["N_h"] for l in represented)
    est = sum((l["N_h"] / N) * l["p_full4"] for l in represented) if represented else None
    if represented and weight_sum:
        var = 0.0
        for l in represented:
            nh, Nh, ph = l["n_classified"], l["N_h"], l["p_full4"]
            fpc = 1.0 - nh / Nh if Nh > nh else 0.0
            var += (Nh / N) ** 2 * fpc * ph * (1 - ph) / max(nh - 1, 1)
        ci = [max(0.0, est - Z95 * math.sqrt(var)), min(1.0, est + Z95 * math.sqrt(var))]
    else:
        ci = None

    seconds_full = sum(classifications[n][1] for n in classifications if full4(classifications[n][0]))
    seconds_total = sum(classifications[n][1] for n in classifications)

    buckets = {"unmatched_typed": [], "topic_type_mismatch": [], "unknown_variant": []}
    for name in sorted(classifications):
        res = classifications[name][0]
        for out_key, in_key in (("unmatched_typed", "unmatched_typed"),
                                ("topic_type_mismatch", "topic_type_mismatch"),
                                ("unknown_variant", "unmatched_unknown")):
            for e in res[in_key]:
                buckets[out_key].append({"bag": name, "channel": e["topic"],
                                         "message_type": e["message_type"], "n": e["n"]})

    vcount = {}
    for r in sample_rows:
        vcount[r.get("vehicle")] = vcount.get(r.get("vehicle"), 0) + 1

    bootstrap_ci = None
    if bootstrap_reps and represented:
        rng = random.Random(f"{seed}|bootstrap")
        by_layer = {}
        for l in represented:
            lid = l["layer_id"]
            veh_bags = {}
            for n in sampled_names:
                if layer_of.get(n) == lid and n in classifications:
                    veh_bags.setdefault(vehicle_of[n], []).append(full4(classifications[n][0]))
            by_layer[lid] = (l["N_h"], veh_bags)
        stats = []
        for _ in range(bootstrap_reps):
            s = 0.0
            for lid, (Nh, veh_bags) in by_layer.items():
                vehicles = sorted(veh_bags)
                picked = rng.choices(vehicles, k=len(vehicles))
                flags = [f for v in picked for f in veh_bags[v]]
                s += (Nh / N) * (sum(flags) / len(flags))
            stats.append(s)
        stats.sort()
        bootstrap_ci = [stats[int(0.025 * len(stats))],
                        stats[min(len(stats) - 1, int(0.975 * len(stats)))]]

    missing_combinations = [
        {"missing": list(k), "n": v}
        for k, v in sorted(combo_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return {
        "schema": "apollo_bag_audit/summary@1",
        "as_of": {"manifest": manifest_as_of, "summarize_generated": generated_at or utc_now_iso()},
        "denominators": {
            "manifest_total_bags": N,
            "pattern_matched": len(patterned),
            "pattern_unmatched": N - len(patterned),
            "sample_target": per_layer * len(layers),
            "sample_selected": len(sampled_names),
            "sample_parse_error": len(parse_errors),
            "classified": n_total,
        },
        "thresholds": {
            "level": "presence (G1 minimal four-channel; usability deferred to gate 2)",
            "min_hz": dict(min_hz),
            "aliases": {role: [list(p) for p in pairs] for role, pairs in aliases.items()},
        },
        "sampling": {
            "seed": seed, "per_layer": per_layer, "per_vehicle_cap": per_vehicle_cap,
            "min_cell": min_cell, "n_layers": len(layers),
            "vehicles_sampled": len([v for v in vcount if v]),
            "max_bags_per_vehicle": max(vcount.values()) if vcount else 0,
            "ci_note": "层内 srs Wilson CI；每车≤2 仅为限抽、不消除车辆聚集相关性；CI 未校正，可能偏窄。",
        },
        "per_layer": per_layer_out,
        "overall": {
            "pooled_sample_full4": {"x": x_total, "n": n_total,
                                    "p": (x_total / n_total) if n_total else None},
            "stratified_full4": {
                "estimate": est, "ci_normal95": ci,
                "effective_weight_sum": (weight_sum / N) if N else None,
                "unrepresented_layers": [l["layer_id"] for l in per_layer_out if l["n_classified"] == 0],
            },
            "seconds_weighted_full4": {
                "seconds_full4": seconds_full, "seconds_total": seconds_total,
                "p": (seconds_full / seconds_total) if seconds_total else None,
            },
            "cluster_bootstrap_ci95": bootstrap_ci,
            "bootstrap_note": "vehicle-level cluster bootstrap within strata (supplementary; review suggestion)",
        },
        "missing_combinations": missing_combinations,
        "parse_errors": parse_errors,
        "unmatched_channels": buckets,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_manifest(path: str):
    rows, as_of = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("manifest_meta"):
                as_of = obj.get("as_of")
                continue
            rows.append(obj)
    return rows, as_of


def load_aliases(path: str):
    if not path:
        return {k: tuple(v) for k, v in DEFAULT_ALIASES.items()}
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple((p[0], p[1]) for p in v) for k, v in raw.items()}


def parse_min_hz(spec: str):
    out = dict(DEFAULT_MIN_HZ)
    for part in spec.split(","):
        role, _, val = part.partition("=")
        out[role.strip()] = float(val)
    return out


def write_csv(path: str, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_manifest(args):
    entries = scan_bag_dir(args.bag_dir)
    rows = [build_manifest_row(n, s, m) for (n, s, m) in entries]
    as_of = utc_now_iso()
    with open(args.out, "w") as f:
        f.write(json.dumps({"manifest_meta": True, "as_of": as_of, "bag_dir": args.bag_dir,
                            "n_total": len(rows)}) + "\n")
        f.writelines(json.dumps(r) + "\n" for r in rows)
    ok = sum(1 for r in rows if r["pattern"] == "ok")
    print(f"manifest: N={len(rows)} pattern_ok={ok} unrecognized={len(rows) - ok} "
          f"as_of={as_of} -> {args.out}")


def cmd_sample(args):
    rows, _ = load_manifest(args.manifest)
    out = draw_sample(rows, per_layer=args.per_layer, seed=args.seed,
                      per_vehicle_cap=args.per_vehicle_cap, min_cell=args.min_cell)
    csv_rows = [{
        "bag": r["name"], "layer": r["layer"], "vehicle": r["vehicle"],
        "date": r["date"], "mode": r["mode"], "month": r["month"],
        "size": r["size"], "mtime": r["mtime"], "seed": args.seed,
    } for r in out["selected"]]
    write_csv(args.out, SAMPLE_COLUMNS, csv_rows)
    print(json.dumps({"layers": out["layers"],
                      "selected": len(out["selected"]),
                      "excluded_unpatterned": out["excluded_unpatterned"],
                      "seed": args.seed}, ensure_ascii=False, indent=2))


def cmd_audit(args):
    rows, _ = load_manifest(args.manifest)
    manifest_by_name = {r["name"]: r for r in rows}
    with open(args.samples) as f:
        sample_names = [r["bag"] for r in csv.DictReader(f)]

    prior = {}
    if args.prior_index and os.path.exists(args.prior_index):
        with open(args.prior_index) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    prior[obj["bag"]] = obj
    prior_channels = {}
    if args.prior_channels and os.path.exists(args.prior_channels):
        with open(args.prior_channels) as f:
            for row in csv.DictReader(f):
                prior_channels.setdefault(row["bag"], []).append({
                    "bag": row["bag"], "channel": row["channel"],
                    "message_type": row["message_type"], "n": int(row["n"] or 0),
                    "duration_s": row["duration_s"], "hz": row["hz"],
                    "first_offset_s": row["first_offset_s"],
                    "last_offset_s": row["last_offset_s"],
                    "read_path": row["read_path"],
                })

    channels_path = os.path.join(args.out_dir, "bag_channels.csv")
    index_path = os.path.join(args.out_dir, "audit_index.jsonl")
    errors_path = os.path.join(args.out_dir, "audit_errors.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_err = n_skip = 0
    t_start = time.monotonic()
    with open(channels_path, "w", newline="") as chan_f, \
            open(index_path, "w") as idx_f, open(errors_path, "w") as err_f:
        chan_w = csv.DictWriter(chan_f, fieldnames=CHANNEL_COLUMNS)
        chan_w.writeheader()
        for name in sample_names:
            mrow = manifest_by_name.get(name, {})
            prior_row = prior.get(name)
            if (prior_row and prior_row.get("status") == "ok"
                    and mrow.get("size") == prior_row.get("size")
                    and mrow.get("mtime") == prior_row.get("mtime")):
                n_skip += 1
                for r in prior_channels.get(name, []):
                    chan_w.writerow(r)
                idx_f.write(json.dumps({**prior_row, "resumed": True}) + "\n")
                print(f"[skip/resume] {name}", flush=True)
                continue
            path = os.path.join(args.bag_dir, name)
            st = os.stat(path)
            result = audit_one_bag(path)
            base = {"bag": name, "size": st.st_size, "mtime": st.st_mtime}
            if result["status"] == "ok":
                n_ok += 1
                for r in result["rows"]:
                    chan_w.writerow(dict(r, bag=name))
                idx_f.write(json.dumps({
                    **base, "status": "ok",
                    "header_message_number": result["header_message_number"],
                    "scanned_messages": result["scanned_messages"],
                    "n_channels": result["n_channels"],
                    "n_channels_with_messages": result["n_channels_with_messages"],
                    "elapsed_s": result["elapsed_s"],
                }) + "\n")
            else:
                n_err += 1
                err_f.write(json.dumps({**base, "reason": result["reason"],
                                        "detail": result["detail"],
                                        "elapsed_s": result["elapsed_s"]}) + "\n")
                idx_f.write(json.dumps({**base, "status": "parse_error",
                                        "reason": result["reason"],
                                        "elapsed_s": result["elapsed_s"]}) + "\n")
            print(f"[{result['status']}] {name} elapsed={result['elapsed_s']}s "
                  f"{result.get('reason', '')}", flush=True)
    print(f"audit: ok={n_ok} parse_error={n_err} skipped(resumed)={n_skip} "
          f"total_elapsed={time.monotonic() - t_start:.1f}s -> {args.out_dir}")


def cmd_summarize(args):
    manifest_rows, manifest_as_of = load_manifest(args.manifest)
    with open(args.samples) as f:
        sample_rows = [{"name": r["bag"], "vehicle": r.get("vehicle")}
                       for r in csv.DictReader(f)]
    with open(args.channels) as f:
        channel_rows = []
        for r in csv.DictReader(f):
            channel_rows.append({
                "bag": r["bag"], "channel": r["channel"], "message_type": r["message_type"],
                "n": int(r["n"] or 0), "duration_s": float(r["duration_s"] or 0),
                "hz": float(r["hz"] or 0), "read_path": r.get("read_path"),
            })
    error_rows = []
    if args.errors and os.path.exists(args.errors):
        with open(args.errors) as f:
            error_rows = [json.loads(line) for line in f if line.strip()]
    bad = [r for r in channel_rows if r["read_path"] != READ_PATH_SCAN]
    if bad:  # §4 guard: only section_scan rows may feed judgment
        raise SystemExit(f"non-scan rows found in {args.channels}: {bad[:3]}")

    aliases = load_aliases(args.aliases)
    min_hz = parse_min_hz(args.min_hz)
    summary = summarize_from_tables(
        manifest_rows=manifest_rows, sample_rows=sample_rows, channel_rows=channel_rows,
        error_rows=error_rows, aliases=aliases, min_hz=min_hz, seed=args.seed,
        per_layer=args.per_layer, per_vehicle_cap=args.per_vehicle_cap,
        min_cell=args.min_cell, bootstrap_reps=args.bootstrap_reps,
        manifest_as_of=manifest_as_of,
    )
    with open(args.out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    ov = summary["overall"]
    print(json.dumps({
        "denominators": summary["denominators"],
        "pooled": ov["pooled_sample_full4"],
        "stratified": ov["stratified_full4"]["estimate"],
        "ci_normal95": ov["stratified_full4"]["ci_normal95"],
        "bootstrap_ci95": ov["cluster_bootstrap_ci95"],
        "seconds_weighted": ov["seconds_weighted_full4"],
        "parse_errors": len(summary["parse_errors"]),
    }, ensure_ascii=False, indent=2))
    print(f"summary -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(prog="apollo_bag_audit",
                                description="lab42 Apollo bag four-channel coverage audit (G1 pre-conversion inventory)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="freeze directory enumeration snapshot")
    m.add_argument("--bag-dir", required=True)
    m.add_argument("--out", required=True, help="manifest.jsonl output path")
    m.set_defaults(func=cmd_manifest)

    s = sub.add_parser("sample", help="stratified sample roster (seeded)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--per-layer", type=int, default=33)
    s.add_argument("--per-vehicle-cap", type=int, default=2)
    s.add_argument("--min-cell", type=int, default=33)
    s.add_argument("--seed", type=int, default=20261118)
    s.add_argument("--out", required=True, help="samples.csv output path")
    s.set_defaults(func=cmd_sample)

    a = sub.add_parser("audit", help="per-bag section-scan parse (lab42)")
    a.add_argument("--manifest", required=True)
    a.add_argument("--samples", required=True)
    a.add_argument("--bag-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--prior-index", default=None,
                   help="previous audit_index.jsonl; bags with unchanged (name,size,mtime) and status ok are skipped")
    a.add_argument("--prior-channels", default=None,
                   help="previous bag_channels.csv; rows of skipped bags are carried over into the new output")
    a.set_defaults(func=cmd_audit)

    z = sub.add_parser("summarize", help="stratified aggregation + CI (thresholds are parameters)")
    z.add_argument("--manifest", required=True)
    z.add_argument("--samples", required=True)
    z.add_argument("--channels", required=True, help="bag_channels.csv from audit")
    z.add_argument("--errors", default=None, help="audit_errors.jsonl from audit")
    z.add_argument("--min-hz", default="localization=25,perception=2.5,planning=2.5,control=25")
    z.add_argument("--aliases", default=None, help="aliases json {role: [[topic,type],...]}; default = measured initial table")
    z.add_argument("--per-layer", type=int, default=33)
    z.add_argument("--per-vehicle-cap", type=int, default=2)
    z.add_argument("--min-cell", type=int, default=33)
    z.add_argument("--seed", type=int, default=20261118)
    z.add_argument("--bootstrap-reps", type=int, default=2000)
    z.add_argument("--out", required=True, help="summary.json output path")
    z.set_defaults(func=cmd_summarize)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
