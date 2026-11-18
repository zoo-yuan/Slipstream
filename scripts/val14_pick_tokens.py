#!/usr/bin/env python3
"""Resolve val14 split tokens against the local sensor dbs THROUGH THE DEVKIT LOADER.

The val dbs are sensor dbs: they have no `scenario` table, and the val14 tokens are
initial lidar_pc tokens resolved by nuplan.database.nuplan_db.nuplan_scenario_queries.
get_scenarios_from_db -- the loader run_simulation's own scenario filter uses -- so
hand-rolled SELECTs are off-limits (gate-2 task ruling; the dead end is recorded in
artifacts/val14/tokens_missing.json: 1118/1118 "missing" via the naive route).

Scans the db dir (skipping the corrupted dbs from artifacts/val14/prep_report.json),
writes artifacts/val14/token_map.json (token -> {db, map_name} for every resolved
val14 token; the P4 intersection report's raw material) and prints the first N tokens
of one map for SCENARIO_TOKENS injection.

Usage (lab42):
  PYTHONPATH=<repo> .venv/bin/python scripts/val14_pick_tokens.py \
      --tokens artifacts/val14/val14_tokens.json --db-dir $NUPLAN_VAL_DB_DIR \
      --map us-nv-las-vegas-strip --limit 32 --out artifacts/val14/p2_vegas32_tokens.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", required=True, help="val14_tokens.json (the 1118 canonical list)")
    parser.add_argument("--db-dir", required=True, help="val sensor db cache dir")
    parser.add_argument("--prep-report", default=str(REPO_ROOT / "artifacts/val14/prep_report.json"),
                        help="prep_report.json whose `bad` list is excluded")
    parser.add_argument("--map", default=None, help="map name to pick tokens from (default: report only)")
    parser.add_argument("--limit", type=int, default=32, help="how many tokens of --map to print")
    parser.add_argument("--out", default=None, help="write the picked tokens to this file (one per line)")
    parser.add_argument("--token-map", default=str(REPO_ROOT / "artifacts/val14/token_map.json"),
                        help="where to write the full token -> {db, map} resolution")
    args = parser.parse_args()

    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    tokens = json.loads(Path(args.tokens).read_text())
    assert "tokens" in tokens and "n_tokens" in tokens, "val14_tokens.json schema"
    want = list(tokens["tokens"])
    bad = {name for name, _err in json.loads(Path(args.prep_report).read_text())["bad"]}
    dbs = sorted(p for p in Path(args.db_dir).glob("*.db") if p.name not in bad)

    token_map = {}
    t0 = time.time()
    for i, db in enumerate(dbs):
        try:
            rows = get_scenarios_from_db(str(db), want, None, None)
        except Exception as e:  # a db can still fail under the loader; report, don't abort
            print(f"[{i+1}/{len(dbs)}] {db.name}: LOADER ERROR {e}", file=sys.stderr)
            continue
        hit = 0
        for row in rows:
            tok = row["token"]
            if isinstance(tok, (bytes, bytearray)):  # the db stores raw 16-byte blobs
                tok = tok.hex()
            token_map[tok] = {"db": db.name, "map_name": row["map_name"]}
            hit += 1
        if hit or (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(dbs)}] {db.name}: {hit} val14 tokens ({time.time()-t0:.0f}s)", file=sys.stderr)

    maps = {}
    for rec in token_map.values():
        maps[rec["map_name"]] = maps.get(rec["map_name"], 0) + 1
    report = {
        "tokens_total": tokens["n_tokens"],
        "tokens_resolved": len(token_map),
        "tokens_unresolved": tokens["n_tokens"] - len(token_map),
        "by_map": maps,
        "dbs_scanned": len(dbs),
        "dbs_skipped_bad": len(bad),
    }
    print(json.dumps(report, indent=1))
    Path(args.token_map).write_text(json.dumps(token_map, indent=1))
    print(f"token map ({len(token_map)} rows) -> {args.token_map}", file=sys.stderr)

    if args.map:
        picked = [t for t in want if t in token_map and token_map[t]["map_name"] == args.map][: args.limit]
        for t in picked:
            print(t)
        if args.out:
            Path(args.out).write_text("\n".join(picked) + "\n")
            print(f"{len(picked)} {args.map} tokens -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
