#!/usr/bin/env python3
"""Discipline #4 (A256 approval): why 543/1118 val14 tokens resolved, not >=95%.

Ground truth established by this analysis: the local val cache is a per-chunk
extraction (1191 usable chunk dbs over 107+ distinct log prefixes) of the whole
v1.1 val split; the 1118 canonical val14 tokens live scattered across those logs
(107 distinct source prefixes resolve hits), NOT in any small 14-file subset —
the "14" in val14 refers to upstream's per-log db packaging, not to this cache.
So the gap is a chunk-coverage gap of the local extraction (plus 11 malformed
chunk dbs), reported here per source log prefix; P4's denominator must use the
resolved∩map counts (vegas 255/1118 total resolved 543).

Usage (lab42, repo root): .venv/bin/python scripts/val14_token_gap.py
Writes artifacts/val14/token_gap.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
V = REPO / "artifacts/val14"
DB_DIR = Path("/media/pc/data2/rino_we/nuplan/nuplan-v1.1-val-extracted/nuplan-v1.1_val/data/cache/val")
SPLITTER = Path.home() / "rino_we/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"


def main():
    tokens = json.loads((V / "val14_tokens.json").read_text())["tokens"]
    token_map = json.loads((V / "token_map.json").read_text())
    bad = {n if n.endswith(".db") else n + ".db" for n, _ in json.loads((V / "prep_report.json").read_text())["bad"]}

    all_chunks = {p.stem for p in DB_DIR.glob("*.db")}
    val_split_chunks = set(yaml.safe_load(SPLITTER.read_text())["log_splits"]["val"])

    resolved_prefixes = Counter(rec["db"].split("_veh-")[0] for rec in token_map.values())
    bad_prefixes = Counter(n.split("_veh-")[0] for n in bad)
    # chunks of the val split that our extraction lacks entirely
    missing_val_chunks = sorted(val_split_chunks - all_chunks)

    report = {
        "tokens_total": len(tokens),
        "tokens_resolved": len(token_map),
        "tokens_unresolved": len(tokens) - len(token_map),
        "resolved_fraction": round(len(token_map) / len(tokens), 4),
        "val_split_chunks_in_upstream_splitter": len(val_split_chunks),
        "val_split_chunks_present_locally": len(val_split_chunks & all_chunks),
        "val_split_chunks_missing_locally": len(missing_val_chunks),
        "local_chunks_total": len(all_chunks),
        "malformed_chunks": sorted(bad),
        "malformed_distinct_log_prefixes": dict(bad_prefixes),
        "resolved_by_source_log_prefix": dict(resolved_prefixes),
        "resolved_distinct_log_prefixes": len(resolved_prefixes),
        "by_map": {
            m: sum(1 for r in token_map.values() if r["map_name"] == m)
            for m in sorted({r["map_name"] for r in token_map.values()})
        },
    }
    print(json.dumps({k: v for k, v in report.items() if k != "resolved_by_source_log_prefix"}, indent=1))
    (V / "token_gap.json").write_text(json.dumps(report, indent=1))
    print(f"-> {V / 'token_gap.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
