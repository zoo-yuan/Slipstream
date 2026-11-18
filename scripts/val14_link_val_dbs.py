#!/usr/bin/env python3
"""Symlink farm of the GOOD val dbs (prep_report.json's bad list excluded) into
artifacts/val14/val_db_farm, so run_simulation's log discovery never touches the
malformed dbs (it crashes hard on them) while the hard rule "lab42 只写 ~/、/tmp、
仓 artifacts/" keeps /media/pc untouched.

Usage (lab42, repo root): .venv/bin/python scripts/val14_link_val_dbs.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path("/media/pc/data2/rino_we/nuplan/nuplan-v1.1-val-extracted/nuplan-v1.1_val/data/cache/val")
DST = REPO / "artifacts/val14/val_db_farm"


def main():
    bad = {n for n, _ in json.loads((REPO / "artifacts/val14/prep_report.json").read_text())["bad"]}
    DST.mkdir(parents=True, exist_ok=True)
    for stale in DST.glob("*.db"):
        stale.unlink()
    dbs = sorted(p for p in SRC.glob("*.db") if p.name not in bad)
    for p in dbs:
        os.symlink(p, DST / p.name)
    n = len(list(DST.glob("*.db")))
    print(f"linked {n} good dbs into {DST} (excluded {len(bad)} malformed)")
    assert n == len(dbs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
