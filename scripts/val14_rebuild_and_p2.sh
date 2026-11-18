#!/bin/bash
# lab42 gate-2 finisher: rebuild the drive kernel with MAX_ENTITIES_PER_CELL=256
# (approved A256), record the binding sha256, then mount the P2 chain
# (dummy smoke 1 scenario -> champion 16x over the 32 vegas tokens).
# Discipline: if a smoke/grid error still reports a cell >256, this stops by itself
# (val14_p2_chain.sh exits nonzero on smoke failure) — do NOT raise the constant again.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V=$REPO/artifacts/val14
LOG=$V/kernel_rebuild.log

echo "[rebuild] $(date) NO_TRAIN=1 build_ext (MAX_ENTITIES_PER_CELL=256)"
if ! (cd "$REPO" && NO_TRAIN=1 .venv/bin/python setup.py build_ext --inplace --force) > "$LOG" 2>&1; then
    echo "[rebuild] FAIL, see $LOG"
    exit 1
fi
SO=$REPO/pufferlib/ocean/drive/binding.cpython-311-x86_64-linux-gnu.so
SHA=$(sha256sum "$SO" | cut -d' ' -f1)
echo "[rebuild] ok binding sha256=$SHA" | tee "$V/kernel_sha.txt"
cd "$REPO"
exec bash scripts/val14_p2_chain.sh
