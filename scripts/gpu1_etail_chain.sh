#!/bin/bash
# E-series formal run on GPU1 (4x steps, serial): E1 -> E2 -> E3
set -u
cd "$(dirname "$0")/.."
LOG=artifacts/gpu1-etail-chain.log
echo "===== gpu1 E-chain start $(date +%F\ %T)" >> "$LOG"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ARM in E1_lane E2_lane_goal03 E3_lane_progv1; do
  if pgrep -f "t3_sweep.py" | grep -qv $$; then :; fi
  echo "===== stage $ARM start $(date +%F\ %T)" >> "$LOG"
  .venv/bin/python scripts/t3_sweep.py --seed 42 --only-arm $ARM --arm-steps 973076528 \
    --face 16x256 --art-dir artifacts/t5-${ARM}-4x --total-budget-s 16000 --allow-kernel-drift \
    >> artifacts/gpu1-${ARM}.log 2>&1
  echo "===== stage $ARM end rc=$? $(date +%F\ %T)" >> "$LOG"
done
echo "===== gpu1 E-chain end $(date +%F\ %T)" >> "$LOG"
