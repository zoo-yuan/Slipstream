#!/bin/bash
# night-chain 2026-11-12/13: 8x(resumed, running) -> 16x -> X_cap2048 -> 32x(partial)
# Usage: nohup setsid bash scripts/night_chain.sh <pid-of-running-8x-orchestrator> &
# All stages on GPU0 only (GPU1 reserved for strata service per collab contract).
# Hard deadline: 2026-11-13 10:15 (owner window ends 10:30; buffer for eval/wrap-up).
set -u
cd "$(dirname "$0")/.."
LOG=artifacts/night-chain.log
echo "===== chain start $(date '+%F %T')" >> "$LOG"

WAIT_PID="${1:-518029}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
echo "===== stage0: 8x orchestrator $WAIT_PID exited $(date '+%F %T')" >> "$LOG"

DEADLINE=$(date -d "2026-11-13 10:15" +%s)
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage() {
  local name=$1 budget_s=$2
  shift 2
  local now cap
  now=$(date +%s)
  local avail=$(( DEADLINE - now ))
  if [ "$avail" -le 900 ]; then
    echo "===== stage $name SKIPPED: only ${avail}s to deadline $(date '+%F %T')" >> "$LOG"
    return 0
  fi
  cap=$(( budget_s < avail ? budget_s : avail ))
  echo "===== stage $name start $(date '+%F %T') (cap ${cap}s) args: $*" >> "$LOG"
  timeout --signal=TERM --kill-after=180 "$cap" \
    .venv/bin/python scripts/t3_sweep.py "$@" \
    >> "artifacts/night-chain-${name}.log" 2>&1
  echo "===== stage $name end rc=$? $(date '+%F %T')" >> "$LOG"
}

# stage 1: 16x pure scale (R recipe, 16 x 243269132 = 3892314112 steps)
run_stage 16x 22000 --seed 42 --only-arm R_ent5_off23 --arm-steps 3892314112 \
  --face 16x256 --art-dir artifacts/t3-f16-16x --total-budget-s 22000 --allow-kernel-drift

# stage 2: capacity first-open (backbone 1024->2048, R recipe verbatim, @8x steps)
run_stage cap2048 17000 --seed 42 --only-arm X_cap2048 --arm-steps 1946157056 \
  --face 16x256 --art-dir artifacts/t3-w3-X_cap2048 --total-budget-s 17000 --allow-kernel-drift

# stage 3: 32x (partial by design; TERM at deadline, rolling checkpoints preserved)
run_stage 32x 50000 --seed 42 --only-arm R_ent5_off23 --arm-steps 7784628224 \
  --face 16x256 --art-dir artifacts/t3-f16-32x --total-budget-s 50000 --allow-kernel-drift

echo "===== chain end $(date '+%F %T')" >> "$LOG"
