#!/bin/bash
# gpu_return_chain: waits for GPU0 to free up (occupied ~4d by another user as of
# 2026-11-13), then runs the queued training arms serially. Deployed detached on lab42.
# Queue (priority order): 16xs43 resume -> cap2048 (needs near-exclusive card) -> V resume -> W fresh.
# All stages auto-eval on completion (t3_sweep built-in). Checkpoint resume = same art-dir relaunch.
set -u
cd "$(dirname "$0")/.."
LOG=artifacts/gpu-return-chain.log
echo "===== gpu-return chain start $(date '+%F %T')" >> "$LOG"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' '; }
we_running() { pgrep -f "t3_sweep.py" >/dev/null 2>&1; }

wait_gpu() { # $1 = max used MiB to consider free
  while :; do
    if we_running; then sleep 120; continue; fi
    used=$(gpu_used_mib)
    [ "${used:-99999}" -le "$1" ] && return 0
    sleep 600
  done
}

run_stage() {
  local name=$1 free_mib=$2; shift 2
  wait_gpu "$free_mib"
  echo "===== stage $name start $(date '+%F %T') (gpu_used=$(gpu_used_mib)MiB)" >> "$LOG"
  .venv/bin/python scripts/t3_sweep.py "$@" >> "artifacts/gpu-return-${name}.log" 2>&1
  echo "===== stage $name end rc=$? $(date '+%F %T')" >> "$LOG"
}

# 1) 16x seed-43 replication resume (74% checkpoint on disk, ~1h)
run_stage 16xs43 8000 --seed 43 --only-arm R_ent5_off23 --arm-steps 3892314112 \
  --face 16x256 --art-dir artifacts/t3-f16-16x-s43 --total-budget-s 20000 --allow-kernel-drift

# 2) capacity first-open retry (backbone 2048 OOMed at ~24GB shared; needs ~near-exclusive card)
run_stage cap2048 7000 --seed 42 --only-arm X_cap2048 --arm-steps 1946157056 \
  --face 16x256 --art-dir artifacts/t3-w3-X_cap2048 --total-budget-s 17000 --allow-kernel-drift

# 3) wave-2 dose arms (V has a checkpoint on disk; W fresh)
run_stage Varm 8000 --seed 42 --only-arm V_ent5_off1520 --face 16x256 \
  --art-dir artifacts/t3-w2-V_ent5_off1520 --total-budget-s 5000 --allow-kernel-drift
run_stage Warm 8000 --seed 42 --only-arm W_ent5_progv1 --face 16x256 \
  --art-dir artifacts/t3-w2-W_ent5_progv1 --total-budget-s 5000 --allow-kernel-drift

echo "===== gpu-return chain end $(date '+%F %T')" >> "$LOG"
