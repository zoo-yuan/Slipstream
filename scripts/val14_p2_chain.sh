#!/bin/bash
# val14 gate-2 P2 chain (lab42): wait for the vegas-32 token file, then
#   1. CKPT=dummy wiring smoke on ONE scenario (closed_loop_reactive_agents_pufferdrive),
#   2. on smoke success, the champion 16x (R_ent5_off23 model_puffer_drive_006702.pt)
#      over the same 32 vegas tokens -> artifacts/val14/p2-champ32.
# Mounted as a single nohup background job; every stage logs to artifacts/val14/p2_chain.log.
# The city bin (nuplan__us-nv-las-vegas-strip.bin + .origin.json registration sidecar,
# residual 0.3682 m < 0.5 m) is already in artifacts/val14/bins.
#
# Usage (lab42): nohup bash scripts/val14_p2_chain.sh > artifacts/val14/p2_chain.log 2>&1 &
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V=$REPO/artifacts/val14
TOK=$V/p2_vegas32_tokens.txt
CKPT_CHAMP=$REPO/artifacts/t3-f16-16x/ckpt/R_ent5_off23/models/model_puffer_drive_006702.pt

echo "[p2_chain] $(date) waiting for $TOK"
# the 11 malformed dbs crash run_simulation's log discovery; the symlink farm
# (scripts/val14_link_val_dbs.py, 1180 good dbs) keeps /media/pc untouched
export NUPLAN_VAL_DB_DIR=$V/val_db_farm

for _ in $(seq 90); do  # up to 15 min for scripts/val14_pick_tokens.py
    [ -s "$TOK" ] && break
    sleep 10
done
if [ ! -s "$TOK" ]; then
    echo "[p2_chain] FAIL: $TOK still empty after 15 min (pick_tokens not done?)"
    exit 1
fi
wc -l "$TOK"

echo "[p2_chain] $(date) stage 1: dummy wiring smoke (1 scenario)"
FIRST=$(head -1 "$TOK")
if ! CKPT=dummy SCENARIO_TOKENS="$FIRST" LIMIT_TOTAL_SCENARIOS=1 GROUP=$V/p2-smoke1 \
        bash "$REPO/scripts/val14_run.sh"; then
    echo "[p2_chain] FAIL: smoke exited nonzero, champion run NOT started"
    exit 2
fi

echo "[p2_chain] $(date) stage 2: champion 16x over $(wc -l < "$TOK") vegas tokens"
if [ ! -f "$CKPT_CHAMP" ]; then
    echo "[p2_chain] FAIL: champion ckpt missing: $CKPT_CHAMP"
    exit 3
fi
CKPT=$CKPT_CHAMP SCENARIO_TOKENS="$(paste -sd, "$TOK")" GROUP=$V/p2-champ32 \
    bash "$REPO/scripts/val14_run.sh"
STATUS=$?
echo "[p2_chain] $(date) champion run exit $STATUS -> $V/p2-champ32"
exit $STATUS
