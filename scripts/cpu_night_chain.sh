#!/bin/bash
set -u
cd "$(dirname "$0")/.."
LOG=artifacts/cpu-night-chain.log
echo "===== cpu chain start $(date +%F\ %T)" >> "$LOG"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD

# 1) E1@1x speed-attainment replay (64 scenarios, zlib kept)
.venv/bin/python -m pufferlib.pufferl eval puffer_drive carla_fast \
  load_model_path=$PWD/artifacts/t5-e1-1x-eval/ckpt/model_puffer_drive_001855.pt \
  eval.num_scenarios=64 eval.render_scenarios=true eval.capture_observations=true eval.keep_zlib_replays=true \
  train.device=cpu train.compile=false \
  eval.output_dir_name=$PWD/artifacts/t5-e1-1x-eval/replay64 >> artifacts/cpu-chain-e1.log 2>&1
RD=$(find artifacts/t5-e1-1x-eval/replay64 -type d -name replays_zlib | head -1)
[ -n "$RD" ] && .venv/bin/python scripts/speed_attainment.py --replays-dir "$RD" \
  --out artifacts/t5-e1-1x-eval/attainment_64.json >> artifacts/cpu-chain-e1.log 2>&1
echo "===== e1 attainment done $(date +%F\ %T)" >> "$LOG"

# 2) 16x champion 1600-scenario big-sample (clean + mix)
for FACE in clean mix; do
  EXTRA=""; [ "$FACE" = mix ] && EXTRA="env.hazard_enable=true env.cosim_partner_slots=44"
  .venv/bin/python -m pufferlib.pufferl eval puffer_drive carla_fast \
    load_model_path=$PWD/artifacts/t3-f16-16x/ckpt/R_ent5_off23/models/model_puffer_drive_006702.pt \
    eval.num_scenarios=1600 $EXTRA train.device=cpu train.compile=false \
    eval.output_dir_name=$PWD/artifacts/t4-16x-bigeval/champ1600 eval.output_subdir=${FACE}_n1600 \
    >> artifacts/cpu-chain-1600.log 2>&1
  echo "===== 1600 $FACE done $(date +%F\ %T)" >> "$LOG"
done
echo "===== cpu chain end $(date +%F\ %T)" >> "$LOG"
