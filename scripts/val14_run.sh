#!/bin/bash
# val14 gate-2 runner: PufferDrive policy under nuPlan's unmodified closed-loop
# run_simulation.py on lab42 (no slurm/apptainer). Stripped adaptation of
# pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh with two hard rulings from the
# locked proposal r1 + deepseek review:
#   - CHALLENGES is FORCED to closed_loop_reactive_agents_pufferdrive only (review
#     blocking #2: the upstream default runs both challenges; a mixed headline score
#     is not comparable to TerraZero 94.19's reactive score). nonreactive is a
#     separate, later run -- do not widen this script.
#   - worker=sequential by default (upstream only ever exercised sequential end to
#     end; ray is deferred until the sequential extrapolation exceeds two nights).
#
# Usage:
#   CKPT=dummy SCENARIO_TOKENS=<token> LIMIT_TOTAL_SCENARIOS=1 bash scripts/val14_run.sh
#   CKPT=artifacts/t3-f16-16x/ckpt/R_ent5_off23/models/model_puffer_drive_006702.pt \
#     SCENARIO_TOKENS="$(cat artifacts/val14/p2_vegas32_tokens.txt | paste -sd,)" \
#     GROUP=artifacts/val14/p2-champ32 bash scripts/val14_run.sh
#
# Required env:
#   CKPT     PufferDrive checkpoint .pt, or "dummy" for the no-policy wiring test.
# Optional env (defaults in [brackets]):
#   PD [repo root from script location]   NUPLAN_DEVKIT_ROOT [editable devkit under
#   /media/pc/data2/rino_we/nuplan-devkit] NUPLAN_MAPS_ROOT [~/rino_we/nuplan/maps_root]
#   NUPLAN_VAL_DB_DIR [data2 val cache]   CITY_BIN_DIR [artifacts/val14/bins]
#   NUPLAN_EXP_ROOT [~/val14_exp]         SPLIT [val14_split]
#   SCENARIO_TOKENS [unset = SPLIT's own 1118]  LIMIT_TOTAL_SCENARIOS [unset]
#   WORKER [sequential] THREADS_PER_NODE [unset] GROUP [artifacts/val14/run_<ts>]
#   COSIM_OBS_HTML [failures] like upstream (all|failures|infractions|0); obs replays
#   land under $GROUP/obs_html, .replay.zlib kept (render deferred).
#   CALLBACKS [simulation_log_callback] -- upstream also renders carl ground-truth
#   videos (carl_visualization_callback); off by default here, they cost wall clock.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD=${PD:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-$PD/.venv/bin/python}
CARL_RUNTIME=${CARL_RUNTIME:-/home/zoo/carl_nuplan_runtime}
export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/media/pc/data2/rino_we/nuplan-devkit}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/home/zoo/rino_we/nuplan/maps_root}
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/media/pc/data2/rino_we/nuplan}
export NUPLAN_VAL_DB_DIR=${NUPLAN_VAL_DB_DIR:-/media/pc/data2/rino_we/nuplan/nuplan-v1.1-val-extracted/nuplan-v1.1_val/data/cache/val}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$HOME/val14_exp}
CITY_BIN_DIR=${CITY_BIN_DIR:-$PD/artifacts/val14/bins}
export PYTHONPATH="$PD:$CARL_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$NUPLAN_EXP_ROOT"

: "${CKPT:?set CKPT=/path/to/model_xxx.pt (or CKPT=dummy for the wiring test)}"
CHALLENGE="closed_loop_reactive_agents_pufferdrive"  # FORCED, see header
SPLIT=${SPLIT:-val14_split}
WORKER=${WORKER:-sequential}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP=${GROUP:-$PD/artifacts/val14/run_${TIMESTAMP}}
OBS_HTML=${COSIM_OBS_HTML:-failures}
[ "$OBS_HTML" = "1" ] && OBS_HTML=all

EXTRA_ARGS=()
[ "$OBS_HTML" != "0" ] && EXTRA_ARGS+=("planner.pufferdrive_planner.obs_html_dir=$GROUP/obs_html" "planner.pufferdrive_planner.obs_html_render=false")
EXTRA_ARGS+=("planner.pufferdrive_planner.city_bin_dir=$CITY_BIN_DIR")
[ -n "${GOAL_SOURCE:-}" ] && EXTRA_ARGS+=("planner.pufferdrive_planner.goal_source=$GOAL_SOURCE")
[ -n "${WORKER:-}" ] && EXTRA_ARGS+=("worker=$WORKER")
[ -n "${THREADS_PER_NODE:-}" ] && EXTRA_ARGS+=("worker.threads_per_node=$THREADS_PER_NODE")
[ -n "${LIMIT_TOTAL_SCENARIOS:-}" ] && EXTRA_ARGS+=("scenario_filter.limit_total_scenarios=$LIMIT_TOTAL_SCENARIOS")
if [ -n "${SCENARIO_TOKENS:-}" ]; then
    # bare 16-hex tokens parse as floats in hydra; pass a JSON list (the devkit's own
    # error message's fix), built from the comma-separated SCENARIO_TOKENS env
    TOKENS_JSON=$(printf '%s' "$SCENARIO_TOKENS" | awk -F, '{for(i=1;i<=NF;i++) printf "\"%s\"%s", $i, (i<NF?",":"")}')
    EXTRA_ARGS+=("scenario_filter.scenario_tokens=[$TOKENS_JSON]")
fi
[ -n "${EXTRA_HYDRA:-}" ] && EXTRA_ARGS+=($EXTRA_HYDRA)

cd "$PD"
echo "[val14_run] challenge=$CHALLENGE split=$SPLIT worker=$WORKER group=$GROUP ckpt=$CKPT"
"$PY" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
    +simulation=$CHALLENGE \
    scenario_builder=nuplan \
    scenario_builder.data_root="$NUPLAN_VAL_DB_DIR" \
    scenario_filter=$SPLIT \
    planner=pufferdrive_planner \
    planner.pufferdrive_planner.checkpoint_path="$CKPT" \
    callback="${CALLBACKS:-[simulation_log_callback]}" \
    main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, csv_main_callback]" \
    hydra.searchpath="[pkg://pufferlib.ocean.cosim.nuplan.config, pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    group="$GROUP" \
    "${EXTRA_ARGS[@]}"
STATUS=$?
echo "[val14_run] $CHALLENGE exit ${STATUS}"
echo "group -> $GROUP"

if [ "$OBS_HTML" = "all" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score 1.01
elif [ "$OBS_HTML" = "failures" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score "${COSIM_OBS_HTML_MAX_SCORE:-0.9}" --prune
elif [ "$OBS_HTML" = "infractions" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score 0 --prune \
        --metric no_ego_at_fault_collisions --metric drivable_area_compliance \
        --metric driving_direction_compliance --metric ego_is_making_progress
fi
exit $STATUS
