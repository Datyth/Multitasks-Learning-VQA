#!/usr/bin/env bash
# Train exactly one experiment with `swift sft`. Experiments are never chained.
#
#   bash scripts/train.sh <experiment> [--config PATH] [--seed N] [--smoke] [--dry-run] [--force]
#
#   <experiment>  e1_t1 e1_t2 e1_t3 e1_t4 e2_joint e3_query e3_verify e4_no_relate e4_relate
#   --config      config file (default: config.yaml)
#   --seed        training seed (default: the config's dataset seed); other seeds go to runs/seed<N>/
#   --smoke       train.smoke_steps optimizer steps, written to runs/smoke/<experiment>
#   --dry-run     print the resolved command and exit; no model is loaded
#   --force       allow overwriting an existing run directory
#
# Set PYTHON to choose the interpreter (default: python on PATH).
set -euo pipefail

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

[[ $# -ge 1 ]] || { usage; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac
EXPERIMENT=$1; shift
SMOKE=(); SEED=(); CONFIG=config.yaml; DRY_RUN=0; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=(--smoke); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --config) CONFIG=${2:?--config needs a path}; shift 2 ;;
    --seed) SEED=(--seed "${2:?--seed needs a value}"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

CHECK=(); [[ $DRY_RUN -eq 1 ]] && CHECK=(--dry-run)
[[ -f $CONFIG ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
PREP=("$PY" src/prepare.py --config "$CONFIG")
RESOLVED=$("${PREP[@]}" train-args "$EXPERIMENT" ${SMOKE[@]+"${SMOKE[@]}"} ${SEED[@]+"${SEED[@]}"} ${CHECK[@]+"${CHECK[@]}"})

ENV_VARS=(); ARGS=(); IN_ARGS=0
while IFS= read -r line; do
  if [[ $IN_ARGS -eq 1 ]]; then ARGS+=("$line")
  elif [[ $line == "--" ]]; then IN_ARGS=1
  else ENV_VARS+=("$line"); fi
done <<< "$RESOLVED"

OUTPUT_DIR=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  [[ ${ARGS[i]} == "--output_dir" ]] && OUTPUT_DIR=${ARGS[i + 1]}
done

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s \\\n' "${ENV_VARS[@]}" "swift sft"
  printf '  %s \\\n' "${ARGS[@]}"
  echo "  # (dry run, nothing executed)"
  exit 0
fi

if [[ -d $OUTPUT_DIR ]] && [[ -n $(ls -A "$OUTPUT_DIR") ]] && [[ $FORCE -eq 0 ]]; then
  echo "Refusing to overwrite $OUTPUT_DIR. Pass --force or move it." >&2
  exit 3
fi
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"
"${PREP[@]}" run-meta "$EXPERIMENT" --phase start ${SMOKE[@]+"${SMOKE[@]}"} ${SEED[@]+"${SEED[@]}"}

START=$SECONDS
set +e
env "${ENV_VARS[@]}" swift sft "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
ELAPSED=$((SECONDS - START))

"${PREP[@]}" run-meta "$EXPERIMENT" --phase finish ${SMOKE[@]+"${SMOKE[@]}"} ${SEED[@]+"${SEED[@]}"} \
  --exit-code "$STATUS" --seconds "$ELAPSED"
echo "[$EXPERIMENT] exit=$STATUS wall=${ELAPSED}s output=$OUTPUT_DIR"
exit "$STATUS"
