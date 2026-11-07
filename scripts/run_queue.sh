#!/usr/bin/env bash
# Run the manual commands in order, waiting for a free GPU before each GPU step.
# Started by hand, never by another script. One step at a time, on one GPU at a time.
#
#   bash scripts/run_queue.sh <stage>... [options]
#
#   <stage>             smoke | mandatory | exploratory | v2 | v2-seeds | v3-one-seed
#                       (v2: new-test E0, v1 E3/E4 adapters, E1-E4 seed 42; v2-seeds: seeds 43, 44)
#   --gpus LIST         GPUs allowed, e.g. 0,2 (default: all)
#   --poll-seconds N    seconds between checks (default: 60)
#   --stable-polls N    consecutive free checks before a GPU counts as free (default: 3)
#   --max-used-mib N    a free GPU uses at most this much memory (default: 1024)
#   --max-util N        a free GPU has at most this utilization % (default: 10)
#   --dry-run           print the plan and exit; touches no GPU
#
# The first failed step stops the queue. Re-running skips the steps listed in
# runs/queue/done.txt; delete a line there to run that step again. Output: runs/queue/queue.log
#
# Set PYTHON to choose the interpreter (default: python on PATH).
set -euo pipefail

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "[queue] $*" >&2; exit 2; }
log() { echo "[queue $(date '+%F %T')] $*"; }

STAGES=(); GPUS=""; POLL=60; STABLE=3; MAX_USED=1024; MAX_UTIL=10; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    smoke|mandatory|exploratory|v2|v2-seeds|v3-one-seed) STAGES+=("$1"); shift ;;
    --gpus) GPUS=${2:?--gpus needs a value}; shift 2 ;;
    --poll-seconds) POLL=${2:?--poll-seconds needs a value}; shift 2 ;;
    --stable-polls) STABLE=${2:?--stable-polls needs a value}; shift 2 ;;
    --max-used-mib) MAX_USED=${2:?--max-used-mib needs a value}; shift 2 ;;
    --max-util) MAX_UTIL=${2:?--max-util needs a value}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ ${#STAGES[@]} -ge 1 ]] || { usage; exit 2; }

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

V2=config.v2_1000train_500test.yaml
V3=config.v3_crossed_open.yaml

# v2 training and evaluation for one training seed, E1-E4. E3/E4 first (the main contrast), then E2,
# then E1; each group is evaluated as soon as it is trained, so partial results are usable.
v2_seed_steps() {
  local seed=$1 e
  train() { for e in "$@"; do echo "gpu|bash scripts/train.sh $e --config $V2 --seed $seed"; done; }
  evaluate() { for e in "$@"; do echo "gpu|python src/evaluate.py --config $V2 --experiment $e --seed $seed"; done; }
  train e3_query e3_verify e4_no_relate e4_relate
  evaluate e3 e4
  train e2_joint
  evaluate e2_joint
  train e1_t1 e1_t2 e1_t3 e1_t4
  evaluate e1
}

# One "<kind>|<command>" per step. kind: gpu waits for a free GPU, cpu runs immediately.
stage_steps() {
  case "$1" in
    smoke) cat <<'EOF'
cpu|bash scripts/train.sh e1_t1 --dry-run
gpu|bash scripts/train.sh e1_t1 --smoke --force
gpu|python src/evaluate.py --experiment e0 --limit 5 --overwrite
gpu|python src/evaluate.py --experiment e1_t1 --limit 5 --overwrite
cpu|python src/report.py --estimate --hours-available 24
EOF
    ;;
    mandatory) cat <<'EOF'
gpu|python src/evaluate.py --experiment e0
gpu|bash scripts/train.sh e1_t1
gpu|bash scripts/train.sh e1_t2
gpu|bash scripts/train.sh e1_t3
gpu|bash scripts/train.sh e1_t4
gpu|python src/evaluate.py --experiment e1
gpu|bash scripts/train.sh e2_joint
gpu|python src/evaluate.py --experiment e2_joint
cpu|python src/report.py
EOF
    ;;
    exploratory) cat <<'EOF'
gpu|bash scripts/train.sh e3_query
gpu|bash scripts/train.sh e3_verify
gpu|python src/evaluate.py --experiment e3
gpu|bash scripts/train.sh e4_no_relate
gpu|bash scripts/train.sh e4_relate
gpu|python src/evaluate.py --experiment e4
cpu|python src/report.py
EOF
    ;;
    v2)
      echo "gpu|python src/evaluate.py --config $V2 --experiment e0"
      echo "gpu|python src/evaluate.py --config $V2 --experiment e3 --adapters-from v1_400train"
      echo "gpu|python src/evaluate.py --config $V2 --experiment e4 --adapters-from v1_400train"
      v2_seed_steps 42
      echo "cpu|python src/report.py --config $V2"
    ;;
    v2-seeds)
      v2_seed_steps 43
      v2_seed_steps 44
      echo "cpu|python src/report.py --config $V2"
    ;;
    v3-one-seed)
      for e in e3_query e3_verify e4_no_relate e4_relate; do
        echo "gpu|bash scripts/train.sh $e --config $V3 --seed 42"
      done
      echo "gpu|python src/evaluate.py --config $V3 --experiment e3 --seed 42"
      echo "gpu|python src/evaluate.py --config $V3 --experiment e4 --seed 42"
      echo "cpu|python src/report.py --config $V3"
      echo "gpu|bash scripts/train.sh e2_joint --config $V3 --seed 42"
      echo "gpu|python src/evaluate.py --config $V3 --experiment e2_joint --seed 42"
      echo "cpu|python src/report.py --config $V3"
    ;;
  esac
}

if [[ $DRY_RUN -eq 1 ]]; then
  for stage in "${STAGES[@]}"; do
    stage_steps "$stage" | sed "s/^/$stage|/"
  done
  exit 0
fi

# Fail now rather than after hours of waiting.
command -v nvidia-smi >/dev/null || die "nvidia-smi not found."
command -v swift >/dev/null || die "swift not found. Run: conda activate datpt_ml"
"$PY" -c "import torch" 2>/dev/null || die "$PY cannot import torch. Run: conda activate datpt_ml"
NEEDS_V2=0
for stage in "${STAGES[@]}"; do [[ $stage == v2* ]] && NEEDS_V2=1; done
CHECK_CONFIG=config.yaml; [[ $NEEDS_V2 -eq 1 ]] && CHECK_CONFIG=$V2
for stage in "${STAGES[@]}"; do [[ $stage == v3-one-seed ]] && CHECK_CONFIG=$V3; done
"$PY" - "$CHECK_CONFIG" <<'EOF' || die "Prerequisites missing, see the message above."
import sys
sys.path.insert(0, "src")
from common import load_config, resolve
cfg = load_config(sys.argv[1])
missing = [str(p) for p in (resolve(cfg, "model_path"), resolve(cfg, "processed_dir") / "manifest.json") if not p.exists()]
if missing:
    sys.exit("Missing: " + ", ".join(missing))
EOF

QUEUE_DIR=runs/queue
DONE_FILE=$QUEUE_DIR/done.txt
mkdir -p "$QUEUE_DIR"
touch "$DONE_FILE"
exec > >(tee -a "$QUEUE_DIR/queue.log") 2>&1
exec 9>"$QUEUE_DIR/lock"
flock -n 9 || die "Another queue is already running in this repository."

# Sets GPU to the first allowed GPU that stays free for $STABLE consecutive checks.
wait_for_gpu() {
  local -A streak=()
  local polls=0 snapshot idx used util seen
  while :; do
    polls=$((polls + 1))
    if ! snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits); then
      log "nvidia-smi failed, retrying"
      sleep "$POLL"
      continue
    fi
    seen=0
    while IFS=', ' read -r idx used util; do
      [[ -z $GPUS || ",$GPUS," == *",$idx,"* ]] || continue
      seen=1
      if [[ $used =~ ^[0-9]+$ && $util =~ ^[0-9]+$ ]] && ((used <= MAX_USED && util <= MAX_UTIL)); then
        streak[$idx]=$((${streak[$idx]:-0} + 1))
        if ((${streak[$idx]} >= STABLE)); then
          GPU=$idx
          return 0
        fi
      else
        streak[$idx]=0
      fi
    done <<< "$snapshot"
    [[ $seen -eq 1 ]] || die "No GPU matches --gpus $GPUS."
    ((polls % 10 == 1)) && log "waiting for a free GPU: $(tr '\n' ';' <<< "$snapshot")"
    sleep "$POLL"
  done
}

for stage in "${STAGES[@]}"; do
  while IFS='|' read -r kind command; do
    key="$stage: $command"
    if grep -Fxq "$key" "$DONE_FILE"; then
      log "skip (already done) $key"
      continue
    fi
    read -r -a argv <<< "$command"
    [[ ${argv[0]} == python ]] && argv[0]=$PY

    gpu_env=()
    if [[ $kind == gpu ]]; then
      wait_for_gpu
      gpu_env=("VQA_GPU=$GPU")
      log "start on GPU $GPU: $key"
    else
      log "start: $key"
    fi

    start=$SECONDS
    status=0
    # stdin here is the step list; a child reading it would swallow the remaining steps.
    env ${gpu_env[@]+"${gpu_env[@]}"} "${argv[@]}" </dev/null || status=$?
    if [[ $status -ne 0 ]]; then
      log "FAILED (exit $status) after $((SECONDS - start))s: $key"
      log "Queue stopped. Fix the cause, then re-run this command to continue from this step."
      exit "$status"
    fi
    echo "$key" >> "$DONE_FILE"
    log "done in $((SECONDS - start))s: $key"
  done < <(stage_steps "$stage")
done
log "All requested stages finished: ${STAGES[*]}"
