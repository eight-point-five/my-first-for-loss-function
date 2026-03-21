#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/home/aiscuser/sail_runs/full_cc3m_20260321"
DATA_ROOT="/home/aiscuser/sail_data_full_cc3m_20260321"
TRAIN_LOG="$RUN_ROOT/train.log"
PIPELINE_LOG="$RUN_ROOT/pipeline.log"

show_counts() {
  python - <<'PY'
import glob
root='/home/aiscuser/sail_data_full_cc3m_20260321'
pts=glob.glob(root+'/**/*.pt', recursive=True)
print('[COUNT] total_pt_files=', len(pts))
for p in sorted(set([x.rsplit('/',1)[0] for x in pts])):
    if 'cc3m_wds_wbf_full_p' in p:
        c=len(glob.glob(p+'/*.pt'))
        print('[COUNT]', p, c)
PY
}

echo "[MONITOR] pipeline (Ctrl+C to stop): $PIPELINE_LOG"
stdbuf -oL -eL tail -F "$PIPELINE_LOG" &
TAIL_PID=$!
trap 'kill $TAIL_PID >/dev/null 2>&1 || true' EXIT

while true; do
  echo "\n================ $(date '+%F %T') ================"
  show_counts

  if [[ -f "$TRAIN_LOG" ]]; then
    echo "[MONITOR] train metrics (latest)"
    grep -E "Train Epoch:|Eval Epoch:|contrastive_loss|LR:|Logit Scale:" "$TRAIN_LOG" | tail -n 20 || true
  else
    echo "[MONITOR] train.log not created yet (still in conversion stage)"
  fi

  sleep 30
done
