#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONVERTER="/home/aiscuser/wds_to_sail_embeddings.py"
OUT_ROOT="${OUT_ROOT:-/home/aiscuser/sail_data_full_cc12m_20260327}"
DATE_TAG="${DATE_TAG:-20260327}"
BATCH_SIZE="${BATCH_SIZE:-256}"
READ_WORKERS="${READ_WORKERS:-16}"
DECODE_WORKERS="${DECODE_WORKERS:-12}"
FORCE="${FORCE:-0}"
PARALLEL="${PARALLEL:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PART_LOG_DIR="${PART_LOG_DIR:-/home/aiscuser/sail_runs/full_cc12m_20260327}"
DATA_DIR="${DATA_DIR:-/blob/hwq/data/cc12m_wds_unedited}"

VISION_MODEL="openai/clip-vit-base-patch32"
TEXT_MODEL="openai/clip-vit-base-patch32"

mkdir -p "$OUT_ROOT"
mkdir -p "$PART_LOG_DIR"

# Safety guard: never write outputs to /blob
if [[ "$OUT_ROOT" == /blob/* || "$PART_LOG_DIR" == /blob/* ]]; then
  echo "[FATAL] output path points to /blob, aborting"
  exit 1
fi

PARTS=(
  "p0|{0000..0543}"
  "p1|{0544..1087}"
  "p2|{1088..1631}"
  "p3|{1632..2175}"
)

run_part() {
  local part="$1"
  local span="$2"
  local gpu_id="${3:-}"
  local dataset_name="cc12m_wds_unedited_full_${part}_${DATE_TAG}"
  local meta_file="$OUT_ROOT/${dataset_name}_conversion_meta.json"
  local shards="$DATA_DIR/cc12m-train-${span}.tar"

  if [[ -f "$meta_file" && "$FORCE" != "1" ]]; then
    echo "[SKIP] $dataset_name already done"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" "$CONVERTER"
    --shards "$shards"
    --dataset-name "$dataset_name"
    --output-root "$OUT_ROOT"
    --batch-size "$BATCH_SIZE"
    --read-workers "$READ_WORKERS"
    --decode-workers "$DECODE_WORKERS"
    --image-key jpg
    --text-key txt
    --extra-text-key json:longLLA_captions
    --vision-model "$VISION_MODEL"
    --text-model "$TEXT_MODEL"
    --agg-mode concat
  )

  if [[ -n "$gpu_id" ]]; then
    echo "[RUN] $dataset_name on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
  else
    echo "[RUN] $dataset_name"
    "${cmd[@]}"
  fi
}

if [[ "$PARALLEL" == "1" ]]; then
  IFS=',' read -r -a gpu_arr <<< "$GPU_IDS"
  if [[ "${#gpu_arr[@]}" -eq 0 ]]; then
    echo "[WARN] empty GPU_IDS, fallback to sequential"
    PARALLEL="0"
  fi
fi

if [[ "$PARALLEL" == "1" ]]; then
  declare -a pids=()
  declare -a part_names=()
  idx=0
  for item in "${PARTS[@]}"; do
    IFS='|' read -r part span <<< "$item"
    gpu_id="${gpu_arr[$((idx % ${#gpu_arr[@]}))]}"
    part_log="$PART_LOG_DIR/convert_cc12m_${part}.log"
    run_part "$part" "$span" "$gpu_id" > "$part_log" 2>&1 &
    pid=$!
    pids+=("$pid")
    part_names+=("$part")
    echo "[LAUNCH] part=$part gpu=$gpu_id pid=$pid log=$part_log"
    idx=$((idx + 1))
  done

  failed=0
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    part="${part_names[$i]}"
    if wait "$pid"; then
      echo "[OK] part=$part finished"
    else
      echo "[ERR] part=$part failed"
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "[FAIL] one or more parts failed"
    exit 1
  fi
else
  for item in "${PARTS[@]}"; do
    IFS='|' read -r part span <<< "$item"
    run_part "$part" "$span"
  done
fi

echo "[DONE] cc12m clip-b32 embedding conversion finished"
