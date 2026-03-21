#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aiscuser"
SAIL_DIR="$ROOT/SAIL"
RUN_ROOT="$ROOT/sail_runs/full_cc3m_20260321"
OUT_ROOT="$ROOT/sail_data_full_cc3m_20260321"
DATA_SHARD_DIR="/blob/hwq/data/cc3m_wds_wbf"

mkdir -p "$RUN_ROOT" "$OUT_ROOT"

# Safety guard: never write to /blob
if [[ "$OUT_ROOT" == /blob/* || "$RUN_ROOT" == /blob/* ]]; then
  echo "[FATAL] output path points to /blob, aborting"
  exit 1
fi

cd "$ROOT"

echo "[STEP1] Full WebDataset -> SAIL embeddings (4-way parallel, read-only from /blob)"

# shard split: 288 shards -> 4 x 72 shards
CUDA_VISIBLE_DEVICES=0 python "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0000..0071}.tar" \
  --dataset-name cc3m_wds_wbf_full_p0_20260321 \
  --output-root "$OUT_ROOT" \
  --batch-size 256 \
  --image-key jpg \
  --text-key short \
  --extra-text-key long \
  --read-workers 16 \
  --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 \
  --text-model openai/clip-vit-base-patch32 \
  --agg-mode concat > "$RUN_ROOT/convert_p0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0072..0143}.tar" \
  --dataset-name cc3m_wds_wbf_full_p1_20260321 \
  --output-root "$OUT_ROOT" \
  --batch-size 256 \
  --image-key jpg \
  --text-key short \
  --extra-text-key long \
  --read-workers 16 \
  --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 \
  --text-model openai/clip-vit-base-patch32 \
  --agg-mode concat > "$RUN_ROOT/convert_p1.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0144..0215}.tar" \
  --dataset-name cc3m_wds_wbf_full_p2_20260321 \
  --output-root "$OUT_ROOT" \
  --batch-size 256 \
  --image-key jpg \
  --text-key short \
  --extra-text-key long \
  --read-workers 16 \
  --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 \
  --text-model openai/clip-vit-base-patch32 \
  --agg-mode concat > "$RUN_ROOT/convert_p2.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0216..0287}.tar" \
  --dataset-name cc3m_wds_wbf_full_p3_20260321 \
  --output-root "$OUT_ROOT" \
  --batch-size 256 \
  --image-key jpg \
  --text-key short \
  --extra-text-key long \
  --read-workers 16 \
  --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 \
  --text-model openai/clip-vit-base-patch32 \
  --agg-mode concat > "$RUN_ROOT/convert_p3.log" 2>&1 &
PID3=$!

echo "[INFO] conversion pids: $PID0 $PID1 $PID2 $PID3"
wait $PID0 $PID1 $PID2 $PID3

echo "[STEP2] Train SAIL alignment layer on full converted embeddings"
cd "$SAIL_DIR"

TEXT_LIST=(
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_20260321_short"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_20260321_short"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_20260321_short"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_20260321_short"
)
IMAGE_LIST=(
  "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_20260321"
  "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_20260321"
  "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_20260321"
  "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_20260321"
)
EXTRA_TEXT_LIST=(
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_20260321_long"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_20260321_long"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_20260321_long"
  "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_20260321_long"
)

python main.py \
  --text-embedding-list "${TEXT_LIST[@]}" \
  --image-embedding-list "${IMAGE_LIST[@]}" \
  --extra-text-embedding-list "${EXTRA_TEXT_LIST[@]}" \
  --dataset-type embedding \
  --seed 42 \
  --resume latest \
  --save-frequency 1 \
  --batch-size 32768 \
  --lr 1e-5 \
  --epochs 20 \
  --workers 8 \
  --optimizer lion \
  --siglip \
  --wd 1e-4 \
  --target-dimension 512 \
  --linear-type star \
  --width-factor 8 \
  --log-every-n-steps 10 \
  --name sail_full_cc3m_clipb32_20260321 \
  --report-to '' \
  --logit_scale 20 \
  --logit_bias -10 > "$RUN_ROOT/train.log" 2>&1

LATEST_CKPT=$(ls -1 "$SAIL_DIR/logs/sail_full_cc3m_clipb32_20260321/checkpoints"/epoch_*.pt | sort -V | tail -n 1)
echo "[INFO] latest_ckpt=$LATEST_CKPT"

echo "[STEP3] Evaluate on Winoground"
mkdir -p "$ROOT/sail_eval_data"
python eval.py \
  --head-weights-path "$LATEST_CKPT" \
  --task winoground \
  --vision-model openai/clip-vit-base-patch32 \
  --text-model openai/clip-vit-base-patch32 \
  --dataset_root_dir "$ROOT/sail_eval_data" \
  --batch_size 64 \
  --agg_mode concat \
  --width_factor 8 > "$RUN_ROOT/eval_winoground.log" 2>&1

echo "[DONE] Full experiment complete"
echo "[DONE] logs: $RUN_ROOT"
echo "[DONE] train artifacts: $SAIL_DIR/logs/sail_full_cc3m_clipb32_20260321"