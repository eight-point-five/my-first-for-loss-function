#!/usr/bin/env bash
set -euo pipefail

# 用法：bash /home/aiscuser/tools/run_sail_full_cc3m_live_template.sh | tee /home/aiscuser/sail_runs/full_cc3m_live_$(date +%F_%H%M%S).log
# 特点：
# 1) 所有输出实时打印到屏幕（并可 tee 到日志）
# 2) python -u + stdbuf 避免缓冲，loss/epoch 变化可实时看到
# 3) 只读 /blob，所有写入都在 /home/aiscuser

ROOT="/home/aiscuser"
SAIL_DIR="$ROOT/SAIL"
RUN_ROOT="$ROOT/sail_runs/full_cc3m_live"
OUT_ROOT="$ROOT/sail_data_full_cc3m_live"
DATA_SHARD_DIR="/blob/hwq/data/cc3m_wds_wbf"

mkdir -p "$RUN_ROOT" "$OUT_ROOT"

if [[ "$OUT_ROOT" == /blob/* || "$RUN_ROOT" == /blob/* ]]; then
  echo "[FATAL] output path points to /blob, aborting"
  exit 1
fi

echo "[STEP1] conversion starts"
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python -u "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0000..0071}.tar" \
  --dataset-name cc3m_wds_wbf_full_p0_live \
  --output-root "$OUT_ROOT" --batch-size 256 --image-key jpg --text-key short --extra-text-key long \
  --read-workers 16 --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 --text-model openai/clip-vit-base-patch32 --agg-mode concat \
  > "$RUN_ROOT/convert_p0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python -u "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0072..0143}.tar" \
  --dataset-name cc3m_wds_wbf_full_p1_live \
  --output-root "$OUT_ROOT" --batch-size 256 --image-key jpg --text-key short --extra-text-key long \
  --read-workers 16 --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 --text-model openai/clip-vit-base-patch32 --agg-mode concat \
  > "$RUN_ROOT/convert_p1.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python -u "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0144..0215}.tar" \
  --dataset-name cc3m_wds_wbf_full_p2_live \
  --output-root "$OUT_ROOT" --batch-size 256 --image-key jpg --text-key short --extra-text-key long \
  --read-workers 16 --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 --text-model openai/clip-vit-base-patch32 --agg-mode concat \
  > "$RUN_ROOT/convert_p2.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python -u "$ROOT/wds_to_sail_embeddings.py" \
  --shards "$DATA_SHARD_DIR/cc3m-train-{0216..0287}.tar" \
  --dataset-name cc3m_wds_wbf_full_p3_live \
  --output-root "$OUT_ROOT" --batch-size 256 --image-key jpg --text-key short --extra-text-key long \
  --read-workers 16 --decode-workers 12 \
  --vision-model openai/clip-vit-base-patch32 --text-model openai/clip-vit-base-patch32 --agg-mode concat \
  > "$RUN_ROOT/convert_p3.log" 2>&1 &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "[STEP1] conversion done"

echo "[STEP2] training starts"
cd "$SAIL_DIR"

python -u main.py \
  --text-embedding-list \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_live_short" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_live_short" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_live_short" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_live_short" \
  --image-embedding-list \
    "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_live" \
    "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_live" \
    "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_live" \
    "$OUT_ROOT/image_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_live" \
  --extra-text-embedding-list \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p0_live_long" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p1_live_long" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p2_live_long" \
    "$OUT_ROOT/text_embedding/openai-clip-vit-base-patch32/cc3m_wds_wbf_full_p3_live_long" \
  --dataset-type embedding --seed 42 --resume latest --save-frequency 1 \
  --batch-size 32768 --lr 1e-5 --epochs 20 --workers 8 --optimizer lion --siglip --wd 1e-4 \
  --target-dimension 512 --linear-type star --width-factor 8 --log-every-n-steps 10 \
  --name sail_full_cc3m_clipb32_live --report-to '' --logit_scale 20 --logit_bias -10 \
  2>&1 | tee "$RUN_ROOT/train.log"

echo "[STEP2] training done"
