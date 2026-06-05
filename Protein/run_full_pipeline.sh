#!/usr/bin/env bash
set -euo pipefail

ROOT="/data2/jiangjiaqi/srzhang/InversionDock"
CODE_DIR="$ROOT/Code/protein_2"
DATA_DIR="$ROOT/Data/Skempi_dataset"
RUNS_DIR="$ROOT/runs/protein_2"
LOG_DIR="$RUNS_DIR/logs"
OUT_DIR="$ROOT/Data/Processed_skempi_backbone_aware"
STAGE2_CKPT_DIR="$RUNS_DIR/ckpts_stage2_backbone_aware"
STAGE3_CKPT_DIR="$RUNS_DIR/ckpts_stage3_structure_aware"
STAGE4_OUT_DIR="$RUNS_DIR/stage4_outputs"

SKEMPI_CSV="${SKEMPI_CSV:-$DATA_DIR/skempi_v2.csv}"
PDB_DIR="${PDB_DIR:-$DATA_DIR/Skempiv2}"
DEVICE="${DEVICE:-cuda:0}"
ENV_NAME="${ENV_NAME:-zsr-inversiondock}"

STAGE1_OVERWRITE="${STAGE1_OVERWRITE:-0}"
STAGE1_ETA="${STAGE1_ETA:-8}"
STAGE1_SIGMA_INIT="${STAGE1_SIGMA_INIT:-10.0}"
STAGE1_TARGET_POINTS="${STAGE1_TARGET_POINTS:-10000}"
STAGE1_FPS_RATIO="${STAGE1_FPS_RATIO:-0.05}"
STAGE1_KNN_K="${STAGE1_KNN_K:-32}"
STAGE1_ZETA="${STAGE1_ZETA:-8}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
STAGE2_SEQ_LEN="${STAGE2_SEQ_LEN:-512}"
STAGE2_MAX_RESIDUES="${STAGE2_MAX_RESIDUES:-256}"
STAGE2_K="${STAGE2_K:-32}"
STAGE2_AMP="${STAGE2_AMP:-1}"

STAGE3_EPOCHS="${STAGE3_EPOCHS:-10}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-2}"
STAGE3_SEQ_LEN="${STAGE3_SEQ_LEN:-512}"
STAGE3_MAX_RESIDUES="${STAGE3_MAX_RESIDUES:-256}"
STAGE3_K="${STAGE3_K:-32}"
STAGE3_USE_NEGATIVE_POSE="${STAGE3_USE_NEGATIVE_POSE:-1}"
STAGE3_AMP="${STAGE3_AMP:-1}"

STAGE4_NUM_RESIDUES="${STAGE4_NUM_RESIDUES:-30}"
STAGE4_REFINE_STEPS="${STAGE4_REFINE_STEPS:-100}"
STAGE4_REC_NPZ="${STAGE4_REC_NPZ:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGE4_PDB="$STAGE4_OUT_DIR/generated_${TIMESTAMP}.pdb"

mkdir -p "$LOG_DIR" "$RUNS_DIR" "$STAGE4_OUT_DIR" "$STAGE2_CKPT_DIR" "$STAGE3_CKPT_DIR"

if [[ ! -f "$SKEMPI_CSV" ]]; then
  echo "[ERROR] SKEMPI CSV not found: $SKEMPI_CSV" >&2
  exit 1
fi

if [[ ! -d "$PDB_DIR" ]]; then
  echo "[ERROR] PDB dir not found: $PDB_DIR" >&2
  exit 1
fi

if [[ ! -d "$CODE_DIR" ]]; then
  echo "[ERROR] Code dir not found: $CODE_DIR" >&2
  exit 1
fi

echo "[PIPELINE] Start: $(date)"
echo "[PIPELINE] ROOT=$ROOT"
echo "[PIPELINE] ENV_NAME=$ENV_NAME"
echo "[PIPELINE] DEVICE=$DEVICE"

source "/data2/jiangjiaqi/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$CODE_DIR"

echo "[PIPELINE] ===== Stage 1: preprocessing ====="
if [[ "$STAGE1_OVERWRITE" == "1" ]]; then
  python -u "$CODE_DIR/stage1_preprocessing.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --pdb_dir "$PDB_DIR" \
    --out_dir "$OUT_DIR" \
    --eta "$STAGE1_ETA" \
    --sigma_init "$STAGE1_SIGMA_INIT" \
    --target_points "$STAGE1_TARGET_POINTS" \
    --fps_ratio "$STAGE1_FPS_RATIO" \
    --knn_k "$STAGE1_KNN_K" \
    --zeta "$STAGE1_ZETA" \
    --device "$DEVICE" \
    --overwrite
else
  python -u "$CODE_DIR/stage1_preprocessing.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --pdb_dir "$PDB_DIR" \
    --out_dir "$OUT_DIR" \
    --eta "$STAGE1_ETA" \
    --sigma_init "$STAGE1_SIGMA_INIT" \
    --target_points "$STAGE1_TARGET_POINTS" \
    --fps_ratio "$STAGE1_FPS_RATIO" \
    --knn_k "$STAGE1_KNN_K" \
    --zeta "$STAGE1_ZETA" \
    --device "$DEVICE"
fi

echo "[PIPELINE] ===== Stage 2: pretrain ====="
if [[ "$STAGE2_AMP" == "1" ]]; then
  python -u "$CODE_DIR/stage2_pretrain.py" \
    --data_root "$OUT_DIR" \
    --epochs "$STAGE2_EPOCHS" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --seq_len "$STAGE2_SEQ_LEN" \
    --max_residues "$STAGE2_MAX_RESIDUES" \
    --K "$STAGE2_K" \
    --save_dir "$STAGE2_CKPT_DIR" \
    --device "$DEVICE" \
    --amp
else
  python -u "$CODE_DIR/stage2_pretrain.py" \
    --data_root "$OUT_DIR" \
    --epochs "$STAGE2_EPOCHS" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --seq_len "$STAGE2_SEQ_LEN" \
    --max_residues "$STAGE2_MAX_RESIDUES" \
    --K "$STAGE2_K" \
    --save_dir "$STAGE2_CKPT_DIR" \
    --device "$DEVICE"
fi

STAGE2_FINAL="$STAGE2_CKPT_DIR/final.pt"
if [[ ! -f "$STAGE2_FINAL" ]]; then
  echo "[ERROR] Stage 2 final checkpoint not found: $STAGE2_FINAL" >&2
  exit 1
fi

echo "[PIPELINE] ===== Stage 3: structure-aware docking ====="
if [[ "$STAGE3_USE_NEGATIVE_POSE" == "1" && "$STAGE3_AMP" == "1" ]]; then
  python -u "$CODE_DIR/stage3_train.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --npz_root "$OUT_DIR" \
    --epochs "$STAGE3_EPOCHS" \
    --batch_size "$STAGE3_BATCH_SIZE" \
    --seq_len "$STAGE3_SEQ_LEN" \
    --max_residues "$STAGE3_MAX_RESIDUES" \
    --K "$STAGE3_K" \
    --save_dir "$STAGE3_CKPT_DIR" \
    --device "$DEVICE" \
    --pretrained "$STAGE2_FINAL" \
    --use_negative_pose \
    --amp
elif [[ "$STAGE3_USE_NEGATIVE_POSE" == "1" ]]; then
  python -u "$CODE_DIR/stage3_train.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --npz_root "$OUT_DIR" \
    --epochs "$STAGE3_EPOCHS" \
    --batch_size "$STAGE3_BATCH_SIZE" \
    --seq_len "$STAGE3_SEQ_LEN" \
    --max_residues "$STAGE3_MAX_RESIDUES" \
    --K "$STAGE3_K" \
    --save_dir "$STAGE3_CKPT_DIR" \
    --device "$DEVICE" \
    --pretrained "$STAGE2_FINAL" \
    --use_negative_pose
elif [[ "$STAGE3_AMP" == "1" ]]; then
  python -u "$CODE_DIR/stage3_train.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --npz_root "$OUT_DIR" \
    --epochs "$STAGE3_EPOCHS" \
    --batch_size "$STAGE3_BATCH_SIZE" \
    --seq_len "$STAGE3_SEQ_LEN" \
    --max_residues "$STAGE3_MAX_RESIDUES" \
    --K "$STAGE3_K" \
    --save_dir "$STAGE3_CKPT_DIR" \
    --device "$DEVICE" \
    --pretrained "$STAGE2_FINAL" \
    --amp
else
  python -u "$CODE_DIR/stage3_train.py" \
    --skempi_csv "$SKEMPI_CSV" \
    --npz_root "$OUT_DIR" \
    --epochs "$STAGE3_EPOCHS" \
    --batch_size "$STAGE3_BATCH_SIZE" \
    --seq_len "$STAGE3_SEQ_LEN" \
    --max_residues "$STAGE3_MAX_RESIDUES" \
    --K "$STAGE3_K" \
    --save_dir "$STAGE3_CKPT_DIR" \
    --device "$DEVICE" \
    --pretrained "$STAGE2_FINAL"
fi

STAGE3_FINAL="$STAGE3_CKPT_DIR/final.pt"
if [[ ! -f "$STAGE3_FINAL" ]]; then
  echo "[ERROR] Stage 3 final checkpoint not found: $STAGE3_FINAL" >&2
  exit 1
fi

if [[ -z "$STAGE4_REC_NPZ" ]]; then
  STAGE4_REC_NPZ="$(ls "$OUT_DIR"/*.npz 2>/dev/null | sort | head -n 1 || true)"
fi

if [[ -z "$STAGE4_REC_NPZ" || ! -f "$STAGE4_REC_NPZ" ]]; then
  echo "[ERROR] Could not resolve Stage 4 receptor npz: $STAGE4_REC_NPZ" >&2
  exit 1
fi

echo "[PIPELINE] ===== Stage 4: generate backbone ====="
echo "[PIPELINE] Stage4 receptor npz: $STAGE4_REC_NPZ"
python -u "$CODE_DIR/stage4_generate.py" \
  --rec_npz "$STAGE4_REC_NPZ" \
  --stage3_ckpt "$STAGE3_FINAL" \
  --num_residues "$STAGE4_NUM_RESIDUES" \
  --output_pdb "$STAGE4_PDB" \
  --refine_steps "$STAGE4_REFINE_STEPS" \
  --device "$DEVICE"

echo "[PIPELINE] Done: $(date)"
echo "[PIPELINE] Stage4 output: $STAGE4_PDB"
