#!/bin/bash
set -e

# -----------------------------
# Global configuration
# -----------------------------
PYTHON=python3
SCRIPT=run_bbbc021_ablation.py

DATA_DIR="/home/jovyan/es_ppo_data_original/data/bbbc021_all"
METADATA_FILE="metadata/bbbc021_df_all.csv"

UNET_CHANNELS="192 384 768 768"
PERT_EMBED_DIM=768
BATCH_SIZE=64

# Evaluation sample sizes
EVAL_SAMPLES_LIST=(1000 2500 5000)

# -----------------------------
# Checkpoints (FIXED PATHS)
# -----------------------------
CHECKPOINTS=(
  "bbbc021_ablation_results/run_20251231_181140_single/models/PPO/config_0/PPO_config_0_best_epoch_0003_fid_18.98.pt"
  "global_pretrained_models/ddpm_base_latest.pt"
  "bbbc021_ablation_results/run_20251231_181140_single/models/ES/config_0/ES_config_0_latest_epoch_0007_fid_48.14.pt"
)

# -----------------------------
# Run evaluations
# -----------------------------
for CKPT in "${CHECKPOINTS[@]}"; do
  echo "============================================================"
  echo "Evaluating checkpoint:"
  echo "  $CKPT"
  echo "============================================================"

  for N in "${EVAL_SAMPLES_LIST[@]}"; do
    echo ""
    echo ">>> Running evaluation with ${N} samples"
    echo ""

    $PYTHON $SCRIPT \
      --mode evaluate \
      --checkpoint-path "$CKPT" \
      --eval-samples "$N" \
      --eval-batch-size "$BATCH_SIZE" \
      --data-dir "$DATA_DIR" \
      --metadata-file "$METADATA_FILE" \
      --unet-channels $UNET_CHANNELS \
      --use-molformer \
      --perturbation-embed-dim "$PERT_EMBED_DIM" \
      --use-ema \
      --aux-device cuda

    echo ""
    echo "✓ Completed evaluation with ${N} samples"
    echo ""
  done

  echo "============================================================"
  echo "✓ Finished all evaluations for checkpoint: $CKPT"
  echo "============================================================"
  echo ""
done

echo "✅ All evaluations completed successfully."
