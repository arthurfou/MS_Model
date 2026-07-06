#!/bin/bash
# Lance un entraînement ms_model sur le cluster SLURM du NUS SoC.
#
# Usage:
#   sbatch train_hpc.sh <config.yaml> [data_root] [wandb_name]
#
# Exemples:
#   sbatch scripts/train_hpc.sh configs/config_lstm_yogya.yaml
#   sbatch scripts/train_hpc.sh configs/config_lstm_yogya.yaml "" "run_test_v2"
#   sbatch scripts/train_hpc.sh configs/config_lstm_yogya.yaml /scratch/evimo/eval "run_test_v2"
#
# data_root et wandb_name sont optionnels (laisser "" pour skipper data_root).
#
# A ADAPTER avant le premier lancement :
#   --partition  : verifier les partitions GPU disponibles avec `sinfo`
#   --gres       : adapter selon le type de GPU voulu (ex: gpu:a100:1)
#   source ...   : chemin vers conda si different de ~/miniconda3

#SBATCH --job-name=ms_model
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.log

set -eo pipefail

CONFIG=${1:-configs/config_lstm_yogya.yaml}
DATA_ROOT=${2:-}
WANDB_NAME=${3:-}

mkdir -p logs

# Adapter ce chemin si conda est installe ailleurs (ex: ~/anaconda3, /opt/conda)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate devo

cd "$SLURM_SUBMIT_DIR"

CMD="python -m ms_model.training.train --config \"$CONFIG\""
[ -n "$DATA_ROOT"   ] && CMD="$CMD --data-root \"$DATA_ROOT\""
[ -n "$WANDB_NAME"  ] && CMD="$CMD --wandb-name \"$WANDB_NAME\""
eval $CMD
