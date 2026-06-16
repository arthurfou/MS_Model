#!/bin/bash
# Lance un entraînement ms_model sur le cluster SLURM du NUS SoC.
#
# Usage:
#   sbatch train_hpc.sh <config.yaml> <data_root>
#
# Exemple:
#   sbatch scripts/train_hpc.sh configs/config_lstm_yogya.yaml /scratch/evimo/eval
#
# <data_root> est optionnel : s'il est absent, on utilise data.root du yaml.
#
# A ADAPTER avant le premier lancement :
#   --partition  : verifier les partitions GPU disponibles avec `sinfo`
#   --gres       : adapter selon le type de GPU voulu (ex: gpu:a100:1)
#   source ...   : chemin vers conda si different de ~/miniconda3

#SBATCH --job-name=ms_model
#SBATCH --partition=gpu          # TODO: verifier avec `sinfo`
#SBATCH --gres=gpu:1             # TODO: adapter au cluster (ex: gpu:a100:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG=${1:-configs/config_lstm_yogya.yaml}
DATA_ROOT=${2:-}

mkdir -p logs

# Adapter ce chemin si conda est installe ailleurs (ex: ~/anaconda3, /opt/conda)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate devo

cd "$SLURM_SUBMIT_DIR"

if [ -n "$DATA_ROOT" ]; then
    python -m ms_model.training.train --config "$CONFIG" --data-root "$DATA_ROOT"
else
    python -m ms_model.training.train --config "$CONFIG"
fi
