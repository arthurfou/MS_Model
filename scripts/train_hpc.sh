#!/bin/bash
# Lance un entraînement ms_model sur le cluster SLURM du NUS SoC.
#
# Usage: sbatch train_hpc.sh [chemin/vers/config.yaml]
#
# ATTENTION: --partition, --gres et le chemin conda ci-dessous sont des
# placeholders à adapter aux specs réelles du cluster (cf. `sinfo`,
# documentation SoC Compute Cluster).

#SBATCH --job-name=ms_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG=${1:-configs/convlstm_evimo.yaml}

mkdir -p logs

# Adapter le chemin selon l'installation conda sur le cluster
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate devo

cd "$SLURM_SUBMIT_DIR"

python -m ms_model.training.train --config "$CONFIG"
