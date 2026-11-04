#!/bin/bash
#SBATCH --job-name=har_features
#SBATCH --output=logs/har_%j.out
#SBATCH --error=logs/har_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

# Run from the directory you submitted from. It must contain:
#   dataset/RawData/   process_data_utils.py   process_data_utils_fixed.py   main_fixed.py
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs output

# --- Python environment ---
eval "$(conda shell.bash hook)"
conda activate ml_ds_proj

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
python process_data_fixed.py
echo "Finished $(date)"