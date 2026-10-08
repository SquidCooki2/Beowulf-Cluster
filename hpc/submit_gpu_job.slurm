#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Usage (from the repo root):  MODEL=malaria_mobilenet sbatch hpc/submit_gpu_job.slurm
# Extra trainer args can be passed via TRAIN_ARGS="--epochs 5".
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load anaconda3
# One-time setup:  conda create -n beowulf python=3.10 && source activate beowulf && pip install -r requirements.txt
source activate "${CONDA_ENV:-beowulf}"

# Keep ALL data/outputs in scratch (home has a 5M inode limit).
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_DIR/torch_cache}"   # pretrained weights cache
mkdir -p "$SCRATCH_DIR"

python -m training.train_local \
    --model "${MODEL:?set MODEL, e.g. MODEL=malaria_mobilenet}" \
    --data-dir "$SCRATCH_DIR/data" \
    --output-dir "$SCRATCH_DIR/output" \
    ${TRAIN_ARGS}
