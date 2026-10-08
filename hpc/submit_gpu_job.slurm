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

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
export TORCH_HOME="${TORCH_HOME:-.torch_cache}"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$TORCH_HOME"

python -m training.train_local \
    --model "${MODEL:?set MODEL, e.g. MODEL=malaria_mobilenet}" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    ${TRAIN_ARGS:-}
