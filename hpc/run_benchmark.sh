#!/bin/bash
# Shared body for the HPC benchmark jobs. Called by benchmark_gpu.slurm / benchmark_cpu.slurm.
#   DEVICE=cuda|cpu   (required)      RUN_ID=...   (default: hpc-$DEVICE-$SLURM_JOB_ID)
#   TRAIN_ARGS / INFER_ARGS: extra flags forwarded to the trainer / inference benchmark
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=cuda or cpu}"
RUN_ID="${RUN_ID:-hpc-${DEVICE}-${SLURM_JOB_ID:-local}}"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load anaconda3
source activate "${CONDA_ENV:-beowulf}"

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
export TORCH_HOME="${TORCH_HOME:-.torch_cache}"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$TORCH_HOME"

THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Benchmark JSON is small, so it lives in the repo's results/ dir (commit/copy it freely).
RESULTS_DIR="${RESULTS_DIR:-results}"

echo "== Training ($DEVICE, run $RUN_ID) =="
python -m training.train_local \
    --device "$DEVICE" --threads "$THREADS" \
    --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
    --results-dir "$RESULTS_DIR" --run-id "$RUN_ID" \
    --recipe full --epochs "${EPOCHS:-10}" ${TRAIN_ARGS:-}

echo "== Inference ($DEVICE, run $RUN_ID) =="
python -m benchmark.infer_local \
    --device "$DEVICE" --threads "$THREADS" \
    --data-dir "$DATA_DIR" \
    --weights-dir "$OUTPUT_DIR/malaria_mobilenet" \
    --results-dir "$RESULTS_DIR" --run-id "$RUN_ID" ${INFER_ARGS:-}

echo "Done. Results in $RESULTS_DIR/$RUN_ID"
