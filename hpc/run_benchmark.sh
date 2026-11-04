#!/bin/bash
# Shared body for the HPC benchmark jobs. Called by benchmark_gpu.slurm / benchmark_cpu.slurm.
#   DEVICE=cuda|cpu   (required)      RUN_ID=...   (default: hpc-$DEVICE-$SLURM_JOB_ID)
#   MODEL=malaria_mobilenet|scienceqa_smolvlm      (default: malaria_mobilenet)
#   TRAIN_ARGS / INFER_ARGS: extra flags forwarded to the trainer / inference benchmark
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=cuda or cpu}"
MODEL="${MODEL:-malaria_mobilenet}"
RUN_ID="${RUN_ID:-hpc-${DEVICE}-${MODEL}-${SLURM_JOB_ID:-local}}"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load anaconda3
source activate "${CONDA_ENV:-beowulf}"

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
export TORCH_HOME="${TORCH_HOME:-.torch_cache}"
export HF_HOME="${HF_HOME:-.hf_cache}"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$TORCH_HOME" "$HF_HOME"

THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Benchmark JSON is small, so it lives in the repo's results/ dir (commit/copy it freely).
RESULTS_DIR="${RESULTS_DIR:-results}"

echo "== Training ($MODEL on $DEVICE, run $RUN_ID) =="
python -m training.train_local \
    --model "$MODEL" --device "$DEVICE" --threads "$THREADS" \
    --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" \
    --results-dir "$RESULTS_DIR" --run-id "$RUN_ID" \
    --recipe full --epochs "${EPOCHS:-10}" ${TRAIN_ARGS:-}

echo "== Inference ($MODEL on $DEVICE, run $RUN_ID) =="
python -m benchmark.infer_local \
    --model "$MODEL" --device "$DEVICE" --threads "$THREADS" \
    --data-dir "$DATA_DIR" \
    --weights-dir "$OUTPUT_DIR/$MODEL" \
    --results-dir "$RESULTS_DIR" --run-id "$RUN_ID" ${INFER_ARGS:-}

echo "Done. Results in $RESULTS_DIR/$RUN_ID"
