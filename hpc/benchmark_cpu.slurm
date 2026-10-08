#!/bin/bash
#SBATCH --job-name=bench_cpu
#SBATCH --partition=Centaurus
# Note: On UNCC student/educational cluster (where gal-gpu nodes live), CPU partition is 'Centaurus'.
# If on the research cluster (Starlight), the CPU partition is 'Nebula' or 'Orion'.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# CPU-only run for an apples-to-apples comparison with the engine (which is CPU-only).
# Verify the partition name and limits with `sinfo` first; edit --partition if needed.
# Usage (from the repo root):  sbatch hpc/benchmark_cpu.slurm
DEVICE=cpu bash hpc/run_benchmark.sh
