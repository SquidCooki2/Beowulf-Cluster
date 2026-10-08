#!/bin/bash
#SBATCH --job-name=bench_gpu
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Usage (from the repo root):  sbatch hpc/benchmark_gpu.slurm
DEVICE=cuda bash hpc/run_benchmark.sh
