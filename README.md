# Beowulf Cluster: Heterogeneous 1F1B Distributed AI Engine

This repository contains the codebase and orchestration tools for a 5-node Beowulf Cluster designed for distributed deep learning training and inference on low-power, heterogeneous office PCs (3x HP ProDesk 400 G5 + 2x Lenovo ThinkCentre M900).

## 🚀 Key Architectural Innovations

1. **Heterogeneous 1F1B (One-Forward-One-Backward) Pipeline Parallelism**:
   - Bounds peak activation RAM per machine to $p=5$ stages.
   - Replaces heavy all-to-all AllReduce gradient broadcasts with lightweight point-to-point boundary activation streaming over ZeroMQ.
2. **Adaptive FLOPS-Proportional Stage Balancing**:
   - Benchmarks node CPU FLOPS at startup.
   - Proportionally allocates layers (e.g. 4-5 layers to 8th/9th Gen HPs, 2-3 layers to 6th Gen Lenovos) to eliminate pipeline stall bubbles.
3. **Malaria Cell Classification**:
   - Pretrained MobileNetV2 classifying parasitized vs uninfected blood-smear cells (NIH dataset, ~27.5k images).

---

## 📁 Repository Structure

```
.
├── config.json             # Cluster IP mapping, port definitions, and stage assignments
├── requirements.txt        # Python dependencies (torch, pyzmq, torchvision, numpy)
├── network/
│   ├── __init__.py
│   └── transceiver.py      # Non-blocking point-to-point ZeroMQ tensor transport
├── engine/
│   ├── __init__.py
│   ├── partitioner.py      # FLOPS-weighted heterogeneous layer partitioner
│   ├── scheduler.py        # 1F1B state machine (Warmup -> 1F1B Steady State -> Cooldown)
│   ├── runner.py           # Real-data training / inference loops for one stage
│   └── inference.py        # Pipelined forward-only inference with end-to-end acks
├── models/
│   ├── __init__.py
│   ├── registry.py         # Model name -> stages, input shape, loss, dataset
│   ├── stage_utils.py      # Splits a model into N sequential stages
│   └── malaria_mobilenet.py # Pretrained MobileNetV2 (malaria cells) sliced into stages
├── data/                   # Dataset loader (malaria, shared fixed split)
├── training/
│   └── train_local.py      # Single-device trainer (HPC GPU/CPU node / laptop)
├── benchmark/
│   ├── metrics.py          # Shared timing / system-info / JSON helpers
│   ├── infer_local.py      # Inference benchmark for the HPC side
│   ├── launch_local.py     # Run all engine stages on one machine (dev / smoke tests)
│   └── compare.py          # Merge engine + HPC results into a table, speedups and plot
├── hpc/
│   ├── benchmark_gpu.slurm # HPC GPU benchmark job (train + inference)
│   ├── benchmark_cpu.slurm # HPC CPU-only benchmark job (train + inference)
│   ├── run_benchmark.sh    # Shared job body
│   └── submit_gpu_job.slurm # Plain training job (no benchmark JSON/inference)
├── run_stage.py            # Main node daemon executed on each physical machine (--mode synthetic|train|infer)
├── launch_cluster.sh       # Bash bootstrap script to launch all 5 nodes over SSH (forwards args)
└── docs/                   # Research blueprints, plus hpc_setup_and_benchmarking.md
```

---

## 🔁 Engine vs HPC Benchmark

Every model in `models/registry.py` is a list of stages. The engine runs stage *i* on node *i*; the HPC trainer joins the stages into one `nn.Sequential`. Both use the same data split and recipe, so training speed, time-to-accuracy and inference latency/throughput can be compared. Full setup: [docs/hpc_setup_and_benchmarking.md](docs/hpc_setup_and_benchmarking.md).

```bash
# UNC Charlotte HPC (data + outputs go to scratch, not home)
sbatch hpc/benchmark_gpu.slurm
sbatch hpc/benchmark_cpu.slurm

# Engine (from the head node)
bash launch_cluster.sh --mode train --epochs 10 --data-dir /shared/data --run-id engine-train
bash launch_cluster.sh --mode infer --data-dir /shared/data --run-id engine-train \
    --weights-dir results/engine-train/weights

# Compare everything in results/
python -m benchmark.compare --results-dir results --target-acc 0.95

# Try the engine on one machine without a cluster
python -m benchmark.launch_local --num-stages 3 -- --mode train --data-dir ./tmp --limit 256 --epochs 1
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch 5-Node Pipeline (From Head Node `10.0.0.10`)
```bash
bash launch_cluster.sh
```
