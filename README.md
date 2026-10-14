# Beowulf Cluster: Heterogeneous Distributed Engine

A Beowulf cluster: defined as a parallel-processing computer cluster built from inexpensive, commodity-grade hardware

This repository contains the codebase and orchestration tools for a 5-node Beowulf Cluster designed for distributed deep learning training and inference on low-power, heterogeneous office PCs (3x HP ProDesk 400 G5 + 2x Lenovo ThinkCentre M900).

## Write-ups coming soon!
We are in the process of development, so full write-ups will come soon!

---

## Repository Structure

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
