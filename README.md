# Beowulf Cluster: Heterogeneous 1F1B Distributed AI Engine

This repository contains the codebase and orchestration tools for a 5-node Beowulf Cluster designed for distributed deep learning training and inference on low-power, heterogeneous office PCs (3x HP ProDesk 400 G5 + 2x Lenovo ThinkCentre M900).

## 🚀 Key Architectural Innovations

1. **Heterogeneous 1F1B (One-Forward-One-Backward) Pipeline Parallelism**:
   - Bounds peak activation RAM per machine to $p=5$ stages.
   - Replaces heavy all-to-all AllReduce gradient broadcasts with lightweight point-to-point boundary activation streaming over ZeroMQ.
2. **Adaptive FLOPS-Proportional Stage Balancing**:
   - Benchmarks node CPU FLOPS at startup.
   - Proportionally allocates layers (e.g. 4-5 layers to 8th/9th Gen HPs, 2-3 layers to 6th Gen Lenovos) to eliminate pipeline stall bubbles.
3. **Agricultural AI Applications**:
   - **Plant Pathology Vision**: 38-class plant leaf disease classification on 54,000+ PlantVillage images.
   - **Crop Yield Regressor**: Neural network predicting metric tons/hectare from environmental/soil metrics.

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
│   └── scheduler.py        # 1F1B state machine (Warmup -> 1F1B Steady State -> Cooldown)
├── models/
│   ├── __init__.py
│   ├── leaf_resnet.py      # 38-Class PlantVillage ResNet-18 partitioned into 5 stages
│   └── crop_regressor.py   # Multi-variable crop yield neural network
├── run_stage.py            # Main node daemon executed on each physical machine
├── launch_cluster.sh       # Bash bootstrap script to launch all 5 nodes over SSH
└── docs/                   # Full research blueprints and presentation documentation
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
