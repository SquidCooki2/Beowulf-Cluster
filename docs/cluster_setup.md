# Beowulf Cluster Setup & Operations Guide

Step-by-step instructions to provision, network, configure, and run the 5-node **AgriBeowulf-5Node** heterogeneous cluster (2x Lenovo ThinkCentre M900 + 3x HP ProDesk 400 G5) for distributed 1F1B pipeline training and inference.

---

## 1. Hardware & Network Topology

All 5 nodes are connected through a local Gigabit Ethernet switch on the `10.0.0.0/24` subnet and use the shared user account **`beowulf`**.

| Stage ID | Hostname | Static IP | Hardware Model | Forward Port (`PULL`) | Backward Port (`PULL`) | Default Layers |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **0 (Head)** | `thinkcentre1` | `10.0.0.10` | Lenovo ThinkCentre M900 | `5550` | `5560` | 3 |
| **1** | `prodesk1` | `10.0.0.11` | HP ProDesk 400 G5 | `5551` | `5561` | 4 |
| **2** | `prodesk2` | `10.0.0.12` | HP ProDesk 400 G5 | `5552` | `5562` | 4 |
| **3** | `prodesk3` | `10.0.0.13` | HP ProDesk 400 G5 | `5553` | `5563` | 4 |
| **4 (Last)** | `thinkcentre2` | `10.0.0.14` | Lenovo ThinkCentre M900 | `5554` | `5564` | 3 |

> [!NOTE]
> The two **Lenovo ThinkCentre M900** nodes (`thinkcentre1` and `thinkcentre2`, 3 layers each) are placed at **Stage 0 (Head)** and **Stage 4 (Terminal)** to leave CPU headroom for Stage 0's data loading, augmentation, and NFS/SSH orchestration duties as well as Stage 4's classifier head and loss computation, while the three **HP ProDesk 400 G5** nodes (`prodesk1`–`prodesk3`) handle the 4-layer middle backbone stages (`1–3`). These assignments match `config.json` in the repository root.

---

## 2. OS, Hostnames & Static IP Configuration

Run on **each node** (Ubuntu Server 22.04 / 24.04 LTS recommended, logged in as user **`beowulf`**):

### 2.1 Set the hostname
```bash
# Replace <hostname> with thinkcentre1, prodesk1, prodesk2, prodesk3, or thinkcentre2
sudo hostnamectl set-hostname <hostname>
```

### 2.2 Add cluster host mappings (`/etc/hosts`)
Append the following block to `/etc/hosts` on **all 5 nodes**:
```text
10.0.0.10  thinkcentre1
10.0.0.11  prodesk1
10.0.0.12  prodesk2
10.0.0.13  prodesk3
10.0.0.14  thinkcentre2
```

### 2.3 Configure static IP (Netplan example)
Edit `/etc/netplan/01-netcfg.yaml` (adjust interface name `eth0`/`eno1` via `ip a` and set the node's IP `10.0.0.10`–`10.0.0.14`):
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eno1:
      dhcp4: no
      addresses:
        - 10.0.0.10/24   # Change per node (.10 through .14)
```
Apply with:
```bash
sudo netplan apply
```

### 2.4 Open firewall ports for ZeroMQ & NFS
If `ufw` is enabled, allow traffic across the cluster subnet (`10.0.0.0/24`), specifically TCP ports `5550–5564` for ZeroMQ tensor transport:
```bash
sudo ufw allow from 10.0.0.0/24
```

---

## 3. Passwordless SSH from Head Node (`thinkcentre1`)

`launch_cluster.sh` launches Worker Stages `4 -> 1` in the background over non-interactive SSH from `thinkcentre1` (`10.0.0.10`).

On **`thinkcentre1` (`10.0.0.10`)** as user `beowulf`:
```bash
# 1. Generate an SSH keypair (press Enter for no passphrase)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# 2. Copy the public key to all worker nodes (and head node itself)
for ip in 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14; do
    ssh-copy-id beowulf@$ip
done

# 3. Verify passwordless login works across all nodes
for ip in 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14; do
    ssh -o StrictHostKeyChecking=no beowulf@$ip "hostname && uptime"
done
```

---

## 4. Shared NFS Directory (`/shared`)

Both `launch_cluster.sh` and `run_stage.py` expect a shared directory mounted at **`/shared`** across all 5 nodes containing:
- `/shared/pipe1f1b` — the cloned `Beowulf-Cluster` repository
- `/shared/data` — the packed dataset (`malaria_128.npz`)
- `/shared/torch_cache` — cached pretrained ImageNet weights

### 4.1 On the Head Node (`thinkcentre1` / `10.0.0.10` — NFS Server)
```bash
sudo apt update && sudo apt install -y nfs-kernel-server
sudo mkdir -p /shared/pipe1f1b /shared/data /shared/torch_cache
sudo chown -R beowulf:beowulf /shared

# Export /shared to the 10.0.0.0/24 cluster subnet
echo "/shared 10.0.0.0/24(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl enable --now nfs-kernel-server
```

### 4.2 On Every Worker Node (`prodesk1`–`prodesk3`, `thinkcentre2` / `10.0.0.11`–`10.0.0.14` — NFS Clients)
```bash
sudo apt update && sudo apt install -y nfs-common
sudo mkdir -p /shared

# Mount immediately and persist in /etc/fstab
echo "10.0.0.10:/shared /shared nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
sudo mount -a
```

Verify on each worker that `ls -la /shared` displays `pipe1f1b`, `data`, and `torch_cache`.

---

## 5. Clone the Repo & Install Python Dependencies

### 5.1 Clone the codebase into `/shared/pipe1f1b` (on `thinkcentre1`)
```bash
cd /shared/pipe1f1b
git clone <your-repo-url> .
```
Because `/shared/pipe1f1b` is on NFS, all 5 nodes immediately see the exact same code, `config.json`, and `results/` folder.

### 5.2 Install Python packages on ALL 5 nodes
`launch_cluster.sh` invokes `python3 run_stage.py` on every node. Install Python 3 and the required CPU-only PyTorch stack on **all 5 nodes** (you can run this from `thinkcentre1` in a loop):

```bash
for ip in 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14; do
    ssh beowulf@$ip "sudo apt update && sudo apt install -y python3 python3-pip && \
        pip3 install --upgrade pip && \
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
        pip3 install -r /shared/pipe1f1b/requirements.txt pillow"
done
```

> [!TIP]
> If you use a virtual environment (e.g., `/shared/venv`) instead of system `python3`, make sure `python3` in non-interactive SSH sessions resolves to that environment or update `python3` in `launch_cluster.sh`.

---

## 6. Prepare Dataset & Prefetch Pretrained Weights (Once on `thinkcentre1`)

To prevent all 5 stages from simultaneously attempting to download ImageNet weights or unpack the dataset on first launch, run this one-time preparation step on **`thinkcentre1` (`10.0.0.10`)**:

```bash
cd /shared/pipe1f1b
export TORCH_HOME=/shared/torch_cache
export HF_HOME=/shared/hf_cache

python3 - <<'EOF'
from data import malaria, scienceqa
from models.registry import get_spec

# 1a. Malaria Dataset (~340 MB download -> /shared/data/malaria_128.npz, ~1.3 GB)
x, y = malaria.load_arrays("/shared/data")
print("Malaria ready -> images:", x.shape, "labels:", y.shape)
get_spec("malaria_mobilenet").create_stages(5, pretrained=True)

# 1b. ScienceQA Multimodal Dataset (~250 MB -> /shared/data/scienceqa_128.npz) & SmolVLM-256M
sx, sy = scienceqa.load_arrays("/shared/data")
print("ScienceQA ready -> images:", sx.shape, "labels:", sy.shape)
get_spec("scienceqa_smolvlm").create_stages(5, pretrained=True)
print("Pretrained weights cached successfully.")
EOF
```

Make `TORCH_HOME=/shared/torch_cache` and `HF_HOME=/shared/hf_cache` available to non-interactive SSH shells on all nodes by adding them to `/etc/environment`:
```bash
for ip in 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14; do
    ssh beowulf@$ip 'printf "TORCH_HOME=/shared/torch_cache\nHF_HOME=/shared/hf_cache\n" | sudo tee -a /etc/environment'
done
```

> [!NOTE]
> During real training/inference, **only Stage 0 (`thinkcentre1`, `10.0.0.10`) loads the ~1.3 GB image array into RAM**, while the terminal stage (`thinkcentre2`, `10.0.0.14`) loads only the lightweight label array (`images=False`). Both use deterministic seeded index permutations so data and labels stay aligned without sending labels over the network.

---

## 7. Verifying & Running the Cluster

All cluster jobs are launched from **`thinkcentre1` (`10.0.0.10`)** inside `/shared/pipe1f1b`.

### Step 1: Single-Machine Local Smoke Test (Optional, No SSH/Network Needed)
Before testing across all 5 physical machines, you can verify the code locally on `127.0.0.1`:
```bash
cd /shared/pipe1f1b
python3 -m benchmark.launch_local -- --mode synthetic --no-pretrained
```

### Step 2: 5-Node Synthetic Connectivity Self-Test
Runs 1 iteration of random synthetic micro-batches (`8 x 4`) forward and backward across all 5 physical nodes over ZeroMQ (`5550–5554` and `5560–5564`):
```bash
cd /shared/pipe1f1b
bash launch_cluster.sh --mode synthetic --no-pretrained
```
You should see every stage bind its `PULL` sockets, connect its `PUSH` sockets, complete the 1F1B Warmup → Steady State → Cooldown cycle, and exit cleanly with:
`✓ 1F1B Pipeline execution finished successfully across all 5 nodes!`

### Step 3: Distributed 1F1B Training on Real Data
Train the sliced `malaria_mobilenet` or `scienceqa_smolvlm` (`SmolVLM-256M-Instruct`) model across all 5 nodes:
```bash
cd /shared/pipe1f1b

# Option A: Malaria MobileNetV2
bash launch_cluster.sh \
    --model malaria_mobilenet \
    --mode train \
    --epochs 10 \
    --data-dir /shared/data \
    --results-dir /shared/pipe1f1b/results \
    --run-id engine-malaria

# Option B: ScienceQA SmolVLM-256M (Vision-Language Model)
bash launch_cluster.sh \
    --model scienceqa_smolvlm \
    --mode train \
    --epochs 10 \
    --data-dir /shared/data \
    --results-dir /shared/pipe1f1b/results \
    --run-id engine-smolvlm
```
**Outputs saved to `/shared/pipe1f1b/results/<run-id>/`:**
- `train_stage0.json` … `train_stage4.json` (per-epoch wall-clock timing on Stage 0, communication wait/send stats on every stage, validation/test accuracy on Stage 4, and peak RSS memory).
- `weights/stage0.pt` … `weights/stage4.pt` (trained model weights for each stage).

> [!IMPORTANT]
> Any flags passed to `launch_cluster.sh` are forwarded to all 5 nodes. Flags that control iteration counts or splits (`--model`, `--mode`, `--epochs`, `--limit`, `--seed`, `--run-id`, `--infer-batch-sizes`) are automatically kept identical across all nodes when launched via `launch_cluster.sh`.

### Step 4: Distributed Pipelined Inference Benchmark
Measure batch-1 round-trip latency (`p50`/`p95`/`p99` via end-to-end acks on Stage 0's clock) and pipelined throughput across batch sizes (`1, 8, 32`):
```bash
cd /shared/pipe1f1b

# Option A: Malaria MobileNetV2
bash launch_cluster.sh \
    --model malaria_mobilenet \
    --mode infer \
    --data-dir /shared/data \
    --weights-dir /shared/pipe1f1b/results/engine-malaria/weights \
    --results-dir /shared/pipe1f1b/results \
    --run-id engine-malaria

# Option B: ScienceQA SmolVLM-256M
bash launch_cluster.sh \
    --model scienceqa_smolvlm \
    --mode infer \
    --data-dir /shared/data \
    --weights-dir /shared/pipe1f1b/results/engine-smolvlm/weights \
    --results-dir /shared/pipe1f1b/results \
    --run-id engine-smolvlm
```
**Outputs saved to `/shared/pipe1f1b/results/engine-train/`:**
- `infer_stage0.json` … `infer_stage4.json`

### Step 5: Generate Benchmark Comparison Report & Plots
To summarize the cluster run (and optionally compare against HPC CPU/GPU runs placed in `results/`):
```bash
cd /shared/pipe1f1b
python3 -m benchmark.compare --results-dir results --target-acc 0.95
```
This merges `train_stage*.json` and `infer_stage*.json` and writes:
- `results/summary.md`
- `results/accuracy_vs_time.png`

---

## 8. Re-Balancing Layers for Heterogeneous Nodes (Optional)

If you swap hardware or want to verify the FLOPS ratio between the **HP ProDesk 400 G5** and **Lenovo ThinkCentre M900** nodes, run the built-in FLOPS benchmark on each node type:

```bash
python3 - <<'EOF'
from engine.partitioner import HeteroPartitioner
gflops = HeteroPartitioner.benchmark_local_flops(matrix_dim=2048, iterations=5)
print(f"Measured GFLOPS: {gflops:.2f}")
EOF
```
You can then pass a dictionary of `{stage_id: gflops}` into `HeteroPartitioner(num_layers=18).compute_layer_distribution(node_gflops)` to compute the optimal layer split.

---

## 9. Troubleshooting

| Symptom | Cause & Fix |
| :--- | :--- |
| **Stages hang after `Bound Forward PULL socket`** | One or more stages failed to start, or firewall rules are blocking TCP ports `5550–5564`. Check `sudo ufw status` and verify all 5 nodes can reach each other on `10.0.0.10–14`. |
| **`Address already in use` (`zmq.error.ZMQError`)** | A previous `run_stage.py` process is still running on a worker node. Clean up across all nodes from `thinkcentre1`:<br>`for ip in 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14; do ssh beowulf@$ip "pkill -f run_stage.py || true"; done` |
| **`FileNotFoundError: /shared/data/malaria_128.npz`** | The NFS share `/shared` is not mounted on one of the worker nodes (`sudo mount -a`), or Step 6 (Dataset preparation) has not been run yet. |
| **`PermissionError` on `mobilenet_v2-*.pth`** | Multiple stages tried to download ImageNet weights to the shared cache at the exact same time. Prefetch weights once on `thinkcentre1` (Step 6) or pass `--no-pretrained` for timing-only runs. |
| **SSH prompts for password or hangs during `launch_cluster.sh`** | Ensure `ssh-copy-id` was completed for `beowulf@10.0.0.11` through `10.0.0.14` from `thinkcentre1` (Step 3). |
