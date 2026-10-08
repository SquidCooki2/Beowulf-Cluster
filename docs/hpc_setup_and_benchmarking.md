# HPC Setup & Benchmark Guide (UNC Charlotte)

How to run the malaria model on the UNC Charlotte HPC (CPU and GPU) so it can be compared with the Beowulf engine. Both platforms run the same model code, the same data split and the same recipe. See [Fairness checklist](#fairness-checklist) before you trust any speedup number.

> [!NOTE]
> Partition names, module names and limits below come from the UNC Charlotte HPC docs and the earlier planning notes. Confirm them on the cluster with `sinfo`, `module avail` and `sacctmgr show assoc user=$USER` before the first run, since they change.

## 0. What you need

- A NinerNET ID with HPC access, and VPN + Duo when off campus.
- Access to a GPU partition (`GPU`) and a CPU partition (this repo assumes `Orion`; check with `sinfo`).
- This repository on the cluster.

## 1. Log in and get the code

```bash
ssh <ninernet_id>@hpc.charlotte.edu     # needs VPN + Duo off campus
cd ~
git clone <your repo URL> Beowulf-Cluster
cd Beowulf-Cluster
```

The code is small, so your user directory `/users/<ninernet_id>` is fine for it.

### Storage: Local `./data` vs `/scratch/$USER`

By default, all scripts and Slurm jobs keep datasets and weights directly inside your cloned repo:
* **Dataset:** `Beowulf-Cluster/data/` (specifically `data/malaria_128.npz`, ~1.3GB)
* **Model outputs:** `Beowulf-Cluster/output/`
* **Torch weight cache:** `Beowulf-Cluster/.torch_cache/`

All three are ignored by `.gitignore` so your Git state stays clean.

**Why this is safe in your home/user directory (`/users/<ninernet_id>`):**
UNC Charlotte user directories have a **5 million file (inode) limit** and a 500GB quota. Because our pipeline packs all 27,558 image cells into **a single `.npz` file**, it consumes only 2 files on disk (the downloaded `.zip` and the packed `.npz`), well within your quota.

If you ever wish to redirect datasets or large model checkpoints to scratch, simply set the environment variables `DATA_DIR=/scratch/$USER/data` or pass `--data-dir /scratch/$USER/data`.

## 2. Create the Python environment (once)

```bash
module load anaconda3
conda create -n beowulf python=3.10 -y
source activate beowulf
pip install -r requirements.txt
```

For GPU runs, the PyTorch build must match the cluster's CUDA version. Check it, then confirm on a GPU node (see step 4):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False` on a GPU node, reinstall torch with a CUDA build from <https://pytorch.org/get-started/locally/> that matches `nvidia-smi`'s CUDA version.

> [!TIP]
> The conda environment itself can create many files. If you hit the inode limit, put it in scratch with `conda create -p /scratch/$USER/envs/beowulf ...` and set `CONDA_ENV=/scratch/$USER/envs/beowulf` when you submit.

## 3. Prepare the data (once, on the login node)

The compute nodes may have no internet access, so download on the login node. This downloads the NIH malaria zip (~340 MB) into `Beowulf-Cluster/data/`, resizes the images to 128×128, and packs them into a single `data/malaria_128.npz` file:

```bash
cd ~/Beowulf-Cluster
mkdir -p data output .torch_cache
export TORCH_HOME=.torch_cache

python - <<'EOF'
from data import malaria
from models.registry import get_spec

x, y = malaria.load_arrays("data")              # downloads, packs to data/malaria_128.npz
print("images:", x.shape, "labels:", y.shape, "parasitized:", int((y == 0).sum()))
get_spec("malaria_mobilenet").create_stages(5)  # caches pretrained ImageNet weights
EOF
```

Packing takes a few minutes. If the admins prefer no heavy work on the login node, run the same snippet inside an interactive job instead (`srun --partition=Orion --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash`), but note that compute nodes may have no internet, so do the download part on the login node.

Expected output: `images: (27558, 128, 128, 3)`. A fixed, seeded 80/10/10 train/val/test split is derived from this one file. The same file is copied to the engine's head node (see [step 6](#6-run-the-engine-side)).

## 4. Smoke test (a few minutes)

Always do this before a long job. It uses an interactive GPU session, 256 images and 1 epoch:

```bash
srun --partition=GPU --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
# then, inside the session:
cd ~/Beowulf-Cluster
module load anaconda3 && source activate beowulf
export TORCH_HOME=.torch_cache

python -m training.train_local --device cuda --epochs 1 --limit 256 --workers 2 --run-id smoke
python -m benchmark.infer_local --device cuda --limit 256 --latency-samples 20 --weights-dir output/malaria_mobilenet --run-id smoke
exit
```

You should see per-epoch lines with `img/s` and a `results/smoke/train.json` and `infer.json`. Delete `results/smoke` afterwards.

## 5. Run the benchmarks

From the repo root, submit both jobs. They run the **same recipe** (`--recipe full`, batch 32, lr 1e-4) and differ only in hardware:

```bash
cd ~/Beowulf-Cluster
mkdir -p logs

sbatch hpc/benchmark_gpu.slurm      # 1 GPU, 8 cores
sbatch hpc/benchmark_cpu.slurm      # CPU only, 8 cores (fair comparison with the CPU-only engine)
```

Each job trains for 10 epochs (override with `EPOCHS=5 sbatch ...`), then runs the inference benchmark. Results go to `results/hpc-cuda-<jobid>/` and `results/hpc-cpu-<jobid>/`.

Useful variations:

```bash
EPOCHS=3 sbatch hpc/benchmark_cpu.slurm                                   # shorter run
TRAIN_ARGS="--batch-size 64" sbatch hpc/benchmark_gpu.slurm               # extra trainer flags
RUN_ID=hpc-gpu-run2 sbatch hpc/benchmark_gpu.slurm                        # named run
CONDA_ENV=/scratch/$USER/envs/beowulf sbatch hpc/benchmark_gpu.slurm      # env in scratch
```

Monitor and manage jobs:

```bash
squeue -u $USER                  # what is queued/running
tail -f logs/bench_gpu-<jobid>.out
scancel <jobid>                  # cancel
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State   # after it finishes
```

For fairness, set the CPU job's core count to match your engine nodes, using `#SBATCH --cpus-per-task` in `hpc/benchmark_cpu.slurm`. Each engine node's CPU has its own core count, so the table will report both. The optional `--threads` flag already follows `--cpus-per-task`.

### GPU partition limits

The GPU partition has a maximum of 12 active GPUs per user and a 30-day maximum job time. This benchmark needs 1 GPU and well under 6 hours.

## 6. Run the engine side

On the cluster head node (`10.0.0.10`), with the repo at `/shared/pipe1f1b`:

```bash
# one-time: copy the packed dataset and prefetch weights so stages don't race to download
scp <ninernet_id>@hpc.charlotte.edu:~/Beowulf-Cluster/data/malaria_128.npz /shared/data/
export TORCH_HOME=/shared/torch_cache
python3 -c "from models.registry import get_spec; get_spec('malaria_mobilenet').create_stages(5)"

# training (same recipe: batch 32 = 8 micro-batches x 4, lr 1e-4, full fine-tune)
bash launch_cluster.sh --mode train --epochs 10 --data-dir /shared/data --run-id engine-train --results-dir /shared/pipe1f1b/results

# inference using the engine's own trained weights
bash launch_cluster.sh --mode infer --data-dir /shared/data \
    --weights-dir /shared/pipe1f1b/results/engine-train/weights --run-id engine-train --results-dir /shared/pipe1f1b/results
```

> [!IMPORTANT]
> Every flag is forwarded to all five nodes, and these must be identical everywhere: `--run-id`, `--limit`, `--seed`, `--epochs`, `--data-dir` contents. All stages must run the same mode at the same time.

Quick connectivity test without data: `bash launch_cluster.sh` (default `--mode synthetic`). You can also run the whole engine on one machine: `python -m benchmark.launch_local -- --mode synthetic --no-pretrained`.

## 7. Compare

Copy the HPC `results/hpc-*` folders and the engine's `results/engine-*` folder into one `results/` directory (anywhere), then:

```bash
scp -r <ninernet_id>@hpc.charlotte.edu:~/Beowulf-Cluster/results/hpc-* results/
python -m benchmark.compare --results-dir results --target-acc 0.95
```

This writes `results/summary.md` (a table plus speedups vs the engine) and `results/accuracy_vs_time.png`. Pick `--target-acc` as an accuracy that every platform reaches (check the best validation accuracy column first); the time to reach it is the fairest training metric, because the platforms may converge at slightly different rates.

### What is measured

| Metric | Definition |
|---|---|
| Training throughput | train images/s, median over epochs (higher is better) |
| s/epoch | median wall-clock of train + validation per epoch |
| Time to target | cumulative wall-clock until validation accuracy ≥ target |
| Latency p50 / p95 | batch size 1, per image, from input tensor to prediction (engine: round trip across all stages, on stage 0's clock; HPC: includes host-to-GPU copy) |
| Inference throughput | whole test split at batch 1/8/32, including preprocessing |
| Engine comm cost | per-stage `comm` in the engine JSON: seconds blocked in receive, send time, bytes moved |

## Fairness checklist

- **Same model and weights:** identical stage split, ImageNet pretrained start, frozen BatchNorm.
- **Same data and split:** fixed seeded 80/10/10 split from one packed file; both use the same flips augmentation.
- **Same optimizer settings:** Adam, lr 1e-4, effective batch 32 (the engine accumulates 8 micro-batches of 4 and takes one step), full fine-tuning.
- **Same CPU settings for the CPU run:** match `--cpus-per-task` and report the CPU model (the JSON records it).
- **Warmup excluded from inference timings;** training epochs include everything (first-epoch overheads are why throughput uses the median).
- Run each configuration **at least 3 times** and report the median or spread, since shared HPC nodes are noisy. Use a different `--run-id` per repeat.
- Do not run other heavy jobs on the engine nodes during measurement.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Inode quota warning during training | The packed `data/malaria_128.npz` only takes 2 inodes, but if you create thousands of custom checkpoints, pass `--output-dir /scratch/$USER/output`. |
| `torch.cuda.is_available()` is `False` on the GPU node | Reinstall torch with a CUDA build matching `nvidia-smi`. |
| Job stuck in `PD` (pending) | `squeue -u $USER -o "%i %T %r"` shows the reason; shorten `--time` or reduce resources. |
| Download or `urlretrieve` fails on a compute node | Compute nodes may be offline. Prepare data and weights on the login node (step 3). |
| `PermissionError` on `mobilenet_v2-*.pth` in engine runs | Several stages tried to download at once. Prefetch once (step 6) or use `--no-pretrained` for pure timing runs. |
| Engine stages hang | All five stages must be started with the same mode and counts; check `ip`/port values in `config.json` and that ports 5550-5564 are open between nodes. |
