# End-to-End Execution Plan: Edge-Deployable Malaria Classification

**Goal:** Train a highly compressed, edge-deployable image classification model to detect *Plasmodium* parasites in thin blood smear images.
**Target Environment:** Train on UNC Charlotte's High Performance Computing (HPC) cluster.
**Deployment Target:** Low-resource offline smartphones (via TensorFlow Lite).

---

## 1. Environment & Infrastructure Specs

**UNC Charlotte HPC Specifications:**
*   **Login Node:** `hpc.charlotte.edu` (Requires NinerNET ID + VPN/Duo).
*   **Scheduler:** Slurm Workload Manager.
*   **Target Partition:** `GPU` (or `Orion` with GPU requests if faculty-aligned). The GPU partition has strict limits: max 12 active GPUs per user, 30-day max job time.
*   **Storage Limit Caution:** Home directories (`/users`) are capped at 500GB and have a strict 5 million file inode limit.
    *   **CRITICAL:** Do *not* extract or store the 27,000+ dataset images in the home directory. All data downloading, extraction, and model saving MUST happen in the user's **scratch directory** (5TB quota).

**Tech Stack Requirements:**
*   **Language:** Python 3.9+
*   **Frameworks:** TensorFlow 2.x, Keras, TensorFlow Datasets (TFDS).
*   **Environment Manager:** Anaconda/Miniconda (loaded via `module load anaconda3` on the cluster).

---

## 2. Data Strategy

*   **Dataset:** NIH Malaria Dataset (National Library of Medicine).
*   **Size:** 27,558 single-cell images (Balanced: 50% Parasitized, 50% Uninfected).
*   **Acquisition:** Do not download manually. The coding agent must use `tfds.load('malaria', split=['train', 'test'])` inside the Python script. TFDS will automatically handle the download and formatting into efficient `tf.data.Dataset` objects, avoiding the inode penalty of storing 27,000 raw loose images on the cluster.

---

## 3. Model Architecture & Training Strategy

*   **Base Model:** `MobileNetV2` (Pre-trained on ImageNet).
    *   *Why:* It is mathematically optimized for mobile processors (depthwise separable convolutions) and achieves high accuracy with a tiny memory footprint.
*   **Input Shape:** `(128, 128, 3)` or `(224, 224, 3)`. Resize the variable-sized TFDS images to a standardized square.
*   **Transfer Learning Pipeline:**
    1.  Load `MobileNetV2` with `include_top=False`.
    2.  Freeze the base model weights.
    3.  Add a `GlobalAveragePooling2D` layer.
    4.  Add a `Dense` layer with a single output node (Sigmoid activation for binary classification).
    5.  **Phase 1:** Train only the custom top layer for ~5-10 epochs (Adam optimizer, relatively high learning rate).
    6.  **Phase 2 (Fine-tuning):** Unfreeze the top layers of MobileNetV2 and train the entire network for another ~10 epochs with a very low learning rate (e.g., `1e-5`).

---

## 4. Instructions for the Coding Agent

Please generate the following two files:

### File 1: `train_malaria.py`
Write a robust, heavily commented Python script that does the following:
1.  **Setup:** Set the TFDS data directory to a path inside `/scratch/your_username/` (use an environment variable or argparse so the user can define it).
2.  **Data Loading:** Use `tfds.load('malaria')`. Implement a preprocessing function to resize images to 128x128 and scale pixel values to `[-1, 1]` (required for MobileNetV2). Use `tf.data` optimizations (`batch()`, `prefetch(tf.data.AUTOTUNE)`, `cache()`).
3.  **Model Building:** Construct the MobileNetV2 transfer learning model as specified in Section 3.
4.  **Training:** Compile with `BinaryCrossentropy` and train the model. Include EarlyStopping callbacks.
5.  **Export & Conversion:** Once training completes, save the standard Keras `.h5` or `.keras` model. **Crucially, immediately instantiate a `tf.lite.TFLiteConverter` and convert the model to a `.tflite` file.** Save this file to the scratch directory.

### File 2: `submit_gpu_job.slurm`
Write a Slurm batch submission script tailored for UNC Charlotte's HPC.
1.  Set `#SBATCH --partition=GPU`.
2.  Request 1 node, a reasonable number of cores (e.g., 4 or 8), and 1 GPU (`--gres=gpu:1`).
3.  Set memory (`--mem=32GB`) and a time limit (`--time=06:00:00`).
4.  Load the required modules (e.g., `module load anaconda3`).
5.  Include placeholders for activating a conda environment.
6.  Execute `python train_malaria.py` ensuring the data directory flags point to the user's scratch space.