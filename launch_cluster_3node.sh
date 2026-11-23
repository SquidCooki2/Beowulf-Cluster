#!/usr/bin/env bash
# ==============================================================================
# 3-Node 1F1B Cluster Launcher (HP ProDesk 400 G5 cluster on 192.168.1.0/24)
# Execute from Head Node: prodesk2 (192.168.1.12)
# Any arguments are forwarded to run_stage.py on every node, e.g.:
#   bash launch_cluster_3node.sh --mode synthetic --no-pretrained
#   bash launch_cluster_3node.sh --mode train --epochs 10 --data-dir /shared/data --run-id engine-3node
# ==============================================================================
set -euo pipefail

SHARED_DIR="/shared/pipe1f1b"
CONFIG="${SHARED_DIR}/config_3node.json"
PYTHON="/shared/venv/bin/python3"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="python3"
fi
ENV_PREFIX="TORCH_HOME=/shared/torch_cache HF_HOME=/shared/hf_cache"
WORKER_ENV_PREFIX="${ENV_PREFIX} HF_HUB_OFFLINE=1"
QUOTED_ARGS=""
if [ "$#" -gt 0 ]; then
    printf -v QUOTED_ARGS '%q ' "$@"
fi

echo "=========================================================="
echo "Launching 3-Node 1F1B Beowulf Cluster (prodesk2 -> prodesk1 -> prodesk3)"
echo "=========================================================="

echo "[+] Starting Worker Stages (2 -> 1) over SSH..."
ssh -o StrictHostKeyChecking=no beowulf@192.168.1.13 \
    "cd ${SHARED_DIR} && ${WORKER_ENV_PREFIX} ${PYTHON} run_stage.py --config ${CONFIG} --stage 2 ${QUOTED_ARGS}" &
PID_2=$!

ssh -o StrictHostKeyChecking=no beowulf@192.168.1.11 \
    "cd ${SHARED_DIR} && ${WORKER_ENV_PREFIX} ${PYTHON} run_stage.py --config ${CONFIG} --stage 1 ${QUOTED_ARGS}" &
PID_1=$!

# Give workers 2 seconds to bind ZeroMQ sockets
sleep 2

echo "[+] Starting Head Node Stage 0 (prodesk2)..."
(cd "${SHARED_DIR}" && TORCH_HOME=/shared/torch_cache HF_HOME=/shared/hf_cache "${PYTHON}" run_stage.py --config "${CONFIG}" --stage 0 "$@")

wait $PID_1 $PID_2
echo "=========================================================="
echo "1F1B Pipeline execution finished successfully across all 3 nodes!"
echo "=========================================================="
