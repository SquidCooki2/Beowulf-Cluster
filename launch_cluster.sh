#!/usr/bin/env bash
# ==============================================================================
# 5-Node 1F1B Cluster Launcher
# Execute from Master / Head Node (10.0.0.10)
# Any arguments are forwarded to run_stage.py on every node, e.g.:
#   bash launch_cluster.sh --mode train --epochs 10 --data-dir /shared/data --run-id engine-1
#   bash launch_cluster.sh --mode infer --weights-dir /shared/pipe1f1b/results/engine-1/weights --run-id engine-1
# ==============================================================================
set -euo pipefail

SHARED_DIR="/shared/pipe1f1b"
ARGS="$*"

echo "=========================================================="
echo "🚀 Launching 5-Node Heterogeneous 1F1B Beowulf Cluster"
echo "=========================================================="

echo "[+] Starting Worker Stages (4 -> 1) over SSH..."
ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.14 "cd ${SHARED_DIR} && python3 run_stage.py --stage 4 ${ARGS}" &
PID_4=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.13 "cd ${SHARED_DIR} && python3 run_stage.py --stage 3 ${ARGS}" &
PID_3=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.12 "cd ${SHARED_DIR} && python3 run_stage.py --stage 2 ${ARGS}" &
PID_2=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.11 "cd ${SHARED_DIR} && python3 run_stage.py --stage 1 ${ARGS}" &
PID_1=$!

# Give workers 2 seconds to bind ZeroMQ sockets
sleep 2

echo "[+] Starting Head Node Stage 0..."
cd ${SHARED_DIR} && python3 run_stage.py --stage 0 ${ARGS}

wait $PID_1 $PID_2 $PID_3 $PID_4
echo "=========================================================="
echo "✓ 1F1B Pipeline execution finished successfully across all 5 nodes!"
echo "=========================================================="
