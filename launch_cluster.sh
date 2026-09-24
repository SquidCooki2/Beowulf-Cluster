#!/usr/bin/env bash
# ==============================================================================
# 5-Node 1F1B Cluster Launcher
# Execute from Master / Head Node (10.0.0.10)
# ==============================================================================
set -euo pipefail

SHARED_DIR="/shared/pipe1f1b"

echo "=========================================================="
echo "🚀 Launching 5-Node Heterogeneous 1F1B Beowulf Cluster"
echo "=========================================================="

echo "[+] Starting Worker Stages (4 -> 1) over SSH..."
ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.14 "python3 ${SHARED_DIR}/run_stage.py --stage 4" &
PID_4=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.13 "python3 ${SHARED_DIR}/run_stage.py --stage 3" &
PID_3=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.12 "python3 ${SHARED_DIR}/run_stage.py --stage 2" &
PID_2=$!

ssh -o StrictHostKeyChecking=no clusteradmin@10.0.0.11 "python3 ${SHARED_DIR}/run_stage.py --stage 1" &
PID_1=$!

# Give workers 2 seconds to bind ZeroMQ sockets
sleep 2

echo "[+] Starting Head Node Stage 0..."
python3 ${SHARED_DIR}/run_stage.py --stage 0

wait $PID_1 $PID_2 $PID_3 $PID_4
echo "=========================================================="
echo "✓ 1F1B Pipeline execution finished successfully across all 5 nodes!"
echo "=========================================================="
