#!/usr/bin/env bash
set -euo pipefail

# Runs the full workflow in order:
# 1) Data download
# 2) Loss dashboard (background)
# 3) SimCLR pretraining
# 4) Downstream experiments
# 5) Results analysis

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

DASHBOARD_PID=""
cleanup() {
  if [[ -n "${DASHBOARD_PID}" ]] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
    echo "Stopping loss dashboard (pid: ${DASHBOARD_PID})"
    kill "$DASHBOARD_PID" || true
  fi
}
trap cleanup EXIT

echo "[1/5] Downloading data"
python -m src.data_handling.downloader

echo "[2/5] Starting loss dashboard in background"
python -m src.analysis.loss_dashboard > logs/loss_dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Loss dashboard started (pid: ${DASHBOARD_PID}). Logs: logs/loss_dashboard.log"
echo "Dashboard URL is usually: http://localhost:7860"

echo "[3/5] Pretraining SimCLR"
python -m src.experiments.pretrain_simclr "$@"

echo "[4/5] Running downstream experiments"
python -m src.experiments.run_experiments

echo "[5/5] Visualizing/analyzing results"
python -m src.analysis.analyze_results

echo "Workflow complete."

