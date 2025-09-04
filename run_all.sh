#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/4] python train_and_plot_3stim.py --load_model --spike"
python train_and_plot_3stim.py --load_model --spike

echo "[2/4] python train_and_plot_laps.py"
python train_and_plot_laps.py

echo "[3/4] running plotting scripts in analysis_plot/"
pushd analysis_plot >/dev/null
python plot_angle_distribution.py
python plot_retiming_comparision.py
python plot_unit_activity.py
popd >/dev/null

echo "All done. Check the figures in the figures/ folder."
