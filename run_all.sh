#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Ensure results directory exists and capture all output to log
mkdir -p ../results
exec > >(tee -a ../results/run_all.log) 2>&1

echo "[1/4] python train_and_plot_3stim.py --load_model --spike"
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_best_model_spiking.pt  --n_eval_episodes 1000 --fig_index 3a2 --heatmap
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_initial_hippo_model.pt  --n_eval_episodes 1000 --fig_index 3a1 --heatmap
python train_and_plot_3stim.py --load_model --spike --delay 100 --load_model --model_path ../data/3stim_best_model_spiking.pt  --n_eval_episodes 1000 --fig_index 4a2 --heatmap
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_rand_model_spiking.pt  --n_eval_episodes 1000 --fig_index 3b2 --heatmap
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_initial_rand_model.pt  --n_eval_episodes 1000 --fig_index 3b1 --heatmap
python train_and_plot_3stim.py --load_model --spike --delay 100 --load_model --model_path ../data/3stim_rand_model_spiking.pt  --n_eval_episodes 1000 --fig_index 4b2
python train_and_plot_3stim.py --load_model --spike --delay 100 --load_model --model_path ../data/3stim_retiming_frozen_hippo.pt  --n_eval_episodes 1000 --fig_index 4b31
python train_and_plot_3stim.py --load_model --spike --delay 100 --load_model --model_path ../data/3stim_retiming_frozen_rand.pt  --n_eval_episodes 1000 --fig_index 4b32
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_hippo_ABfrozen_train.pt  --n_eval_episodes 1000 --fig_index x1
python train_and_plot_3stim.py --load_model --spike --delay 30 --load_model --model_path ../data/3stim_rand_ABfrozen_train.pt  --n_eval_episodes 1000 --fig_index x2


echo "[2/4] python train_and_plot_laps.py"
python train_and_plot_laps.py

echo "[3/4] running plotting scripts in analysis_plot/"
pushd analysis_plot >/dev/null
python svm_time_classification.py
python visualization_of_relative_change.py
python delay_time_analysis.py
popd >/dev/null

# Move figures to results directory (idempotent across reruns)
if [ -d figures ]; then
    rm -rf ../results/figures
    mv figures ../results/
    echo "Figures moved to ../results/figures."
else
    echo "No figures directory found to move."
fi

echo "All done. Full log: ../results/run_all.log"
