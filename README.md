## ssm_timecells — minimal

A stripped-down copy of the project: the two training entry points, the modules
they import, the three plotting scripts, and the selected response figures.
No experiment/sweep drivers, no Slurm launchers, no checkpoints, no run logs.
All `#` comments have been removed from the Python sources; docstrings are kept.

```
train_and_plot_3stim.py      3-stimulus interval discrimination (main training script)
train_and_plot_laps.py       lap counting (train / eval + plots)
basic_lap_state.py           lap-state helpers used by train_and_plot_laps.py
ssm_observer_1d.py           delay-period analysis used by train_and_plot_3stim.py
agents/                      SSM and LSTM actor-critic cores + HiPPO init
envs/                        int_discrim.py, lap_counting.py
utils/utils_analysis.py      unit sorting / decoding helpers
analysis_plot/               three standalone analysis figures
figures/response_selected/   the nine response-letter figures, their CSV/NPZ
                             inputs, and the code that rebuilds them
```

### 1) Environment

Python 3.9+ (tested on 3.12).

```bash
pip install -r requirements.txt
```

For a GPU build of PyTorch, follow the official CUDA-matched instructions.
CPU-only works; the scripts auto-select CUDA when it is available.

`agents/ssm_init.py` builds the HiPPO-LegS initialization through JAX. If the
installed JAX picks up a mismatched CUDA plugin, force it onto the CPU — the
initialization is a one-off and costs nothing there:

```bash
export JAX_PLATFORMS=cpu
```

### 2) Training

3-stimulus interval discrimination:

```bash
python train_and_plot_3stim.py \
  --spike \
  --n_total_episodes 200000 \
  --n_eval_episodes 100 \
  --n_neurons 50 \
  --lr 3e-4 --weight_decay 1e-6 --entropy 0.3 \
  --seed 72 --delay 30 --eval_every 100
```

Other flags: `--layer2` (two SSM layers), `--init_method`, `--freeze_lambda`,
`--freeze_B`, `--save_dir` (default `./training/3stim`), and
`--load_model --model_path <ckpt.pt>` to skip training and only evaluate/plot.

Note: `--spike` is the configuration this task was trained and reported in. The
non-spiking path fails during evaluation (`eval_accuracy_3stim` unpacks four
return values from a forward that returns three) — this is carried over from the
full repo unchanged, not introduced here.

Lap counting:

```bash
python train_and_plot_laps.py \
  --mode train \
  --n_total_episodes 10000 \
  --n_eval_episodes 100 \
  --n_neurons 80 \
  --lr 3e-3 --weight_decay 1e-6 --entropy 0.1 \
  --lap_length 30 --lap_count 4 --eval_every 100
```

Other flags: `--spike`, `--layer2`, `--approx`, `--vary_lap_len`,
`--save_dir` (default `./training/lap_counting`).

Evaluation and plots from an existing checkpoint:

```bash
python train_and_plot_laps.py \
  --mode eval --model_path <ckpt.pt> \
  --n_eval_episodes 100 --n_neurons 80 \
  --lap_length 30 --lap_count 4 --eval_save_dir ./figures/
```

No `.pt` files ship with this copy, so `--mode eval` and `--load_model` need a
checkpoint you trained or supplied yourself; the `--model_path` defaults still
point at the `../data/` layout of the full repo.

### 3) Analysis figures

Self-contained, no local imports; run from the repository root:

```bash
python analysis_plot/svm_time_classification.py
python analysis_plot/visualization_of_relative_change.py
python analysis_plot/delay_time_analysis.py
```

### 4) Response figures

`figures/response_selected/` needs neither checkpoints nor a run tree — the
numbers behind every panel are the CSV/NPZ files in its `data/`:

```bash
cd figures/response_selected
python scripts/make_all.py        # ~20 s, rebuilds all nine figures
```

See `figures/response_selected/README.md` for what each figure shows and which
review point it answers. Its `scripts/provenance/` subfolder is a record of how
`data/` was extracted from the original run artifacts; those extractors need the
full `training/` tree and are not required to rebuild any figure.
