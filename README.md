## ssm_timecells — minimal

A stripped-down copy of the project: the two training entry points, the modules
they import, the three plotting scripts, and the selected response figures.
No experiment/sweep drivers, no Slurm launchers, no run logs.
All `#` comments have been removed from the Python sources; docstrings are kept.

```
train_and_plot_3stim.py      3-stimulus interval discrimination (main training script)
train_landmark_laps.py       lap counting under randomised timing (redesigned task)
eval_lap_counting.py         evaluate a lap checkpoint across lap counts
ssm_observer_1d.py           delay-period analysis used by train_and_plot_3stim.py
agents/                      SSM and LSTM actor-critic cores + HiPPO init
envs/                        int_discrim.py, lap_random.py, lap_landmark.py
utils/                       utils_analysis.py (sorting/decoding), vp.py (Victor-Purpura)
models/                      the three released checkpoints -- see models/README.md
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

Lap counting under randomised timing (the redesigned task -- lap count and lap
length are both drawn fresh every episode):

```bash
python train_landmark_laps.py \
  --mode train --env random --seed 2 \
  --k_range 2 6 \
  --lap_len_range 10 60 \
  --pause_count_range 0 2 --pause_len_range 0 25 \
  --n_total_episodes 20000 \
  --eval_every 2000 --n_eval_episodes 300 \
  --n_neurons 80 \
  --lr 3e-3 --weight_decay 1e-6 --entropy 0.1 \
  --select_on vp \
  --save_dir ./training/lap_random
```

Those are the conditions the released checkpoint was trained under. `--env
landmark` runs the landmark-cued variant instead, and `--mode grid` its
fixed-vs-varied lap-length conditions (landmark only — it will refuse a
`--env random` checkpoint). Checkpoints are selected on VP timing score by
default; `--select_on acc` selects on count accuracy, which saturates and is the
weaker criterion.

To evaluate a lap checkpoint across lap counts:

```bash
python eval_lap_counting.py --ckpt models/lap_counting_best.pt \
    --k_min 2 --k_max 16 --n_episodes 300
```

Defaults reproduce the training distribution, so `K > 6` measures extrapolation
to lap counts never seen. It prints count accuracy, VP timing score, hit/miss
rates and false alarms per episode per K, and `--out results.json` writes the
rows.

This supersedes the earlier fixed-30-step lap task. The old
`train_and_plot_laps.py` / `envs/lap_counting.py` pipeline is not carried here,
and checkpoints from it are not interchangeable with this one -- the two
environments give the same 2-D observation different meanings.

### Released checkpoints

`models/` carries the best model of each family, picked by re-evaluating every
candidate on one common seed rather than trusting the accuracy recorded during
training. `models/README.md` gives the numbers, the full candidate tables and
the provenance path for each file.

```bash
python train_and_plot_3stim.py --spike --delay 30 --n_neurons 50 \
    --init_method hippo --load_model --model_path models/3stim_hippo_best.pt \
    --n_eval_episodes 2000
```

### 3) Analysis figures

Self-contained, no local imports; run from the repository root:

```bash
python analysis_plot/visualization_of_relative_change.py
python analysis_plot/delay_time_analysis.py
```

`analysis_plot/svm_time_classification.py` is also here but cannot run as
shipped: it reads `lap_counting_<seed>_activity.npy`, which only the superseded
`train_and_plot_laps.py` wrote. It needs either that activity file from an
earlier run or a port to the redesigned task.

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
