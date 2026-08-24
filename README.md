## ssm_timecells

```
train_and_plot_3stim.py      3-stimulus interval discrimination (main training script)
train_landmark_laps.py       lap counting under randomised timing (redesigned task)
eval_lap_counting.py         evaluate a lap checkpoint across lap counts
ssm_observer_1d.py           delay-period analysis used by train_and_plot_3stim.py
agents/                      SSM and LSTM actor-critic cores + HiPPO init
envs/                        int_discrim.py, lap_random.py, lap_landmark.py
utils/                       utils_analysis.py (sorting/decoding), vp.py (Victor-Purpura)
models/                      the three released checkpoints -- see models/README.md
figures/                     the figures, their CSV/NPZ
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

Checkpoints are selected on VP timing score by
default; `--select_on acc` selects on count accuracy, which saturates and is the
weaker criterion.

To evaluate a lap checkpoint across lap counts:

```bash
python eval_lap_counting.py --ckpt models/lap_counting_best.pt \
    --k_min 2 --k_max 16 --n_episodes 300
```


### Released checkpoints

`models/` carries the best model of each family, picked by re-evaluating every
candidate on one common seed rather than trusting the accuracy recorded during
training. `models/README.md` gives the numbers, the full candidate tables and
the script that produced each file.

```bash
python train_and_plot_3stim.py --spike --delay 30 --n_neurons 50 \
    --init_method hippo --load_model --model_path models/3stim_hippo_best.pt \
    --n_eval_episodes 2000
```

### 3) Figures

```bash
cd figures
python scripts/make_all.py        # ~20 s, rebuilds all nine figures
```

See `figures/README.md` for what each figure shows and which
review point it answers.
