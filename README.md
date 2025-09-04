## ssm_timecells

Minimal setup and usage to train, evaluate, and reproduce plots.

### 1) Environment setup

- Python 3.9+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

If you plan to use GPU with PyTorch, install the CUDA‑matched build from the official instructions.

Note: This project was only tested on a Linux system with CUDA 12.6 and PyTorch 2.6.0.

### 2) Quick run: generate figures and evaluation

This runs with the provided checkpoints and plotting scripts.

```bash
bash run_all.sh
```

Outputs are written under `figures/`.

### 3) Training modes

You can train the two tasks independently.

- 3‑stimulus interval discrimination (with intermediate choice):

```bash
python train_and_plot_3stim.py \
  --n_total_episodes 200000 \
  --n_eval_episodes 100 \
  --n_neurons 50 \
  --lr 3e-4 \
  --weight_decay 1e-6 \
  --entropy 0.3 \
  --seed 72 \
  --delay 30 \
  --eval_every 100
```

Optional flags:
- `--spike`: enable spiking SSM
- `--layer2`: use two SSM layers
- `--save_dir`: training output root (defaults to `./training/3stim`)
- `--load_model --model_path <ckpt.pt>`: skip training and only evaluate/plot

- Lap counting:

```bash
python train_and_plot_laps.py \
  --mode train \
  --n_total_episodes 10000 \
  --n_eval_episodes 100 \
  --n_neurons 80 \
  --lr 3e-3 \
  --weight_decay 1e-6 \
  --entropy 0.1 \
  --lap_length 30 \
  --lap_count 4 \
  --eval_every 100
```

Optional flags:
- `--spike`, `--layer2`, `--approx`
- `--save_dir` for training outputs (defaults to `./training/lap_counting`)

To run evaluation and generate plots for lap counting from an existing checkpoint:

```bash
python train_and_plot_laps.py \
  --mode eval \
  --model_path lap_best_model.pt \
  --n_eval_episodes 100 \
  --n_neurons 80 \
  --lap_length 30 \
  --lap_count 4 \
  --eval_save_dir ./figures/
```

### 4) Notes and tips

- The scripts will auto‑select CUDA if available. For CPU‑only environments, ensure PyTorch CPU wheels are installed.
- Random seeds can be set via the `--seed` flag.
- Figures will be written into `figures/`. Training checkpoints are saved under `training/...`.
- `run_all.sh` assumes the working directory is the repository root and will call the plotting utilities in `analysis_plot/`.

### 5) Known dependencies that may require system packages

- `jax`/`jaxlib` wheels are platform/accelerator specific. If not required (only the 3‑stim path is used), you can omit them from installation.
- `matplotlib-venn` requires `matplotlib`.
- `gym` is used for its `spaces` only; no MuJoCo/Box2D is needed.
